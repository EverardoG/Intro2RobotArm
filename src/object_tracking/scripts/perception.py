#!/usr/bin/python3
#  Standard libraries
from typing import Tuple, Any
import math
import cv2
import numpy as np

# ROS libaries
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

# Arm fpv libraries
from sensor.msg import Led
from armpi_fpv import Misc
from armpi_fpv import apriltag
from object_tracking.srv import SetTarget

class ArmPerceptionNode():
    def __init__(self) -> None:
        """Initialize arm perception node."""
        rospy.init_node('perception_node')
        rospy.loginfo("Arm perception code Init")

        # Get lab range from ros param server
        self.color_range = rospy.get_param('/lab_config_manager/color_range_list', {})
        # Frame size
        self.size = (320, 240)
        # Target - either color or april tag
        self._target = None
        # April tag detector
        self.detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        # List of tags
        self.tags = ['tag1', 'tag2', 'tag3']
        # Colors for LED
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'tag1' : (255, 255, 0),
            'tag2' : (255, 0, 255),
            'tag3' : (0, 255, 255)
        }
        # Start counter for publishing points
        self._point_count = 0
        # Publisher for processed images visualizing what perception is doing
        self.image_pub = rospy.Publisher('/perception/image_result', Image, queue_size=1)  # register result image publisher
        # Publisher for LED commands
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
        # Publisher for target points
        self.point_pub = rospy.Publisher('/perception/target_info', PointStamped, queue_size=1)
        # Service for setting up the target color for the perception code
        self.set_target_srv = rospy.Service('/perception/set_target', SetTarget, self.set_target)
        # Subscriber for raw images
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        return None

    def set_target(self, msg):
        """Set a target. Change LED color to correspond to target color, or corrseponding color if target is arcuo tag."""
        rospy.loginfo("%s", msg)
        if msg.data in self.tags or msg.data in self.range_rgb:
            self._target = msg.data
            led = Led()
            led.index = 0
            led.rgb.r = self.range_rgb[self._target][2]
            led.rgb.g = self.range_rgb[self._target][1]
            led.rgb.b = self.range_rgb[self._target][0]
            self.rgb_pub.publish(led)
            led.index = 1
            self.rgb_pub.publish(led)
            rospy.sleep(0.1)
            return [True, 'set_target']
        else:
            rospy.logerr(f"\'{msg.data}\' is not a valid target")
            return [False, 'set_target']

    def run(self) -> None:
        """Run the arm perception node"""
        while not rospy.is_shutdown():
            rospy.spin()
        return None

    def image_callback(self, image_msg: Image) -> None:
        """Take the raw image, process it, and publish the processed image"""
        # Convert the image message into an open cv image
        image = np.ndarray(shape=(image_msg.height, image_msg.width, 3), dtype=np.uint8,
                       buffer=image_msg.data)  # 将自定义图像消息转化为图像 (Convert custom image messages to images)
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process the frame to get an rgb image with sensing shown on it
        # Also get parameters for controller
        frame = cv2_img.copy()
        processed_frame, center_x, center_y, angle_or_area_max = self.process_frame(frame)
        # rospy.logdebug(f"area_max: {area_max}")
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB).tostring()

        # Populate image message with the new image and publish it
        image_msg.data = rgb_image
        self.image_pub.publish(image_msg)

        # If something was detected in the frame, publish its position
        if center_x is not None and center_y is not None:
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.get_rostime()
            point_msg.header.seq = self._point_count
            point_msg.header.frame_id = "camera_view"
            point_msg.point.x = center_x
            point_msg.point.y = center_y
            point_msg.point.z = angle_or_area_max
            self.point_pub.publish(point_msg)
            self._point_count += 1

        return None

    @staticmethod
    def getAreaMaxContour(contours: Any)->Tuple[Any, float]:
        """Helper function for image processing"""
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None

        for c in contours:  # 历遍所有轮廓
            contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                if contour_area_temp > 10:  # 只有在面积大于300时，最大面积的轮廓才是有效的，以过滤干扰
                    area_max_contour = c

        return area_max_contour, contour_area_max  # 返回最大的轮廓

    def detect_color(self, frame: np.array)->Tuple[np.array, int, int]:
        """Detect the target color in the frame. Return the center x and y if color is target is found."""
        # Grab frame dimensions
        img_h, img_w = frame.shape[:2]

        # Draw a crosshair in the center of the frame
        cv2.line(frame, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
        cv2.line(frame, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)

        # Resize the frame and convert it to LAB space
        frame_resize = cv2.resize(frame, self.size, interpolation=cv2.INTER_NEAREST)
        frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间

        area_max = 0
        area_max_contour = 0

        # target color range is the color range for the specific target color
        target_color_range = self.color_range[self._target]
        frame_mask = cv2.inRange(frame_lab, tuple(target_color_range['min']), tuple(target_color_range['max']))  # 对原图像和掩模进行位运算 (Bitwise operations on the original image and mask)
        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀 (corrosion)
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀 (swell)
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓 (find the outline)
        area_max_contour, area_max = self.getAreaMaxContour(contours)  # 找出最大轮廓 (find the largest contour)

        center_x = None
        center_y = None
        # If the maxmimum area found is above a threshold
        if area_max > 100:  # 有找到最大面积 (have found the largest area)
            (center_x, center_y), radius = cv2.minEnclosingCircle(area_max_contour)  # 获取最小外接圆 (Get the smallest circumcircle)
            center_x = int(Misc.map(center_x, 0, self.size[0], 0, img_w))
            center_y = int(Misc.map(center_y, 0, self.size[1], 0, img_h))
            radius = int(Misc.map(radius, 0, self.size[0], 0, img_w))
            # If the radius is too large, this indicates the target is too big to be valid and must be a false positive
            if radius < 100:
                # Draw a circle around the target color
                cv2.circle(frame, (int(center_x), int(center_y)), int(radius), self.range_rgb[self._target], 2)
            else:
                center_x, center_y, area_max = None, None, None

        return frame, center_x, center_y, area_max

    def detect_tag(self, frame: np.array)->Tuple[np.array, int, int]:
        """Detect the target tag in the frame. Return the center x and y and angle if tag is found."""
        # Make the image grayscale and run apriltag detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, return_image=False)

        center_x = None
        center_y = None
        angle = None

        # If the detector detected a tag...
        if len(detections) != 0:
            # Go through each detection
            for detection in detections:
                # Grab the id to determine if it's the tag being searched for
                tag_id = int(detection.tag_id)
                if self.tags[tag_id-1] == self._target:
                    # Draw the contours around the corners of the april tag
                    corners = np.rint(detection.corners)  # 获取四个角点 (get four corners)
                    cv2.drawContours(frame, [np.array(corners, np.int)], -1, self.range_rgb[self._target], 2)

                    # Get the position and orientation (just one angle) of the detected tag
                    center_x, center_y = int(detection.center[0]), int(detection.center[1])  # 中心点 (center point)
                    angle = int(math.degrees(math.atan2(corners[0][1] - corners[1][1], corners[0][0] - corners[1][0])))  # 计算旋转角 (Calculate the rotation angle)

                    # Put text where the tag was detected
                    cv2.putText(frame, str(tag_id), (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.range_rgb[self._target], 2)

        return frame, center_x, center_y, angle

    def process_frame(self, frame: np.array)->Tuple[np.array, int, int, int]:
        """Process the frame. Take in a raw frame, detect the cube in the frame,
        and return the processed frame and parameters for describing the cube
        pose in the frame."""

        # _target - Current target the robot is looking for
        # color_range is the ranges for specific colors in the LAB space
        # If the current target the robot is looking for is included in the color ranges,
        # then detect the color
        if self._target in self.color_range:
            processed_frame, center_x, center_y, area_max = self.detect_color(frame)
            angle_or_area_max = area_max
        # If the target is one of the tags, then detect the tag
        elif self._target in self.tags:
            processed_frame, center_x, center_y, angle = self.detect_tag(frame)
            angle_or_area_max = angle
        # Otherwise don't detect anything
        else:
            processed_frame = frame.copy()
            center_x = None
            center_y = None
            angle_or_area_max = None

        # Return the processed frame along with parameters describing perceived object
        return processed_frame, center_x, center_y, angle_or_area_max

if __name__ == '__main__':
    perception_node = ArmPerceptionNode()
    perception_node.run()