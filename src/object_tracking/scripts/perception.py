#!/usr/bin/python3
#  Standard libraries
from typing import Tuple, Any
import math
import cv2
import numpy as np

# ROS libaries
import rospy
from sensor_msgs.msg import Image

# Arm fpv libraries
from sensor.msg import Led
from armpi_fpv import Misc
from object_tracking.srv import SetTarget

class ArmPerceptionNode():
    def __init__(self) -> None:
        """Initialize arm perception node."""
        rospy.loginfo("Arm perception code Init")
        rospy.init_node('perception_node')
        # Subscriber for raw images
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        # Publisher for processed images visualizing what perception is doing
        self.image_pub = rospy.Publisher('/perception/image_result', Image, queue_size=1)  # register result image publisher
        # Publisher for LED commands
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)

        # Get lab range from ros param server
        self.color_range = rospy.get_param('/lab_config_manager/color_range_list', {})

        # Service for setting up the target color for the perception code
        self.set_target_srv = rospy.Service('/perception/set_target', SetTarget, self.set_target)
        # Frame size
        self.size = (320, 240)
        # Target color
        self._target_color = None
        # Colors for LED
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }
        return None

    def set_target(self, msg):
        """Set a target color. Change LED color to correspond to target color."""
        rospy.loginfo("%s", msg)
        self._target_color = msg.data
        led = Led()
        led.index = 0
        led.rgb.r = self.range_rgb[self._target_color][2]
        led.rgb.g = self.range_rgb[self._target_color][1]
        led.rgb.b = self.range_rgb[self._target_color][0]
        self.rgb_pub.publish(led)
        led.index = 1
        self.rgb_pub.publish(led)
        rospy.sleep(0.1)
        return [True, 'set_target']

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
        processed_frame, area_max, center_x, center_y = self.process_frame(frame)
        # rospy.loginfo(f"area_max: {area_max}")
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB).tostring()

        # Populate image message with the new image and publish it
        image_msg.data = rgb_image
        self.image_pub.publish(image_msg)
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

    def process_frame(self, frame: np.array)->Tuple[np.array, int, int, float]:
        """Process the frame. Take in a raw frame, detect the cube in the frame,
        and return the processed frame and parameters for describing the cube
        position in the frame."""
        # Make an image copy and grab the dimensions
        img_copy = frame.copy()
        img_h, img_w = frame.shape[:2]

        # Draw a crosshair in the center of the frame
        cv2.line(frame, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
        cv2.line(frame, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)

        # Resize the frame and convert it to LAB space
        frame_resize = cv2.resize(img_copy, self.size, interpolation=cv2.INTER_NEAREST)
        frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间

        area_max = 0
        area_max_contour = 0

        # _target_color - Current color the robot is looking for
        # color_range is the ranges for specific colors in the LAB space
        # If the current color the robot is looking for is included in the color ranges
        if self._target_color in self.color_range:
            # target color range is the color range for the specific target color
            target_color_range = self.color_range[self._target_color]
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
            # If the radius is above some threshold (too big?), publish the image as is
            if radius < 100:
                # Draw a circle around the target color
                cv2.circle(frame, (int(center_x), int(center_y)), int(radius), self.range_rgb[self._target_color], 2)

        # Return the processed frame along with parameters describing perceived object
        return frame, area_max, center_x, center_y

if __name__ == '__main__':
    perception_node = ArmPerceptionNode()
    perception_node.run()