#!/usr/bin/python3
# Standard libraries
from typing import Tuple, Any

# External libraries
from kinematics import ik_transform

# ROS libraries
import rospy
from geometry_msgs.msg import PointStamped

# Hiwonder libraries
from object_tracking.srv import SetTarget
from armpi_fpv import PID, bus_servo_control
from hiwonder_servo_msgs.msg import MultiRawIdPosDur

class MotionNode():
    def __init__(self) -> None:
        """Initialize arm motion node."""
        rospy.init_node('motion_node')
        rospy.loginfo("Arm motion code Init")
        # Target for motion
        self._target = None
        # Frame size for motion calculations
        self.size = (640, 480)
        # PID Controllers for tracking
        # For default Hiwonder code: x_pid I=0.005
        self.x_pid = PID.PID(P=0.06, I=0.0, D=0)
        self.y_pid = PID.PID(P=0.00001, I=0, D=0)
        self.z_pid = PID.PID(P=0.00003, I=0, D=0)
        # Inverse kinematics solver
        self.ik = ik_transform.ArmIK()
        # Variables for distance control
        self.Z_DIS = 0.2
        self.x_dis = 500
        self.y_dis = 0.167
        self.z_dis = self.Z_DIS
        # Variables for valid targets
        self.valid_colors = ['red', 'green', 'blue']
        self.valid_tags = ['tag1', 'tag2', 'tag3']
        self.valid_targets = self.valid_colors + self.valid_tags
        # Keep track of last time a message was recieved
        self._last_msg_stamp = None
        # Setup service for updating internal target
        self.set_target_srv = rospy.Service('/motion/set_target', SetTarget, self.set_target)
        # Setup service proxy for relaying target to perception node
        self.set_target_proxy = rospy.ServiceProxy('/perception/set_target', SetTarget)
        # Subscriber for target points
        self.point_sub = rospy.Subscriber('/perception/target_info', PointStamped, self.track_target)
        # Publisher for servo positions
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        pass

    def set_target(self, msg) -> Tuple[bool, str]:
        """Set target for motion. Relay target to perception node."""
        # Check if the requested target is valid
        if msg.data not in self.valid_targets:
            success=False
            rospy.logerr(f"Target was not valid: {msg.data}. Valid targets are: {self.valid_targets}")
        # Relay the requested target to the perception node if target is valid
        else:
            try:
                success = self.set_target_proxy(msg).success
                self._target = msg.data
            # Catch the error if service proxy fails
            except rospy.ServiceException as exc:
                success = False
                rospy.logerr("Perception set_target service did not process request: "+str(exc))
        # Return whether setting target was sucessful
        return success, "motion_set_target"

    # def track_color(self, msg: PointStamped) -> None:
    #     """Command motors so robot keeps color target in center of frame"""
        # # Unpack message
        # self._last_msg_stamp = msg.header.stamp
        # center_x = msg.point.x
        # center_y = msg.point.y
        # area_max = msg.point.z

        # img_w, img_h = self.size
        # self.x_pid.SetPoint = img_w / 2.0  # 设定
        # self.x_pid.update(center_x)  # 当前
        # rospy.loginfo(f"self.x_pid.SetPoint: {self.x_pid.SetPoint} | center_x: {center_x}")
        # dx = self.x_pid.output
        # self.x_dis += int(dx)  # 输出

        # self.x_dis = 200 if self.x_dis < 200 else self.x_dis
        # self.x_dis = 800 if self.x_dis > 800 else self.x_dis

        # self.y_pid.SetPoint = 900  # 设定
        # if abs(area_max - 900) < 50:
        #     area_max = 900
        # self.y_pid.update(area_max)  # 当前
        # dy = self.y_pid.output
        # self.y_dis += dy  # 输出
        # self.y_dis = 0.12 if self.y_dis < 0.12 else self.y_dis
        # self.y_dis = 0.25 if self.y_dis > 0.25 else self.y_dis

        # self.z_pid.SetPoint = img_h / 2.0
        # self.z_pid.update(center_y)
        # dy = self.z_pid.output
        # self.z_dis += dy

        # self.z_dis = 0.22 if self.z_dis > 0.22 else self.z_dis
        # self.z_dis = 0.17 if self.z_dis < 0.17 else self.z_dis

        # target = self.ik.setPitchRanges((0, round(self.y_dis, 4), round(self.z_dis, 4)), -90, -85, -95)
        # if target:
        #     servo_data = target[1]
        #     bus_servo_control.set_servos(self.joints_pub, 20, (
        #         (3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5']), (6, self.x_dis)))
        # return None

    def track_target(self, msg: PointStamped) -> None:
        """Command motors so that robot keeps target in center of frame"""
        if self._target in self.valid_targets:
            # Unpack message
            self._last_msg_stamp = msg.header.stamp
            center_x = msg.point.x
            center_y = msg.point.y
            area_max = msg.point.z

            img_w, img_h = self.size
            self.x_pid.SetPoint = img_w / 2.0  # 设定
            self.x_pid.update(center_x)  # 当前
            # rospy.loginfo(f"self.x_pid.SetPoint: {self.x_pid.SetPoint} | center_x: {center_x}")
            dx = self.x_pid.output
            self.x_dis += int(dx)  # 输出

            self.x_dis = 200 if self.x_dis < 200 else self.x_dis
            self.x_dis = 800 if self.x_dis > 800 else self.x_dis

            # Only update y if target is a color
            if self._target in self.valid_colors:
                self.y_pid.SetPoint = 900  # 设定
                if abs(area_max - 900) < 50:
                    area_max = 900
                self.y_pid.update(area_max)  # 当前
                dy = self.y_pid.output
                self.y_dis += dy  # 输出
                self.y_dis = 0.12 if self.y_dis < 0.12 else self.y_dis
                self.y_dis = 0.25 if self.y_dis > 0.25 else self.y_dis

            self.z_pid.SetPoint = img_h / 2.0
            self.z_pid.update(center_y)
            dy = self.z_pid.output
            self.z_dis += dy

            self.z_dis = 0.22 if self.z_dis > 0.22 else self.z_dis
            self.z_dis = 0.17 if self.z_dis < 0.17 else self.z_dis

            target = self.ik.setPitchRanges((0, round(self.y_dis, 4), round(self.z_dis, 4)), -90, -85, -95)
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(self.joints_pub, 20, (
                    (3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5']), (6, self.x_dis)))
        else:
            rospy.logerr("Invalid target encountered in track_target()")
        return None

    def run(self) -> None:
        """Run the arm motion node"""
        while not rospy.is_shutdown():
            rospy.spin()
        return None

if __name__ == '__main__':
    motion_node = MotionNode()
    motion_node.run()