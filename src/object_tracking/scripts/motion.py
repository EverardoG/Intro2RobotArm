#!/usr/bin/python3
# Standard libraries
from typing import Tuple, Any

# ROS libraries
import rospy

# Arm fpv libraries
from object_tracking.srv import SetTarget

class MotionNode():
    def __init__(self) -> None:
        """Initialize arm motion node."""
        rospy.init_node('motion_node')
        rospy.loginfo("Arm motion code Init")

        # Setup service for updating internal target
        self.set_target_srv = rospy.Service('/motion/set_target', SetTarget, self.set_target)
        # Setup service proxy for relaying target to perception node
        self.set_target_proxy = rospy.ServiceProxy('/perception/set_target', SetTarget)

        pass

    def set_target(self, msg) -> Tuple[bool, str]:
        # Relay the new target to the perception node
        try:
            success = self.set_target_proxy(msg).success
        # Catch the error if service fails
        except rospy.ServiceException as exc:
            success = False
            rospy.logerr("Perception set_target service did not process request: "+str(exc))
        # Return whether setting target was sucessful
        return success, "motion_set_target"

    def run(self) -> None:
        """Run the arm motion node"""
        while not rospy.is_shutdown():
            rospy.spin()
        return None

if __name__ == '__main__':
    motion_node = MotionNode()
    motion_node.run()