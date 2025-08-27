#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class MovingObstacle:
    def __init__(self):
        rospy.init_node("moving_obstacle_controller", anonymous=True)

        self.obj_type = rospy.get_param("~type", "unknown")
        self.model_name = rospy.get_param("~model_name")
        self.start_y = rospy.get_param("~start_y")
        self.end_y = rospy.get_param("~end_y")
        self.x = rospy.get_param("~x")
        self.z = rospy.get_param("~z")
        self.velocity = rospy.get_param("~velocity")
        self.update_rate = rospy.get_param("~update_rate")

        self.current_y = self.start_y
        self.direction = -1

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.rate = rospy.Rate(self.update_rate)

        rospy.loginfo(f"Started controller for model: {self.model_name} (type: {self.obj_type}) with velocity: {self.velocity}")

    def run(self):
        while not rospy.is_shutdown():
            self.current_y += self.direction * self.velocity / self.update_rate

            if self.current_y >= self.start_y:
                self.direction = -1
            elif self.current_y <= self.end_y:
                self.direction = 1

            state_msg = ModelState()
            state_msg.model_name = self.model_name
            state_msg.pose.position.x = self.x
            state_msg.pose.position.y = self.current_y
            state_msg.pose.position.z = self.z
            state_msg.pose.orientation.w = 1.0
            state_msg.twist.linear.y = self.direction * self.velocity

            try:
                self.set_model_state(state_msg)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

            self.rate.sleep()

if __name__ == "__main__":
    try:
        obstacle = MovingObstacle()
        obstacle.run()
    except rospy.ROSInterruptException:
        pass
