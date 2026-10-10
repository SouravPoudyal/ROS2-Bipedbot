#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped


class SimpleController(Node):

    def __init__(self):
        super().__init__("simple_controller")
        # Defaults match the bipedbot: tyre radius 0.037 m, wheel centres at x = +/-0.06 m
        self.declare_parameter("wheel_radius", 0.037)
        self.declare_parameter("wheel_separation", 0.12)

        self.wheel_radius_ = self.get_parameter("wheel_radius").get_parameter_value().double_value
        self.wheel_separation_ = self.get_parameter("wheel_separation").get_parameter_value().double_value

        self.get_logger().info("Using wheel radius %.4f" % self.wheel_radius_)
        self.get_logger().info("Using wheel separation %.4f" % self.wheel_separation_)

        self.wheel_cmd_pub_ = self.create_publisher(
            Float64MultiArray, "simple_velocity_controller/commands", 10)
        self.vel_sub_ = self.create_subscription(
            TwistStamped, "bipedbot_controller/cmd_vel", self.velCallback, 10)

        # [v, w]^T = M [w_right, w_left]^T
        self.speed_conversion_ = np.array([
            [self.wheel_radius_ / 2, self.wheel_radius_ / 2],
            [self.wheel_radius_ / self.wheel_separation_, -self.wheel_radius_ / self.wheel_separation_]])
        self.inv_conversion_ = np.linalg.inv(self.speed_conversion_)
        self.get_logger().info("The conversion matrix is %s" % self.speed_conversion_)

    def velCallback(self, msg):
        # Differential-drive kinematics: given v and w, compute the wheel speeds (rad/s)
        robot_speed = np.array([[msg.twist.linear.x],
                                [msg.twist.angular.z]])
        wheel_speed = self.inv_conversion_ @ robot_speed   # [[w_right], [w_left]]

        wheel_speed_msg = Float64MultiArray()
        # Order must match "joints:" in bipedbot_controllers.yaml -> [left, right]
        wheel_speed_msg.data = [float(wheel_speed[1, 0]), float(wheel_speed[0, 0])]
        self.wheel_cmd_pub_.publish(wheel_speed_msg)


def main():
    rclpy.init()
    simple_controller = SimpleController()
    rclpy.spin(simple_controller)
    simple_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()