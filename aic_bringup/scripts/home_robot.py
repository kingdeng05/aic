#!/usr/bin/env python3

#
#  Copyright (C) 2025 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys
import time
import rclpy
import numpy as np
from rclpy.executors import ExternalShutdownException

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
    TargetMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, Point, Quaternion, Wrench, Vector3
from std_srvs.srv import Trigger


class HomeTrajectoryNode(Node):
    def __init__(self):
        super().__init__("home_trajectory_node")
        self.get_logger().info("HomeTrajectoryNode started")

        # Declare parameters.
        self.use_aic_control = self.declare_parameter("use_aic_controller", True).value
        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value
        self.home_joint_positions = [0.6, -1.3, -1.9, -1.57, 1.57, 0.6]

        # Cartesian TCP target (in base_link frame). Defaults preserve the
        # historical hard-coded values so a manual `ros2 run` invocation still
        # behaves as before. The launch file overrides these for the
        # randomized-home flow.
        self.home_x = self.declare_parameter("home_x", -0.4).value
        self.home_y = self.declare_parameter("home_y", 0.2).value
        self.home_z = self.declare_parameter("home_z", 0.3).value
        self.home_qx = self.declare_parameter("home_qx", -0.707).value
        self.home_qy = self.declare_parameter("home_qy", -0.707).value
        self.home_qz = self.declare_parameter("home_qz", 0.0).value
        self.home_qw = self.declare_parameter("home_qw", 0.0).value

        # How long and how fast to republish the pose target. The impedance
        # controller treats the latest received target as its setpoint, but
        # publishing for a brief window ensures the message is received past
        # transient subscriber delays and lets the arm settle.
        self.publish_duration_s = self.declare_parameter(
            "publish_duration_s", 2.0
        ).value
        self.publish_rate_hz = self.declare_parameter("publish_rate_hz", 50.0).value
        self.tare_after = self.declare_parameter("tare_after", True).value
        # Create publisher if needed.
        if self.use_aic_control:
            # Change to pose target mode, in case it was in joint target mode previously
            change_target_mode_client = self.create_client(
                ChangeTargetMode, f"/{self.controller_namespace}/change_target_mode"
            )
            while not change_target_mode_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for change_target_mode service...")
            target_mode_request = ChangeTargetMode.Request()
            target_mode_request.target_mode.mode = TargetMode.MODE_CARTESIAN
            future = change_target_mode_client.call_async(target_mode_request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if not response.success:
                self.get_logger().error("Unable to set target mode")
                rclpy.shutdown()
                return
            self.get_logger().info("Set target mode to CARRTESIAN")

            self.publisher = self.create_publisher(
                MotionUpdate, f"/{self.controller_namespace}/pose_commands", 10
            )

            while self.publisher.get_subscription_count() == 0:
                self.get_logger().info(
                    f"Waiting for subscriber to '{self.controller_namespace}/pose_commands'..."
                )
                time.sleep(1.0)

        else:
            self.action_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/joint_trajectory_controller/follow_joint_trajectory",
            )
            while not self.action_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {self.action_client._action_name}")

        # `main()` invokes send_trajectory() directly; no timer needed. (The
        # original code created a 1 s timer that fired send_trajectory once
        # too, but with the rate-publish + tare code below the timer can
        # re-fire while we're inside spin_until_future_complete and recurse
        # into the callback — "Executor is already spinning".)

    def _tare_force_torque(self):
        client = self.create_client(
            Trigger, f"/{self.controller_namespace}/tare_force_torque_sensor"
        )
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "tare_force_torque_sensor service unavailable; skipping FT tare"
            )
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done() and future.result() is not None and future.result().success:
            self.get_logger().info("Tared FT sensor at home pose")
        else:
            self.get_logger().warn("Tare call did not complete cleanly")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return
        self.get_logger().info("Home trajectory goal accepted")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        rclpy.shutdown()

    def send_trajectory(self):
        if self.use_aic_control:
            msg = MotionUpdate()
            msg.header.frame_id = "base_link"
            msg.pose = Pose(
                position=Point(x=self.home_x, y=self.home_y, z=self.home_z),
                orientation=Quaternion(
                    x=self.home_qx, y=self.home_qy, z=self.home_qz, w=self.home_qw
                ),
            )
            msg.target_stiffness = np.diag(
                [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
            ).flatten()
            msg.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
            msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
            msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

            # Republish at rate for duration so the controller actually settles
            # past transient subscriber delays. A single publish often races
            # the controller's first read.
            period = 1.0 / max(float(self.publish_rate_hz), 1e-3)
            end_time = time.monotonic() + float(self.publish_duration_s)
            n_published = 0
            while time.monotonic() < end_time:
                msg.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(msg)
                time.sleep(period)
                n_published += 1
            self.get_logger().info(
                f"Published {n_published} pose targets over "
                f"{self.publish_duration_s:.1f}s "
                f"(target=({self.home_x:.4f}, {self.home_y:.4f}, {self.home_z:.4f}))"
            )

            if self.tare_after:
                self._tare_force_torque()
        else:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            home_point = JointTrajectoryPoint()
            home_point.positions = self.home_joint_positions
            home_point.time_from_start.sec = 1
            goal.trajectory.points.append(home_point)
            self.send_goal_future = self.action_client.send_goal_async(goal)
            self.send_goal_future.add_done_callback(self.goal_response_callback)


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = HomeTrajectoryNode()
            node.send_trajectory()
            if node.use_aic_control:
                # Keep alive for a short duration to ensure message delivery.
                rclpy.spin_once(node, timeout_sec=2.0)
                rclpy.shutdown()
            else:
                rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main(sys.argv)
