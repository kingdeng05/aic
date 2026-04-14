#!/usr/bin/python3
"""
Episode reset node — exposes a /episode_reset service that resets the sim.
Uses a minimal rclpy node ONLY for the service server + delete/spawn clients.
Controller/joints/tare calls go through a helper script to avoid Zenoh conflicts.
"""

import os
import subprocess
import time
import math
import json
import sys

os.environ["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
os.environ["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from simulation_interfaces.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from ament_index_python.packages import get_package_share_directory


HOME_JOINT_POSITIONS = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

DEFAULT_TASK_BOARD_POSE = {
    "x": 0.15, "y": -0.2, "z": 1.14,
    "roll": 0.0, "pitch": 0.0, "yaw": 3.1415,
}

DEFAULT_CABLE_POSE = {
    "x": 0.172, "y": 0.024, "z": 1.518,
    "roll": 0.4432, "pitch": -0.48, "yaw": 1.3303,
}

# Path to the helper script that runs ros2 service calls in a clean process
# Use the source path directly since the installed path follows symlinks
HELPER_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "_reset_helper.py"
)


def euler_to_quaternion(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


class EpisodeResetNode(Node):
    def __init__(self):
        super().__init__("episode_reset_node")
        cb_group = ReentrantCallbackGroup()

        self.declare_parameter("cable_type", "sfp_sc_cable")
        self.declare_parameter("attach_cable_to_gripper", True)
        self.declare_parameter("nic_card_mount_0_present", True)
        self.declare_parameter("nic_card_mount_0_translation", 0.0)
        self.declare_parameter("nic_card_mount_0_yaw", 0.0)
        # If True, apply a random Cartesian translation offset after homing.
        self.declare_parameter("randomize_start_pose", False)
        # Max absolute offset per axis in meters (uniform [-x, +x] in x/y/z).
        self.declare_parameter("random_offset_m", 0.06)
        for key, val in DEFAULT_TASK_BOARD_POSE.items():
            self.declare_parameter(f"task_board_{key}", val)
        for key, val in DEFAULT_CABLE_POSE.items():
            self.declare_parameter(f"cable_{key}", val)

        self.delete_entity_client = self.create_client(
            DeleteEntity, "/gz_server/delete_entity", callback_group=cb_group
        )
        self.spawn_entity_client = self.create_client(
            SpawnEntity, "/gz_server/spawn_entity", callback_group=cb_group
        )

        self.spawned_cable_name = "cable_0"
        self.spawned_task_board_name = "task_board"

        self.reset_service = self.create_service(
            Trigger, "/episode_reset", self.handle_reset, callback_group=cb_group
        )
        self.get_logger().info("Episode reset node ready.")

    def _call_service(self, client, request, name, timeout=30.0):
        future = client.call_async(request)
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > timeout:
                self.get_logger().error(f"{name} timed out")
                return None
            time.sleep(0.05)
        return future.result()

    def _run_helper(self, command, timeout=60):
        """Run helper script in a separate process to avoid Zenoh session conflicts.

        `command` may include args separated by spaces, e.g. "home_random 0.03".
        """
        self.get_logger().info(f"Running helper: {command}...")
        env = os.environ.copy()
        env["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
        env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
        try:
            result = subprocess.run(
                ["/usr/bin/python3", HELPER_SCRIPT, *command.split()],
                capture_output=True, text=True, timeout=timeout, env=env
            )
            self.get_logger().info(f"Helper stdout: {result.stdout.strip()}")
            if result.returncode != 0:
                self.get_logger().error(f"Helper stderr: {result.stderr.strip()[-300:]}")
                return False
            return True
        except Exception as e:
            self.get_logger().error(f"Helper exception: {e}")
            return False

    def _delete_entity(self, name):
        req = DeleteEntity.Request()
        req.entity = name
        result = self._call_service(self.delete_entity_client, req, f"delete({name})")
        if result and result.result.result == 1:
            self.get_logger().info(f"Deleted {name}")
            return True
        self.get_logger().warn(f"Delete {name} failed")
        return False

    def _run_xacro(self, xacro_file, args):
        cmd = ["xacro", xacro_file] + [f"{k}:={v}" for k, v in args.items()]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return r.stdout if r.returncode == 0 else None
        except Exception:
            return None

    def _spawn_entity(self, name, resource_string, pose_dict):
        req = SpawnEntity.Request()
        req.name = name
        req.allow_renaming = True
        req.resource_string = resource_string
        q = euler_to_quaternion(pose_dict["roll"], pose_dict["pitch"], pose_dict["yaw"])
        req.initial_pose = PoseStamped()
        req.initial_pose.header.frame_id = "world"
        req.initial_pose.pose = Pose(
            position=Point(x=pose_dict["x"], y=pose_dict["y"], z=pose_dict["z"]),
            orientation=q,
        )
        result = self._call_service(self.spawn_entity_client, req, f"spawn({name})")
        if result and result.result.result == 1:
            self.get_logger().info(f"Spawned: {result.entity_name}")
            return result.entity_name
        return None

    def _spawn_task_board(self):
        desc = get_package_share_directory("aic_description")
        pose = {k: self.get_parameter(f"task_board_{k}").value for k in DEFAULT_TASK_BOARD_POSE}
        xacro_args = {
            **{k: str(v) for k, v in pose.items()},
            "nic_card_mount_0_present": str(self.get_parameter("nic_card_mount_0_present").value).lower(),
            "nic_card_mount_0_translation": str(self.get_parameter("nic_card_mount_0_translation").value),
            "nic_card_mount_0_yaw": str(self.get_parameter("nic_card_mount_0_yaw").value),
        }
        sdf = self._run_xacro(f"{desc}/urdf/task_board.urdf.xacro", xacro_args)
        if not sdf:
            return False
        name = self._spawn_entity("task_board", sdf, pose)
        if name:
            self.spawned_task_board_name = name
        return name is not None

    def _spawn_cable(self):
        desc = get_package_share_directory("aic_description")
        xacro_args = {
            "cable_type": self.get_parameter("cable_type").value,
            "attach_cable_to_gripper": str(self.get_parameter("attach_cable_to_gripper").value).lower(),
        }
        sdf = self._run_xacro(f"{desc}/urdf/cable.sdf.xacro", xacro_args)
        if not sdf:
            return False
        pose = {k: self.get_parameter(f"cable_{k}").value for k in DEFAULT_CABLE_POSE}
        name = self._spawn_entity("cable_0", sdf, pose)
        if name:
            self.spawned_cable_name = name
        return name is not None

    def handle_reset(self, request, response):
        self.get_logger().info("=== Episode reset requested ===")
        try:
            # Step 1: Delete entities first (cable is attached to gripper)
            self.get_logger().info(f"Deleting: {self.spawned_cable_name}, {self.spawned_task_board_name}")
            self._delete_entity(self.spawned_cable_name)
            time.sleep(1.0)
            self._delete_entity(self.spawned_task_board_name)
            time.sleep(1.0)

            # Step 2: Home robot (deactivate, reset joints, reactivate, hold position, tare)
            # Done in a separate process to avoid Zenoh conflicts
            randomize = self.get_parameter("randomize_start_pose").value
            if randomize:
                offset = self.get_parameter("random_offset_m").value
                cmd = f"home_random {offset}"
            else:
                cmd = "home"
            if not self._run_helper(cmd):
                response.success = False
                response.message = "Failed to home robot"
                return response

            # Step 3: Respawn task board
            self.get_logger().info("Respawning task board...")
            if not self._spawn_task_board():
                response.success = False
                response.message = "Failed to spawn task board"
                return response
            time.sleep(1.0)

            # Step 4: Respawn cable
            self.get_logger().info("Respawning cable...")
            if not self._spawn_cable():
                response.success = False
                response.message = "Failed to spawn cable"
                return response

            response.success = True
            response.message = "Episode reset complete"
            self.get_logger().info("=== Episode reset complete ===")
        except Exception as e:
            self.get_logger().error(f"Reset failed: {e}")
            response.success = False
            response.message = str(e)
        return response


def main():
    rclpy.init()
    node = EpisodeResetNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
