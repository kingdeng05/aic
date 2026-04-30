#
#  Copyright (C) 2026 Intrinsic Innovation LLC
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

import os
import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Callable, Dict, Any, List
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        policy_path = Path(os.environ.get(
            "AIC_ACT_MODEL_PATH",
            # "/home/fuheng/ws_aic/src/aic/outputs/train/act_cable_insertion_v5/checkpoints/100000/pretrained_model",
            # "/home/fuheng/ws_aic/src/aic/outputs/train/cheatcode-nic-30/checkpoints/100000/pretrained_model",
            # "/home/fuheng/ws_aic/src/aic/outputs/train/act_cable_insertion_v7/checkpoints/100000/pretrained_model",
            "/home/fuheng/ws_aic/src/aic/outputs/train/act_cable_insertion_v8/checkpoints/060000/pretrained_model",
        ))

        # Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)

        # Load Policy Architecture & Weights
        self.policy = ACTPolicy(config)
        model_weights_path = policy_path / "model.safetensors"
        self.policy.load_state_dict(load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

        # -------------------------------------------------------------------------
        # 2. Normalization Stats Loading
        # -------------------------------------------------------------------------
        # Observation normalization stats (preprocessor)
        obs_stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        # Action denormalization stats (postprocessor)
        action_stats = load_file(
            policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )

        # Helper to extract and shape stats for broadcasting
        def get_obs_stat(key, shape):
            return obs_stats[key].to(self.device).view(*shape)

        def get_action_stat(key, shape):
            return action_stats[key].to(self.device).view(*shape)

        # Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
        self.img_stats = {
            "left": {
                "mean": get_obs_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_obs_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_obs_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_obs_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_obs_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_obs_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }
        print(f"Image stats: {self.img_stats}")

        # Robot State Stats (1, 26)
        self.state_mean = get_obs_stat("observation.state.mean", (1, -1))
        self.state_std = get_obs_stat("observation.state.std", (1, -1))
        print(f"Robot state mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

        # Action Stats (1, 6) - Used for Un-normalization
        self.action_mean = get_action_stat("action.mean", (1, -1))
        self.action_std = get_action_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

        # Service clients for episode prep. Re-tare so the wrench input matches
        # the training distribution (training data was recorded with tared wrench).
        # Force JOINT target mode so joint_motion_updates aren't silently dropped
        # if the controller was left in CARTESIAN by a reset helper.
        self._tare_client = parent_node.create_client(
            Trigger, "/aic_controller/tare_force_torque_sensor"
        )
        self._change_target_mode_client = parent_node.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        self.get_logger().info("Normalization statistics loaded successfully.")

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Converts ROS Image -> Resized -> Permuted -> Normalized Tensor."""
        # 1. Bytes to Numpy (H, W, C)
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        # 2. Resize
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # 3. To Tensor -> Permute (HWC -> CHW) -> Float -> Div(255) -> Batch Dim
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # 4. Normalize (Apply Mean/Std)
        # Formula: (x - mean) / std
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        """Convert ROS Observation message into dictionary of normalized tensors."""

        # --- Process Cameras ---
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        # --- Process Robot State ---
        # 32-dim state matching the training adapter (see aic_robot_aic_controller.py).
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        # obs_msg.joint_states.position is in CONTROLLER order from aic_adapter:
        # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]
        # But the dataset was recorded from raw /joint_states topic in ALPHABETICAL order:
        # [elbow, gripper, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3]
        # Reorder to match training data.
        ctrl = obs_msg.joint_states.position
        joints_alpha = [
            ctrl[2],  # elbow
            ctrl[6],  # gripper
            ctrl[1],  # shoulder_lift
            ctrl[0],  # shoulder_pan
            ctrl[3],  # wrist_1
            ctrl[4],  # wrist_2
            ctrl[5],  # wrist_3
        ]

        # /fts_broadcaster/wrench is RAW (pre-tare). Training data subtracts
        # controller_state.fts_tare_offset; mirror that here so inference state
        # ≈0 in free space matches the training distribution.
        raw_w = obs_msg.wrist_wrench.wrench
        tare_w = obs_msg.controller_state.fts_tare_offset.wrench

        state_np = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # TCP Linear Vel (3)
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                # TCP Angular Vel (3)
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                # TCP Error (6)
                *obs_msg.controller_state.tcp_error,
                # Joint Positions (7) in alphabetical order to match training
                *joints_alpha,
                # Tared wrench (6): force xyz, torque xyz
                raw_w.force.x - tare_w.force.x,
                raw_w.force.y - tare_w.force.y,
                raw_w.force.z - tare_w.force.z,
                raw_w.torque.x - tare_w.torque.x,
                raw_w.torque.y - tare_w.torque.y,
                raw_w.torque.z - tare_w.torque.z,
            ],
            dtype=np.float32,
        )

        # Normalize State
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        # Force JOINT mode. aic_model caches its own _target_mode and may skip
        # its switch service call if it thinks we're already in JOINT, so we
        # call directly to make sure the controller actually accepts joint cmds.
        if self._change_target_mode_client.wait_for_service(timeout_sec=2.0):
            req = ChangeTargetMode.Request()
            req.target_mode.mode = TargetMode.MODE_JOINT
            self._change_target_mode_client.call(req)

        # Re-tare FTS so the policy's wrench input is centered like the training
        # distribution. Training data subtracted controller_state.fts_tare_offset;
        # if no tare call has happened this episode, that offset is zero and the
        # policy sees the raw 20+N gravity load — far OOD.
        if self._tare_client.wait_for_service(timeout_sec=2.0):
            self._tare_client.call(Trigger.Request())
            self.get_logger().info("FTS tared at episode start.")
        else:
            self.get_logger().warn("Tare service unavailable; proceeding un-tared.")

        start_time = time.time()

        # Run inference for 30 seconds
        while time.time() - start_time < 120.0:
            loop_start = time.time()

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, 7] (first action of chunk) - 7 joint positions
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            # Formula: (norm * std) + mean
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean

            # 4. Extract joint positions
            # raw_action_tensor is [1, 7] in alphabetical joint order:
            # [elbow, gripper/left_finger, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3]
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Joint target (alpha order): {action}")

            # Reorder to controller order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            # (drop gripper joint at index 1)
            controller_positions = [
                float(action[3]),  # shoulder_pan
                float(action[2]),  # shoulder_lift
                float(action[0]),  # elbow
                float(action[4]),  # wrist_1
                float(action[5]),  # wrist_2
                float(action[6]),  # wrist_3
            ]

            joint_motion_update = self.make_joint_motion_update(controller_positions)
            move_robot(joint_motion_update=joint_motion_update)
            send_feedback("in progress...")

            # Maintain control rate to match training data (30Hz loop = 33ms sleep)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.0333 - elapsed))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True

    def make_joint_motion_update(self, positions: List[float]) -> JointMotionUpdate:
        msg = JointMotionUpdate()

        target_state = JointTrajectoryPoint()
        target_state.positions = positions
        target_state.velocities = [0.0] * len(positions)
        msg.target_state = target_state

        msg.target_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        msg.target_damping = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        return msg
