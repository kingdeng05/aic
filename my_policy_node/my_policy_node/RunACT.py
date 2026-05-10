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
from typing import Dict
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from geometry_msgs.msg import Pose, Point, Quaternion

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt decode (Zhou et al.). (6,) -> (3, 3)."""
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w). Shepperd's method."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


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
            # "/home/sai/.cache/huggingface/lerobot/checkpoints/080000/pretrained_model",
            "/home/sai/.cache/huggingface/lerobot/models/outputs/train/nic_card_mount_0_merged_trimmed_rot6d_slim/checkpoints/020000/pretrained_model"
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

        self.state_mean = get_obs_stat("observation.state.mean", (1, -1))
        self.state_std = get_obs_stat("observation.state.std", (1, -1))
        print(f"Robot state mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

        self.action_mean = get_action_stat("action.mean", (1, -1))
        self.action_std = get_action_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

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
        # Slim 12-dim state: tcp_velocity (linear+angular) + tared wrench (force+torque).
        tcp_vel = obs_msg.controller_state.tcp_velocity

        # /fts_broadcaster/wrench is RAW (pre-tare). Training data subtracts
        # controller_state.fts_tare_offset; mirror that here so inference state
        # ≈0 in free space matches the training distribution.
        raw_w = obs_msg.wrist_wrench.wrench
        tare_w = obs_msg.controller_state.fts_tare_offset.wrench

        state_np = np.array(
            [
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
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

        start_time = time.time()

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
                # shape [1, 9]: [px, py, pz, rot6d.0..5]
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean
            action = raw_action_tensor[0].cpu().numpy()

            # 4. Decode Cartesian pose: [px,py,pz, rot6d] -> (position, quaternion)
            pos = action[:3].astype(np.float64)
            R = _rot6d_to_matrix(action[3:9].astype(np.float64))
            qx, qy, qz, qw = _matrix_to_quat_xyzw(R)

            self.get_logger().info(
                f"TCP target: pos={pos.tolist()}  quat(xyzw)=[{qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f}]"
            )

            pose = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=Quaternion(
                    x=float(qx), y=float(qy), z=float(qz), w=float(qw)
                ),
            )
            self.set_pose_target(move_robot, pose, frame_id="base_link")
            send_feedback("in progress...")

            # Maintain control rate to match training data (30Hz loop = 33ms sleep)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.0333 - elapsed))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True
