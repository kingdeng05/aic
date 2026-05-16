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


# Stable target enumeration. MUST stay in lockstep with TASK_ID_TABLE in
# aic_robot_aic_controller.py — re-ordering or inserting breaks every trained
# checkpoint's one-hot encoding. Append-only.
TASK_ID_TABLE: list[tuple[str, str]] = [
    ("nic_card_mount_0", "sfp_port_0"),
    ("nic_card_mount_0", "sfp_port_1"),
    ("nic_card_mount_1", "sfp_port_0"),
    ("nic_card_mount_1", "sfp_port_1"),
    ("nic_card_mount_2", "sfp_port_0"),
    ("nic_card_mount_2", "sfp_port_1"),
    ("nic_card_mount_3", "sfp_port_0"),
    ("nic_card_mount_3", "sfp_port_1"),
    ("nic_card_mount_4", "sfp_port_0"),
    ("nic_card_mount_4", "sfp_port_1"),
    ("sc_port_0",        "sc_port"),
    ("sc_port_1",        "sc_port"),
]
TASK_ID_DIM: int = len(TASK_ID_TABLE)
TASK_ID_INDEX: dict[tuple[str, str], int] = {k: i for i, k in enumerate(TASK_ID_TABLE)}
# Aliases: the portal's sample_config trial_3 sends port_name="sc_port_base"
# while our table key uses "sc_port". Map both spellings to the same SC index
# so SC trials produce a real one-hot, not an all-zero fallback.
TASK_ID_INDEX[("sc_port_0", "sc_port_base")] = TASK_ID_INDEX[("sc_port_0", "sc_port")]
TASK_ID_INDEX[("sc_port_1", "sc_port_base")] = TASK_ID_INDEX[("sc_port_1", "sc_port")]


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
        # Dual-head load: NIC head + SC head. Head is selected per-task in
        # insert_cable() based on task.target_module_name.
        # -------------------------------------------------------------------------
        nic_path = Path(os.environ.get(
            "AIC_ACT_MODEL_PATH_NIC",
            "/home/sai/.cache/huggingface/lerobot/models/outputs/train/nic_multislot_200_merged_trimmed_rot6d_slim/checkpoints/200000/pretrained_model",
        ))
        sc_path = Path(os.environ.get(
            "AIC_ACT_MODEL_PATH_SC",
            "/home/sai/ws_aic_challenge/src/aic/models/sc_overnight3_cart6d_cs25",
        ))
        self.heads = {
            "nic": self._load_head("NIC", nic_path),
            "sc":  self._load_head("SC",  sc_path),
        }
        # Default until first task arrives — picked properly in insert_cable().
        self.active = self.heads["nic"]

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

        # Ablation: when AIC_ACT_BLANK_IMAGES=1, feed black frames to all cameras to
        # test whether the policy is relying on visual cues vs. proprioceptive state.
        self.blank_images = os.environ.get("AIC_ACT_BLANK_IMAGES", "") == "1"
        if self.blank_images:
            self.get_logger().warning(
                "AIC_ACT_BLANK_IMAGES=1 — feeding zero (black) images to the policy."
            )

    def _load_head(self, label: str, policy_path: Path) -> dict:
        """Load one ACT head (policy weights + normalization stats) into a dict."""
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]
        config = draccus.decode(ACTConfig, config_dict)

        policy = ACTPolicy(config)
        policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        policy.eval()
        policy.to(self.device)

        obs_stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        action_stats = load_file(
            policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )

        def gobs(key, shape):
            return obs_stats[key].to(self.device).view(*shape)

        def gact(key, shape):
            return action_stats[key].to(self.device).view(*shape)

        img_stats = {
            side: {
                "mean": gobs(f"observation.images.{side}_camera.mean", (1, 3, 1, 1)),
                "std":  gobs(f"observation.images.{side}_camera.std",  (1, 3, 1, 1)),
            }
            for side in ("left", "center", "right")
        }
        state_mean = gobs("observation.state.mean", (1, -1))
        state_std = gobs("observation.state.std", (1, -1))
        # Replace zero std with 1.0 to avoid div-by-zero -> NaN on constant
        # channels (e.g. NIC-only or SC-only one-hot indices). Matches lerobot's
        # NormalizerProcessor behavior.
        state_std = torch.where(state_std == 0, torch.ones_like(state_std), state_std)
        action_mean = gact("action.mean", (1, -1))
        action_std = gact("action.std", (1, -1))

        self.get_logger().info(f"[{label}] ACT loaded on {self.device} from {policy_path}")
        self.get_logger().info(f"[{label}] state_mean shape={tuple(state_mean.shape)}  action_mean shape={tuple(action_mean.shape)}")

        return {
            "policy": policy,
            "img_stats": img_stats,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }

    def _blank_image_tensor(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Normalized zero (black) image tensor at the model's expected shape."""
        zeros = torch.zeros(1, 3, 256, 288, device=self.device)
        return (zeros - mean) / std

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
        """Convert ROS Observation message into dictionary of normalized tensors,
        using the currently-active head's normalization stats (set in insert_cable)."""

        img_stats = self.active["img_stats"]

        # --- Process Cameras ---
        if self.blank_images:
            obs = {
                "observation.images.left_camera": self._blank_image_tensor(
                    img_stats["left"]["mean"], img_stats["left"]["std"]
                ),
                "observation.images.center_camera": self._blank_image_tensor(
                    img_stats["center"]["mean"], img_stats["center"]["std"]
                ),
                "observation.images.right_camera": self._blank_image_tensor(
                    img_stats["right"]["mean"], img_stats["right"]["std"]
                ),
            }
        else:
            obs = {
                "observation.images.left_camera": self._img_to_tensor(
                    obs_msg.left_image,
                    self.device,
                    self.image_scaling,
                    img_stats["left"]["mean"],
                    img_stats["left"]["std"],
                ),
                "observation.images.center_camera": self._img_to_tensor(
                    obs_msg.center_image,
                    self.device,
                    self.image_scaling,
                    img_stats["center"]["mean"],
                    img_stats["center"]["std"],
                ),
                "observation.images.right_camera": self._img_to_tensor(
                    obs_msg.right_image,
                    self.device,
                    self.image_scaling,
                    img_stats["right"]["mean"],
                    img_stats["right"]["std"],
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

        # Append 12-D task one-hot (set in insert_cable() before loop entry).
        # Total obs.state -> 24-D, matching the *_rot6d_slim training schema.
        state_np = np.concatenate([state_np, self._task_one_hot])

        # Normalize State using the active head's stats.
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.active["state_mean"]) / self.active["state_std"]

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        # --- Head selection: NIC vs SC based on target_module_name ---
        mod = task.target_module_name
        if mod.startswith("nic_card_mount_"):
            self.active = self.heads["nic"]; head_label = "NIC"
        elif mod.startswith("sc_port_"):
            self.active = self.heads["sc"];  head_label = "SC"
        else:
            self.get_logger().warning(
                f"unknown target_module {mod!r}; defaulting to NIC head"
            )
            self.active = self.heads["nic"]; head_label = "NIC"
        self.get_logger().info(f"selected {head_label} head for task {task.id}")
        self.active["policy"].reset()

        # Encode (target_module_name, port_name) -> 12-D one-hot. All-zero fallback
        # if the pair isn't in TASK_ID_TABLE — policy will see no task signal, log
        # a warning so eval-time misconfig is visible.
        target_key = (task.target_module_name, task.port_name)
        self._task_one_hot = np.zeros(TASK_ID_DIM, dtype=np.float32)
        if target_key in TASK_ID_INDEX:
            idx = TASK_ID_INDEX[target_key]
            self._task_one_hot[idx] = 1.0
            self.get_logger().info(f"task one-hot index {idx} for {target_key}")
        else:
            self.get_logger().warning(
                f"target {target_key!r} not in TASK_ID_TABLE; sending all-zero one-hot"
            )

        start_time = time.time()

        while time.time() - start_time < 120.0:
            loop_start = time.time()

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # 2. Model Inference (use the active head selected at task start)
            with torch.inference_mode():
                # shape [1, 9]: [px, py, pz, rot6d.0..5]
                normalized_action = self.active["policy"].select_action(obs_tensors)

            # 3. Un-normalize Action using the active head's action stats.
            raw_action_tensor = (normalized_action * self.active["action_std"]) + self.active["action_mean"]
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
