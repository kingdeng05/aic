#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""CheatCode-as-teleoperator: scripted trajectory source for aic_record.

Runs the same approach -> descend -> hold schedule as
aic_example_policies.ros.CheatCode.insert_cable, but as a lerobot
Teleoperator that emits one PoseTargetActionDict per tick. An OU-style
perturbation is applied to the commanded target so recorded demonstrations
exhibit teleop-style overshoot-and-correct behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, cast

import rclpy
from geometry_msgs.msg import Transform
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .cheatcode_perturbation import OUPerturbation, PerturbationConfig
from .cheatcode_trajectory import (
    IntegratorState,
    calc_gripper_pose,
    wait_for_tf,
)
from .types import PoseTargetActionDict


@TeleoperatorConfig.register_subclass("cheatcode")
@dataclass(kw_only=True)
class CheatCodeTeleopConfig(TeleoperatorConfig):
    # Task identifiers — must be set by caller to match the spawned task.
    cable_name: str = "ethernet_cable"
    plug_name: str = "plug0"
    target_module_name: str = "task_board"
    port_name: str = "ethernet_port0"

    # Trajectory shape (mirrors CheatCode.insert_cable).
    approach_ticks: int = 100  # ~5 s at 20 Hz
    approach_z_offset_m: float = 0.2
    descend_step_m: float = 0.0005
    descend_final_z_offset_m: float = -0.015

    # Perturbation.
    approach_noise_xyz_m: float = 0.004
    descent_noise_xyz_m: float = 0.001
    approach_rot_noise_deg: float = 2.0
    ou_theta: float = 0.05
    noise_seed: int | None = None

    # Success predicate.
    success_xy_tol_m: float = 0.002
    success_z_tol_m: float = 0.001
    success_hold_ticks: int = 10

    # TF wait timeout on connect().
    tf_wait_timeout_s: float = 30.0

    # Controller integrator windup cap (matches CheatCode).
    integrator_max_windup: float = 0.05
    integrator_i_gain: float = 0.15

    # If True, build the descent target in the port's local frame so the
    # insertion axis follows the port's orientation rather than world z. More
    # robust to randomized port poses; flagged for A/B against the legacy
    # world-frame descent.
    descent_in_port_frame: bool = True

    # Per-tick slew limit on commanded pose (defense against episode-
    # boundary discontinuities or TF hiccups). Disable by setting <=0.
    max_step_xyz_m: float = 0.02
    max_step_rot_deg: float = 5.0


class CheatCodeTeleop(Teleoperator):
    def __init__(self, config: CheatCodeTeleopConfig):
        super().__init__(config)
        self.config = config

        self._is_connected = False
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._executor_thread: Thread | None = None
        self._tf_buffer: Buffer | None = None
        self._tf_listener: TransformListener | None = None

        self._port_transform: Transform | None = None
        self._integrator = IntegratorState(max_windup=config.integrator_max_windup)
        self._perturbation = OUPerturbation(
            PerturbationConfig(
                approach_noise_xyz_m=config.approach_noise_xyz_m,
                descent_noise_xyz_m=config.descent_noise_xyz_m,
                approach_rot_noise_deg=config.approach_rot_noise_deg,
                ou_theta=config.ou_theta,
                seed=config.noise_seed,
            )
        )

        self._tick: int = 0
        self._z_offset: float = config.approach_z_offset_m
        self._phase: str = "approach"  # approach | descend | hold
        self._success_streak: int = 0
        self._last_action: PoseTargetActionDict | None = None

    # --- Teleoperator API -------------------------------------------------

    @property
    def name(self) -> str:
        return "cheatcode"

    @property
    def action_features(self) -> dict:
        return PoseTargetActionDict.__annotations__

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("cheatcode_teleop_node")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        self._is_connected = True

        # Matches CheatCode.py: f"task_board/{target_module_name}/{port_name}_link"
        port_frame = (
            f"task_board/{self.config.target_module_name}/{self.config.port_name}_link"
        )
        plug_frame = f"{self.config.cable_name}/{self.config.plug_name}_link"
        for frame in [port_frame, plug_frame, "gripper/tcp"]:
            ok = wait_for_tf(
                self._tf_buffer,
                "base_link",
                frame,
                clock=self._node.get_clock(),
                timeout_sec=self.config.tf_wait_timeout_s,
                logger=self._node.get_logger(),
            )
            if not ok:
                raise RuntimeError(
                    f"CheatCodeTeleop.connect(): TF '{frame}' not available"
                )

        # Snapshot the port pose once per episode — ground-truth.
        port_tf = self._tf_buffer.lookup_transform("base_link", port_frame, Time())
        self._port_transform = port_tf.transform

        self._reset_episode_state()

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=1.0)
        self._is_connected = False
        self._node = None
        self._executor = None
        self._executor_thread = None
        self._tf_buffer = None
        self._tf_listener = None

    # --- Action generation ------------------------------------------------

    def _reset_episode_state(self) -> None:
        self._tick = 0
        self._z_offset = self.config.approach_z_offset_m
        self._phase = "approach"
        self._success_streak = 0
        self._integrator.reset()
        self._perturbation.reset()
        self._last_action = None

    def reset(self) -> None:
        """Called by the record wrapper between episodes to clear OU bias,
        phase/tick/integrator state, and the cached last action. Also
        re-snapshots the port TF so port-pose randomization (if any) is
        picked up fresh.
        """
        if not self._is_connected or self._tf_buffer is None:
            return
        try:
            port_frame = (
                f"task_board/{self.config.target_module_name}/{self.config.port_name}_link"
            )
            port_tf = self._tf_buffer.lookup_transform("base_link", port_frame, Time())
            self._port_transform = port_tf.transform
        except TransformException as ex:
            if self._node is not None:
                self._node.get_logger().warn(
                    f"CheatCodeTeleop.reset(): port TF refresh failed: {ex}"
                )
        self._reset_episode_state()

    def _apply_slew_limit(self, action: PoseTargetActionDict) -> PoseTargetActionDict:
        if self._last_action is None:
            return action
        max_xyz = self.config.max_step_xyz_m
        if max_xyz <= 0.0:
            return action
        dx = action["position.x"] - self._last_action["position.x"]
        dy = action["position.y"] - self._last_action["position.y"]
        dz = action["position.z"] - self._last_action["position.z"]
        step = math.sqrt(dx * dx + dy * dy + dz * dz)
        if step > max_xyz:
            k = max_xyz / step
            action = cast(
                PoseTargetActionDict,
                {
                    **action,
                    "position.x": self._last_action["position.x"] + dx * k,
                    "position.y": self._last_action["position.y"] + dy * k,
                    "position.z": self._last_action["position.z"] + dz * k,
                },
            )
            if self._node is not None:
                self._node.get_logger().warn(
                    f"CheatCodeTeleop: clamped commanded pose step {step * 100:.2f}cm → "
                    f"{max_xyz * 100:.2f}cm (tick={self._tick}, phase={self._phase})"
                )
        return action

    def _advance_schedule(self) -> tuple[float, float, float, bool]:
        """Return (slerp_fraction, position_fraction, z_offset, reset_integrator)
        for the current tick, and advance state.
        """
        if self._phase == "approach":
            f = min(1.0, (self._tick + 1) / max(1, self.config.approach_ticks))
            slerp_fraction = f
            position_fraction = f
            z_offset = self.config.approach_z_offset_m
            reset_integrator = True
            if self._tick + 1 >= self.config.approach_ticks:
                self._phase = "descend"
        elif self._phase == "descend":
            slerp_fraction = 1.0
            position_fraction = 1.0
            # Step z_offset down each tick; don't go below the final value.
            self._z_offset = max(
                self.config.descend_final_z_offset_m,
                self._z_offset - self.config.descend_step_m,
            )
            z_offset = self._z_offset
            reset_integrator = False
        else:  # hold
            slerp_fraction = 1.0
            position_fraction = 1.0
            z_offset = self._z_offset
            reset_integrator = False
        return slerp_fraction, position_fraction, z_offset, reset_integrator

    def _check_success(self, plug_xyz: tuple[float, float, float]) -> None:
        assert self._port_transform is not None
        dx = plug_xyz[0] - self._port_transform.translation.x
        dy = plug_xyz[1] - self._port_transform.translation.y
        dist_xy = math.hypot(dx, dy)
        dist_z = abs(plug_xyz[2] - self._port_transform.translation.z)
        if (
            dist_xy < self.config.success_xy_tol_m
            and dist_z < self.config.success_z_tol_m
        ):
            self._success_streak += 1
        else:
            self._success_streak = 0
        if (
            self._phase == "descend"
            and self._success_streak >= self.config.success_hold_ticks
        ):
            self._phase = "hold"
            if self._node is not None:
                self._node.get_logger().info(
                    "CheatCodeTeleop: insertion success, entering HOLD"
                )

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected or self._tf_buffer is None or self._port_transform is None:
            raise DeviceNotConnectedError()

        slerp_fraction, position_fraction, z_offset, reset_integrator = (
            self._advance_schedule()
        )

        try:
            pose, plug_xyz = calc_gripper_pose(
                tf_buffer=self._tf_buffer,
                port_transform=self._port_transform,
                cable_name=self.config.cable_name,
                plug_name=self.config.plug_name,
                integrator=self._integrator,
                slerp_fraction=slerp_fraction,
                position_fraction=position_fraction,
                z_offset=z_offset,
                reset_xy_integrator=reset_integrator,
                i_gain=self.config.integrator_i_gain,
                descent_in_port_frame=self.config.descent_in_port_frame,
            )
        except TransformException as ex:
            if self._node is not None:
                self._node.get_logger().warn(
                    f"CheatCodeTeleop: TF lookup failed on tick {self._tick}: {ex}"
                )
            if self._last_action is not None:
                return cast(dict, self._last_action)
            # No prior pose and TF missing — return a zero/identity action.
            return cast(
                dict,
                {
                    "position.x": 0.0,
                    "position.y": 0.0,
                    "position.z": 0.0,
                    "orientation.w": 1.0,
                    "orientation.x": 0.0,
                    "orientation.y": 0.0,
                    "orientation.z": 0.0,
                },
            )

        self._check_success(plug_xyz)

        # Apply perturbation to the commanded pose (does NOT affect the true
        # plug/port TFs used for PI correction — the next tick will pull us
        # back, producing overshoot-and-correct).
        phase_for_noise = "approach" if self._phase == "approach" else "descend"
        px, py, pz = self._perturbation.perturb_xyz(
            (pose.position.x, pose.position.y, pose.position.z), phase_for_noise
        )
        qw, qx, qy, qz = self._perturbation.perturb_orientation(
            (
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ),
            phase_for_noise,
        )

        action: PoseTargetActionDict = {
            "position.x": float(px),
            "position.y": float(py),
            "position.z": float(pz),
            "orientation.w": float(qw),
            "orientation.x": float(qx),
            "orientation.y": float(qy),
            "orientation.z": float(qz),
        }
        action = self._apply_slew_limit(action)
        self._last_action = action
        self._tick += 1
        return cast(dict, action)