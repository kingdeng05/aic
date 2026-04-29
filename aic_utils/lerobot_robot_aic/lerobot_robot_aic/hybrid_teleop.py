#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""Hybrid CheatCode -> Keyboard EE teleop.

CheatCodeTeleop drives the approach autonomously; on a takeover key the
human takes over via AICKeyboardEETeleop and finishes the insertion.

The robot must be configured with ``teleop_target_mode="pose"`` because
this teleop emits PoseTargetActionDict every tick (manual twist deltas
are integrated onto the last commanded pose so the action stream is
uniform across the handoff).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

from .aic_teleop import AICKeyboardEETeleop, AICKeyboardEETeleopConfig
from .cheatcode_teleop import CheatCodeTeleop, CheatCodeTeleopConfig
from .types import MotionUpdateActionDict, PoseTargetActionDict

@TeleoperatorConfig.register_subclass("hybrid_cheatcode_keyboard")
@dataclass(kw_only=True)
class HybridCheatCodeKeyboardTeleopConfig(TeleoperatorConfig):
    # Task identifiers — must be set by caller to match the spawned task.
    cable_name: str = "ethernet_cable"
    plug_name: str = "plug0"
    target_module_name: str = "task_board"
    port_name: str = "ethernet_port0"

    # Trajectory shape (mirrors CheatCode.insert_cable).
    approach_ticks: int = 100
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
    descent_in_port_frame: bool = False

    # Per-tick slew limit on commanded pose (defense against episode-
    # boundary discontinuities or TF hiccups). Disable by setting <=0.
    max_step_xyz_m: float = 0.02
    max_step_rot_deg: float = 5.0

    # --- Keyboard fields (mirrors AICKeyboardEETeleopConfig) ---
    # NOTE: in the hybrid these are per-tick *position/orientation deltas*
    # (m and rad) added to the last commanded pose, not velocity targets
    # like in the standalone keyboard teleop. At 30 Hz, 0.002 m/tick = 6 cm/s.
    use_gripper: bool = True
    high_command_scaling: float = 5e-4
    low_command_scaling: float = 2.5e-4

    # --- Hybrid-only ---
    takeover_key: str = "x"
    release_key: str | None = None  # if set, hand control back to cheatcode

    def _build_cheatcode_config(self) -> CheatCodeTeleopConfig:
        return CheatCodeTeleopConfig(
            id=self.id,
            cable_name=self.cable_name,
            plug_name=self.plug_name,
            target_module_name=self.target_module_name,
            port_name=self.port_name,
            approach_ticks=self.approach_ticks,
            approach_z_offset_m=self.approach_z_offset_m,
            descend_step_m=self.descend_step_m,
            descend_final_z_offset_m=self.descend_final_z_offset_m,
            approach_noise_xyz_m=self.approach_noise_xyz_m,
            descent_noise_xyz_m=self.descent_noise_xyz_m,
            approach_rot_noise_deg=self.approach_rot_noise_deg,
            ou_theta=self.ou_theta,
            noise_seed=self.noise_seed,
            success_xy_tol_m=self.success_xy_tol_m,
            success_z_tol_m=self.success_z_tol_m,
            success_hold_ticks=self.success_hold_ticks,
            tf_wait_timeout_s=self.tf_wait_timeout_s,
            integrator_max_windup=self.integrator_max_windup,
            integrator_i_gain=self.integrator_i_gain,
            descent_in_port_frame=self.descent_in_port_frame,
            max_step_xyz_m=self.max_step_xyz_m,
            max_step_rot_deg=self.max_step_rot_deg,
        )

    def _build_keyboard_config(self) -> AICKeyboardEETeleopConfig:
        return AICKeyboardEETeleopConfig(
            id=self.id,
            use_gripper=self.use_gripper,
            high_command_scaling=self.high_command_scaling,
            low_command_scaling=self.low_command_scaling,
        )

def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )

def _quat_normalize(
    q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)

def _apply_angular_delta(
    q: tuple[float, float, float, float], wx: float, wy: float, wz: float
) -> tuple[float, float, float, float]:
    """Compose a small-angle rotation (wx, wy, wz) onto quaternion q."""
    if wx == 0.0 and wy == 0.0 and wz == 0.0:
        return q
    dq = _quat_normalize((1.0, wx * 0.5, wy * 0.5, wz * 0.5))
    return _quat_normalize(_quat_mul(dq, q))

class HybridCheatCodeKeyboardTeleop(Teleoperator):
    config_class = HybridCheatCodeKeyboardTeleopConfig

    def __init__(self, config: HybridCheatCodeKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self._cheatcode = CheatCodeTeleop(config._build_cheatcode_config())
        self._keyboard = AICKeyboardEETeleop(config._build_keyboard_config())
        self._mode: str = "cheatcode"
        self._last_pose: PoseTargetActionDict | None = None

    @property
    def name(self) -> str:
        return "hybrid_cheatcode_keyboard"

    @property
    def action_features(self) -> dict:
        return PoseTargetActionDict.__annotations__

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._cheatcode.is_connected and self._keyboard.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self._cheatcode.configure()
        self._keyboard.configure()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self._cheatcode.connect(calibrate=calibrate)
        self._keyboard.connect()

    def disconnect(self) -> None:
        try:
            self._keyboard.disconnect()
        finally:
            self._cheatcode.disconnect()

    def reset(self) -> None:
        self._cheatcode.reset()
        kb_reset = getattr(self._keyboard, "reset", None)
        if callable(kb_reset):
            kb_reset()
        self._mode = "cheatcode"
        self._last_pose = None

    def _check_mode_switch(self) -> None:
        q = self._keyboard.misc_keys_queue
        drained: list[str] = []
        while not q.empty():
            try:
                drained.append(q.get_nowait())
            except Exception:
                break
        for key in drained:
            if key == self.config.takeover_key and self._mode == "cheatcode":
                self._mode = "manual"
                print(
                    f"HybridTeleop: takeover key '{key}' pressed -> MANUAL mode"
                )
            elif (
                self.config.release_key is not None
                and key == self.config.release_key
                and self._mode == "manual"
            ):
                self._mode = "cheatcode"
                print(
                    f"HybridTeleop: release key '{key}' pressed -> CHEATCODE mode"
                )

    def _integrate_manual(
        self, kb_action: MotionUpdateActionDict
    ) -> PoseTargetActionDict:
        assert self._last_pose is not None
        last = self._last_pose
        new_pos = (
            last["position.x"] + float(kb_action["linear.x"]),
            last["position.y"] + float(kb_action["linear.y"]),
            last["position.z"] + float(kb_action["linear.z"]),
        )
        last_q = (
            last["orientation.w"],
            last["orientation.x"],
            last["orientation.y"],
            last["orientation.z"],
        )
        qw, qx, qy, qz = _apply_angular_delta(
            last_q,
            float(kb_action["angular.x"]),
            float(kb_action["angular.y"]),
            float(kb_action["angular.z"]),
        )
        return {
            "position.x": new_pos[0],
            "position.y": new_pos[1],
            "position.z": new_pos[2],
            "orientation.w": qw,
            "orientation.x": qx,
            "orientation.y": qy,
            "orientation.z": qz,
        }

    def get_action(self) -> dict[str, Any]:
        # Always pump the keyboard so its misc_keys_queue captures the
        # takeover key even while cheatcode is driving.
        kb_action = cast(
            MotionUpdateActionDict, self._keyboard.get_action()
        )
        self._check_mode_switch()

        if self._mode == "cheatcode" or self._last_pose is None:
            pose = cast(PoseTargetActionDict, self._cheatcode.get_action())
            self._last_pose = pose
            return cast(dict, pose)

        pose = self._integrate_manual(kb_action)
        self._last_pose = pose
        return cast(dict, pose)