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
import random
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, cast

import numpy as np
import rclpy
from geometry_msgs.msg import Transform
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d.quaternions import quat2mat

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

    # Trajectory shape.
    #
    # Approach is split into two sub-phases so the gripper translates above
    # the port FIRST (keeping the existing orientation, so wrist cameras can
    # see the port come into frame) and then rotates to port-aligned without
    # changing position. approach_translate_ticks + approach_rotate_ticks
    # replaces the legacy single "approach_ticks" parameter; setting either
    # to 0 disables that sub-phase.
    approach_translate_ticks: int = 60
    approach_rotate_ticks: int = 40
    approach_z_offset_m: float = 0.2
    descend_step_m: float = 0.00025  # slower than legacy 0.0005 for SC port
    # Floor is intentionally very deep: the open-loop tick ramp will keep
    # commanding deeper than physically reachable until the success predicate
    # (port-local-z) fires or the episode times out. The ramp itself is not
    # the terminal signal anymore — _check_success is.
    descend_final_z_offset_m: float = -1.0

    # Lift-and-adjust: if the plug stops moving in z while we keep commanding
    # descent, we presume contact and the plug is hung up on the port lip.
    # Lift the gripper back up by `lift_amount_m`, reset the XY integrator,
    # and descend again. Capped at `max_lift_retries`.
    stuck_threshold_ticks: int = 30   # ~1 s at 30 Hz
    plug_stuck_eps_m: float = 1e-4    # <0.1 mm/tick of plug.z motion
    lift_amount_m: float = 0.025      # 2.5 cm — visible and large enough to clear lip
    lift_step_m: float = 0.001        # 4x descend_step_m so lift completes in ~250 ms
    max_lift_retries: int = 10
    # Pre-bias the XY integrator with a small random offset on each lift
    # completion. Without this, the integrator resets to (0,0) and re-converges
    # to the *same* wedge angle each retry, leading to a "lift -> same wedge"
    # cycle. With it, each retry approaches the port from a slightly different
    # azimuth, breaking out of repeated wedges. Set 0 to disable.
    lift_xy_jitter_m: float = 0.004

    # Perturbation.
    approach_noise_xyz_m: float = 0.004
    descent_noise_xyz_m: float = 0.001
    approach_rot_noise_deg: float = 2.0
    ou_theta: float = 0.05
    noise_seed: int | None = None

    # Success predicate. Evaluated in the port's *local* frame so it's well
    # defined for ports whose insertion axis isn't aligned with world axes.
    # SDF convention: port +Z is the insertion direction, entrance is at
    # -entrance_depth, port frame origin is inside the receptacle. We declare
    # success when the plug has reached port-local Z = 0 (within
    # success_z_tol_m on the entrance side) AND off-axis drift is within
    # success_xy_tol_m. With this predicate, the open-loop descent floor at
    # descend_final_z_offset_m no longer acts as a terminal signal — only
    # actual plug geometry does.
    success_xy_tol_m: float = 0.002
    success_z_tol_m: float = 0.005
    success_hold_ticks: int = 10

    # TF wait timeout on connect().
    tf_wait_timeout_s: float = 30.0

    # Controller integrator windup cap (matches CheatCode).
    integrator_max_windup: float = 0.05
    integrator_i_gain: float = 0.15

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
        # Phases: translate (move above port keeping orientation),
        # rotate (slerp orientation to port-aligned in place),
        # descend (step Z down to descend_final_z_offset_m),
        # lifting (back off Z after stuck-on-port detected),
        # hold (terminal — geometric success predicate satisfied).
        self._phase: str = "translate"
        self._success_streak: int = 0
        self._last_action: PoseTargetActionDict | None = None
        # Lift-and-adjust bookkeeping.
        self._stuck_ticks: int = 0
        self._last_plug_z: float | None = None
        self._lift_target_z_offset: float = config.approach_z_offset_m
        self._lift_retries: int = 0
        # Separate RNG for lift-retry jitter so it isn't entangled with the
        # OU perturbation seed semantics.
        self._lift_rng = random.Random(config.noise_seed)

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
        self._phase = "translate"
        self._success_streak = 0
        self._integrator.reset()
        self._perturbation.reset()
        self._last_action = None
        self._stuck_ticks = 0
        self._last_plug_z = None
        self._lift_target_z_offset = self.config.approach_z_offset_m
        self._lift_retries = 0

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

        Phase machine:
          translate -> rotate -> descend -> [lifting -> descend ...] -> hold

        translate: position_fraction ramps 0->1 while slerp_fraction stays 0
                   so the gripper moves to "above port" with its current
                   orientation (wrist cameras see the port appear in view).
        rotate:    position_fraction held at 1, slerp_fraction ramps 0->1
                   so the gripper rotates to port-aligned in place.
        descend:   step z_offset down by descend_step_m each tick, with the
                   integrator active so XY drifts toward the port.
        lifting:   step z_offset UP by descend_step_m each tick until
                   _lift_target_z_offset is reached, then return to descend
                   with a freshly-reset integrator. Triggered by
                   _maybe_trigger_lift() when plug.z stops moving but we are
                   still commanding descent.
        hold:      terminal — descend converged within geometric tolerance.
        """
        translate_n = max(0, self.config.approach_translate_ticks)
        rotate_n = max(0, self.config.approach_rotate_ticks)

        if self._phase == "translate":
            denom = max(1, translate_n)
            position_fraction = min(1.0, (self._tick + 1) / denom)
            slerp_fraction = 0.0
            z_offset = self.config.approach_z_offset_m
            reset_integrator = True
            if self._tick + 1 >= translate_n:
                self._phase = "rotate" if rotate_n > 0 else "descend"
                # Restart sub-phase counter for the rotate phase.
                self._tick = -1  # will be incremented at end of get_action
        elif self._phase == "rotate":
            denom = max(1, rotate_n)
            slerp_fraction = min(1.0, (self._tick + 1) / denom)
            position_fraction = 1.0
            z_offset = self.config.approach_z_offset_m
            reset_integrator = True
            if self._tick + 1 >= rotate_n:
                self._phase = "descend"
                self._tick = -1
        elif self._phase == "descend":
            slerp_fraction = 1.0
            position_fraction = 1.0
            self._z_offset = max(
                self.config.descend_final_z_offset_m,
                self._z_offset - self.config.descend_step_m,
            )
            z_offset = self._z_offset
            reset_integrator = False
        elif self._phase == "lifting":
            slerp_fraction = 1.0
            position_fraction = 1.0
            # Step z_offset back UP to the lift target at lift_step_m/tick.
            self._z_offset = min(
                self._lift_target_z_offset,
                self._z_offset + self.config.lift_step_m,
            )
            z_offset = self._z_offset
            # Hold integrator fixed while lifting; XY isn't trying to track.
            reset_integrator = False
            if self._z_offset >= self._lift_target_z_offset - 1e-6:
                # Lifted enough; reset XY integrator (optionally with a small
                # random pre-bias so the next descent doesn't drive the plug
                # to the same wedge point), then try descending again.
                jitter = self.config.lift_xy_jitter_m
                if jitter > 0.0:
                    i_gain = max(self.config.integrator_i_gain, 1e-6)
                    # Map desired metric XY bias back to integrator state.
                    jx_m = (self._lift_rng.random() * 2.0 - 1.0) * jitter
                    jy_m = (self._lift_rng.random() * 2.0 - 1.0) * jitter
                    self._integrator.x = jx_m / i_gain
                    self._integrator.y = jy_m / i_gain
                else:
                    jx_m = 0.0
                    jy_m = 0.0
                    self._integrator.reset()
                self._stuck_ticks = 0
                self._last_plug_z = None
                self._phase = "descend"
                if self._node is not None:
                    self._node.get_logger().info(
                        f"CheatCodeTeleop: lift complete at z_offset="
                        f"{self._z_offset:+.4f}, resuming descend "
                        f"(integrator pre-bias: x={jx_m * 1000:+.2f}mm, "
                        f"y={jy_m * 1000:+.2f}mm)"
                    )
        else:  # hold
            slerp_fraction = 1.0
            position_fraction = 1.0
            z_offset = self._z_offset
            reset_integrator = False
        return slerp_fraction, position_fraction, z_offset, reset_integrator

    def _maybe_trigger_lift(self, plug_z: float) -> None:
        """If we're descending and the plug has stopped moving in z while we
        keep commanding it down, presume the plug hit the port lip and lift
        back up to retry. No-op while not in 'descend' phase. After
        max_lift_retries, just give up and let descend run to its terminal.
        """
        if self._phase != "descend":
            return
        if self._lift_retries >= self.config.max_lift_retries:
            return
        # Need at least one previous sample to detect "no motion".
        if self._last_plug_z is None:
            self._last_plug_z = plug_z
            return
        if abs(plug_z - self._last_plug_z) < self.config.plug_stuck_eps_m:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        self._last_plug_z = plug_z
        if self._stuck_ticks >= self.config.stuck_threshold_ticks:
            self._lift_target_z_offset = self._z_offset + self.config.lift_amount_m
            self._lift_retries += 1
            self._phase = "lifting"
            if self._node is not None:
                # Look up plug port-local z + gripper TCP for diagnostics so
                # we can tell from the log alone whether the gripper actually
                # moves when lift fires (GUI delta of 8mm was invisible).
                extras = ""
                try:
                    assert self._port_transform is not None
                    assert self._tf_buffer is not None
                    q_port = (
                        self._port_transform.rotation.w,
                        self._port_transform.rotation.x,
                        self._port_transform.rotation.y,
                        self._port_transform.rotation.z,
                    )
                    R_port = quat2mat(np.array(q_port))
                    port_pos = np.array(
                        [
                            self._port_transform.translation.x,
                            self._port_transform.translation.y,
                            self._port_transform.translation.z,
                        ]
                    )
                    plug_tf = self._tf_buffer.lookup_transform(
                        "base_link",
                        f"{self.config.cable_name}/{self.config.plug_name}_link",
                        Time(),
                    )
                    plug_pos = np.array(
                        [
                            plug_tf.transform.translation.x,
                            plug_tf.transform.translation.y,
                            plug_tf.transform.translation.z,
                        ]
                    )
                    plug_pl = R_port.T @ (plug_pos - port_pos)
                    tcp_tf = self._tf_buffer.lookup_transform(
                        "base_link", "gripper/tcp", Time()
                    )
                    tcp = (
                        tcp_tf.transform.translation.x,
                        tcp_tf.transform.translation.y,
                        tcp_tf.transform.translation.z,
                    )
                    extras = (
                        f" | plug_port_local=({plug_pl[0]:+.4f},"
                        f"{plug_pl[1]:+.4f},{plug_pl[2]:+.4f}) "
                        f"tcp_world=({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f})"
                    )
                except (TransformException, AssertionError):
                    pass
                self._node.get_logger().info(
                    f"CheatCodeTeleop: plug stuck (z={plug_z:+.4f}), "
                    f"lifting to z_offset={self._lift_target_z_offset:+.4f} "
                    f"(retry {self._lift_retries}/{self.config.max_lift_retries})"
                    f"{extras}"
                )

    def _check_success(self, plug_xyz: tuple[float, float, float]) -> None:
        assert self._port_transform is not None
        # Express plug in port-local frame. SDF convention: port +Z is the
        # insertion direction, entrance is at port_local_z = -entrance_depth.
        # The plug is "inserted" when its tip reaches the port frame origin
        # (within success_z_tol_m on the entrance side) AND off-axis drift is
        # within success_xy_tol_m. World-frame XYZ tolerance (the old
        # predicate) couldn't fire for ports whose +Z isn't world +Z, because
        # the gripper-vs-plug rigid offset moves the plug below the port-z in
        # world frame even at full insertion.
        q_port = (
            self._port_transform.rotation.w,
            self._port_transform.rotation.x,
            self._port_transform.rotation.y,
            self._port_transform.rotation.z,
        )
        R_port = quat2mat(np.array(q_port))
        port_pos = np.array(
            [
                self._port_transform.translation.x,
                self._port_transform.translation.y,
                self._port_transform.translation.z,
            ]
        )
        plug_in_port_local = R_port.T @ (np.array(plug_xyz) - port_pos)
        dist_xy = math.hypot(
            float(plug_in_port_local[0]), float(plug_in_port_local[1])
        )
        plug_port_local_z = float(plug_in_port_local[2])
        if (
            dist_xy < self.config.success_xy_tol_m
            and plug_port_local_z >= -self.config.success_z_tol_m
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
                    "CheatCodeTeleop: insertion success "
                    f"(port_local_z={plug_port_local_z:+.4f}m, "
                    f"xy={dist_xy * 1000:.2f}mm), entering HOLD"
                )

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected or self._tf_buffer is None or self._port_transform is None:
            raise DeviceNotConnectedError()

        # Once HOLD has fired (insertion success), park the action. Otherwise
        # the integrator and OU perturbation keep accumulating during the
        # post-success tail and the recorded xy drifts ~1cm over the episode
        # remainder. Returning the last action keeps the plateau clean so the
        # downstream trim script chops on a true plateau.
        if self._phase == "hold" and self._last_action is not None:
            self._tick += 1
            return cast(dict, self._last_action)

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
        self._maybe_trigger_lift(plug_xyz[2])

        # Apply perturbation to the commanded pose (does NOT affect the true
        # plug/port TFs used for PI correction — the next tick will pull us
        # back, producing overshoot-and-correct).
        phase_for_noise = (
            "approach" if self._phase in ("translate", "rotate") else "descend"
        )
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
