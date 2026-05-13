#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""Shared scripted-trajectory helpers lifted from CheatCode.

Kept TF-buffer-injected so both the original Policy-style runner and the
lerobot CheatCodeTeleop can use the same geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.time import Time
from tf2_ros import Buffer, TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import quat2mat


@dataclass
class IntegratorState:
    x: float = 0.0
    y: float = 0.0
    max_windup: float = 0.05

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0

    def step(self, ex: float, ey: float) -> None:
        self.x = float(np.clip(self.x + ex, -self.max_windup, self.max_windup))
        self.y = float(np.clip(self.y + ey, -self.max_windup, self.max_windup))


def wait_for_tf(
    tf_buffer: Buffer,
    target_frame: str,
    source_frame: str,
    clock,
    timeout_sec: float = 10.0,
    logger=None,
) -> bool:
    """Poll TF buffer until `source_frame -> target_frame` is available."""
    from rclpy.duration import Duration

    start = clock.now()
    timeout = Duration(seconds=timeout_sec)
    attempt = 0
    while (clock.now() - start) < timeout:
        try:
            tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return True
        except TransformException:
            if logger is not None and attempt % 20 == 0:
                logger.info(
                    f"Waiting for transform '{source_frame}' -> '{target_frame}'..."
                    " -- are you running with `ground_truth:=true`?"
                )
            attempt += 1
            clock.sleep_for(Duration(seconds=0.1))
    if logger is not None:
        logger.error(
            f"Transform '{source_frame}' -> '{target_frame}' not available"
            f" after {timeout_sec}s"
        )
    return False


def calc_gripper_pose(
    tf_buffer: Buffer,
    port_transform: Transform,
    cable_name: str,
    plug_name: str,
    integrator: IntegratorState,
    slerp_fraction: float = 1.0,
    position_fraction: float = 1.0,
    z_offset: float = 0.1,
    reset_xy_integrator: bool = False,
    i_gain: float = 0.15,
) -> tuple[Pose, tuple[float, float, float]]:
    """Port of CheatCode.calc_gripper_pose.

    Returns the target gripper Pose AND the current (plug_xyz) so callers
    can compute a success predicate without re-looking up the TF.
    """
    q_port = (
        port_transform.rotation.w,
        port_transform.rotation.x,
        port_transform.rotation.y,
        port_transform.rotation.z,
    )
    plug_tf = tf_buffer.lookup_transform(
        "base_link", f"{cable_name}/{plug_name}_link", Time()
    )
    q_plug = (
        plug_tf.transform.rotation.w,
        plug_tf.transform.rotation.x,
        plug_tf.transform.rotation.y,
        plug_tf.transform.rotation.z,
    )
    q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
    q_diff = quaternion_multiply(q_port, q_plug_inv)
    gripper_tf = tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
    q_gripper = (
        gripper_tf.transform.rotation.w,
        gripper_tf.transform.rotation.x,
        gripper_tf.transform.rotation.y,
        gripper_tf.transform.rotation.z,
    )
    q_gripper_target = quaternion_multiply(q_diff, q_gripper)
    q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

    gripper_xyz = (
        gripper_tf.transform.translation.x,
        gripper_tf.transform.translation.y,
        gripper_tf.transform.translation.z,
    )
    plug_xyz = (
        plug_tf.transform.translation.x,
        plug_tf.transform.translation.y,
        plug_tf.transform.translation.z,
    )

    # Position control: descend along the port's *local* Z (the insertion
    # axis), not base_link Z. Otherwise ports whose insertion axis is not
    # aligned with world -Z (e.g. the SC port, whose Z points along world +X
    # given task_board_yaw=π + per-port RPY of (π/2,0,π/2)) get pushed
    # perpendicular to their entrance and the plug never enters.
    R_port = quat2mat(np.array(q_port))   # 3x3, columns = port axes in world
    R_plug = quat2mat(np.array(q_plug))

    port_pos = np.array(
        [
            port_transform.translation.x,
            port_transform.translation.y,
            port_transform.translation.z,
        ]
    )
    plug_pos = np.array(plug_xyz)
    gripper_pos = np.array(gripper_xyz)

    # Plug position expressed in port-local frame; off-axis drift is the
    # plug's X/Y in this frame. Integrator accumulates that off-axis error
    # so we apply correction perpendicular to the insertion direction.
    plug_in_port_local = R_port.T @ (plug_pos - port_pos)
    err_x = -float(plug_in_port_local[0])
    err_y = -float(plug_in_port_local[1])
    if reset_xy_integrator:
        integrator.reset()
    else:
        integrator.step(err_x, err_y)

    # Desired plug-tip position in world: z_offset along port-local *-Z*,
    # plus the integrator's correction in port-local X/Y. The minus sign
    # reflects the SDF convention used by every port in this repo: the port
    # frame's +Z is the insertion direction (entrance link is at
    # port_local_z = -entrance_depth), so the *approach* pose is on the
    # negative-Z side of the port frame and "deeper into the port" means
    # negative z_offset. With this convention:
    #   approach_z_offset_m = +0.2   -> 20 cm "above" the port (approach)
    #   descend_final_z_offset_m = -0.015 -> 1.5 cm past the port frame.
    plug_target_local = np.array(
        [i_gain * integrator.x, i_gain * integrator.y, -z_offset]
    )
    plug_target_world = port_pos + R_port @ plug_target_local

    # Gripper-to-plug offset is rigid in the plug's frame (cable is welded
    # to the gripper). As the gripper slerps toward port-aligned, the plug
    # follows, so the world-frame offset rotates accordingly.
    plug_to_gripper_local = R_plug.T @ (gripper_pos - plug_pos)
    q_plug_slerp = np.array(quaternion_slerp(q_plug, q_port, slerp_fraction))
    R_plug_slerp = quat2mat(q_plug_slerp)
    gripper_target_world = plug_target_world + R_plug_slerp @ plug_to_gripper_local

    blend_xyz = (
        position_fraction * float(gripper_target_world[0])
        + (1.0 - position_fraction) * gripper_xyz[0],
        position_fraction * float(gripper_target_world[1])
        + (1.0 - position_fraction) * gripper_xyz[1],
        position_fraction * float(gripper_target_world[2])
        + (1.0 - position_fraction) * gripper_xyz[2],
    )

    pose = Pose(
        position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
        orientation=Quaternion(
            w=q_gripper_slerp[0],
            x=q_gripper_slerp[1],
            y=q_gripper_slerp[2],
            z=q_gripper_slerp[3],
        ),
    )
    return pose, plug_xyz
