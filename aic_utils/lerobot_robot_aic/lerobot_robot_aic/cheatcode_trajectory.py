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
    port_xy = (port_transform.translation.x, port_transform.translation.y)
    plug_xyz = (
        plug_tf.transform.translation.x,
        plug_tf.transform.translation.y,
        plug_tf.transform.translation.z,
    )
    plug_tip_gripper_offset_z = gripper_xyz[2] - plug_xyz[2]

    tip_x_error = port_xy[0] - plug_xyz[0]
    tip_y_error = port_xy[1] - plug_xyz[1]

    if reset_xy_integrator:
        integrator.reset()
    else:
        integrator.step(tip_x_error, tip_y_error)

    target_x = port_xy[0] + i_gain * integrator.x
    target_y = port_xy[1] + i_gain * integrator.y
    target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset_z

    blend_xyz = (
        position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
        position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
        position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
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
