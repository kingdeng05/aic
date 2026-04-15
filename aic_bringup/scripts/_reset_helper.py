#!/usr/bin/python3
"""Helper script that runs ros2 service calls in a clean process (no existing rclpy/Zenoh session)."""

import random
import subprocess
import sys
import time

# Nominal home TCP pose observed from recorded data (Cluster A centroid).
# Orientation = gripper pointing down (180° rotation about X axis).
NOMINAL_HOME_POS = (-0.3719, 0.1943, 0.3286)
NOMINAL_HOME_QUAT = (1.0, 0.0, 0.0, 0.0)  # x, y, z, w

# Max random offset per axis (meters) when using home_random.
DEFAULT_RANDOM_OFFSET_M = 0.06  # ±6 cm per axis


def run_service_call(service, srv_type, request, timeout=15):
    r = subprocess.run(
        ["ros2", "service", "call", service, srv_type, request],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        print(f"FAIL {service}: {r.stderr.strip()[-200:]}", file=sys.stderr)
        return False
    print(f"OK {service}")
    return True


def home():
    joint_names = "['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']"
    joint_pos = "[-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]"

    print("Deactivating aic_controller...")
    if not run_service_call(
        "/controller_manager/switch_controller",
        "controller_manager_msgs/srv/SwitchController",
        "{deactivate_controllers: ['aic_controller'], strictness: 1}",
    ):
        return False

    print("Resetting joints to home...")
    if not run_service_call(
        "/scoring/reset_joints",
        "aic_engine_interfaces/srv/ResetJoints",
        f"{{joint_names: {joint_names}, initial_positions: {joint_pos}}}",
    ):
        return False

    print("Waiting for physics to settle...")
    time.sleep(3.0)

    print("Reactivating aic_controller...")
    if not run_service_call(
        "/controller_manager/switch_controller",
        "controller_manager_msgs/srv/SwitchController",
        "{activate_controllers: ['aic_controller'], strictness: 1}",
    ):
        return False

    # Switch to joint mode and publish home joint positions so the controller
    # tracks the correct target instead of its stale pre-reset Cartesian target
    print("Switching to joint mode and sending home position...")
    run_service_call(
        "/aic_controller/change_target_mode",
        "aic_control_interfaces/srv/ChangeTargetMode",
        "{target_mode: {mode: 2}}",  # MODE_JOINT = 2
    )
    time.sleep(0.3)

    # Publish home joint positions via ros2 topic pub (one-shot)
    joint_cmd = (
        '{target_state: {positions: [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110], '
        'velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, '
        'target_stiffness: [85.0, 85.0, 85.0, 85.0, 85.0, 85.0], '
        'target_damping: [75.0, 75.0, 75.0, 75.0, 75.0, 75.0], '
        'trajectory_generation_mode: {mode: 2}}'  # MODE_POSITION = 2
    )
    # Publish repeatedly for 5 seconds so the controller continuously tracks home
    print("Publishing home joint commands for 5 seconds...")
    pub_proc = subprocess.Popen(
        ["ros2", "topic", "pub", "--rate", "50",
         "/aic_controller/joint_commands",
         "aic_control_interfaces/msg/JointMotionUpdate",
         joint_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(5.0)
    pub_proc.terminate()
    pub_proc.wait()
    print("Done publishing home commands")

    # Switch back to cartesian mode for teleop
    print("Switching back to cartesian mode...")
    run_service_call(
        "/aic_controller/change_target_mode",
        "aic_control_interfaces/srv/ChangeTargetMode",
        "{target_mode: {mode: 1}}",  # MODE_CARTESIAN = 1
    )

    print("Taring FT sensor...")
    run_service_call(
        "/aic_controller/tare_force_torque_sensor",
        "std_srvs/srv/Trigger",
        "{}",
    )

    print("Home complete")
    return True


def apply_random_cartesian_offset(max_offset_m: float = DEFAULT_RANDOM_OFFSET_M) -> bool:
    """After homing, nudge the TCP in Cartesian space by a small random xyz offset.

    Orientation is kept at the nominal home orientation. Only translation varies.
    """
    dx = random.uniform(-max_offset_m, max_offset_m)
    dy = random.uniform(-max_offset_m, max_offset_m)
    dz = random.uniform(-max_offset_m, max_offset_m)

    tx = NOMINAL_HOME_POS[0] + dx
    ty = NOMINAL_HOME_POS[1] + dy
    tz = NOMINAL_HOME_POS[2] + dz
    qx, qy, qz, qw = NOMINAL_HOME_QUAT

    print(f"Applying random Cartesian offset: dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}")

    # Already in cartesian mode after home() finishes its mode switch.

    # Build 36-element row-major 6x6 diagonal stiffness/damping matrices.
    def diag36(v: float) -> str:
        mat = [0.0] * 36
        for i in range(6):
            mat[i * 6 + i] = v
        return "[" + ",".join(f"{x}" for x in mat) + "]"

    pose_cmd = (
        '{header: {frame_id: "base_link"}, '
        f'pose: {{position: {{x: {tx}, y: {ty}, z: {tz}}}, '
        f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}}}, '
        f'target_stiffness: {diag36(85.0)}, '
        f'target_damping: {diag36(75.0)}, '
        'feedforward_wrench_at_tip: {force: {x: 0.0, y: 0.0, z: 0.0}, torque: {x: 0.0, y: 0.0, z: 0.0}}, '
        'wrench_feedback_gains_at_tip: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], '
        'trajectory_generation_mode: {mode: 2}}'  # MODE_POSITION = 2
    )

    print("Publishing Cartesian pose target for 8 seconds to let robot settle...")
    pub_proc = subprocess.Popen(
        ["ros2", "topic", "pub", "--rate", "50",
         "/aic_controller/pose_commands",
         "aic_control_interfaces/msg/MotionUpdate",
         pose_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(8.0)
    pub_proc.terminate()
    pub_proc.wait()

    print("Taring FT sensor (again, at randomized pose)...")
    run_service_call(
        "/aic_controller/tare_force_torque_sensor",
        "std_srvs/srv/Trigger",
        "{}",
    )

    print(f"home_random complete at offset ({dx:+.3f}, {dy:+.3f}, {dz:+.3f})")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "home"
    if cmd == "home":
        ok = home()
        sys.exit(0 if ok else 1)
    elif cmd == "home_random":
        max_off = DEFAULT_RANDOM_OFFSET_M
        if len(sys.argv) > 2:
            max_off = float(sys.argv[2])
        ok = home() and apply_random_cartesian_offset(max_off)
        sys.exit(0 if ok else 1)
    elif cmd == "random_offset":
        max_off = DEFAULT_RANDOM_OFFSET_M
        if len(sys.argv) > 2:
            max_off = float(sys.argv[2])
        ok = apply_random_cartesian_offset(max_off)
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
