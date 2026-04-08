#!/usr/bin/python3
"""Helper script that runs ros2 service calls in a clean process (no existing rclpy/Zenoh session)."""

import subprocess
import sys
import time


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
        "{target_mode: {mode: 1}}",  # MODE_JOINT = 1
    )
    time.sleep(0.3)

    # Publish home joint positions via ros2 topic pub (one-shot)
    joint_cmd = (
        '{target_state: {positions: [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110], '
        'velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, '
        'target_stiffness: [85.0, 85.0, 85.0, 85.0, 85.0, 85.0], '
        'target_damping: [75.0, 75.0, 75.0, 75.0, 75.0, 75.0], '
        'trajectory_generation_mode: {mode: 1}}'  # MODE_POSITION = 1
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
        "{target_mode: {mode: 0}}",  # MODE_CARTESIAN = 0
    )

    print("Taring FT sensor...")
    run_service_call(
        "/aic_controller/tare_force_torque_sensor",
        "std_srvs/srv/Trigger",
        "{}",
    )

    print("Home complete")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "home"
    if cmd == "home":
        ok = home()
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
