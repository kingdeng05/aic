# lerobot_robot_aic

This package contains a [LeRobot](https://huggingface.co/lerobot) interface for the AIC robot.

## Usage

This describe some of the things you can do with LeRobot, for more information, see the official [LeRobot docs](https://huggingface.co/docs/lerobot/en/index).

The LeRobot driver is installed in a [pixi](https://prefix.dev/tools/pixi) workspace. In general, you can prefix a command with `pixi run` or enter the environment with `pixi shell`.

### Teleoperating with LeRobot

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --display_data=true
```

Options for `--teleop.type` (and setting `--robot.teleop_target_mode` accordingly):

- `aic_keyboard_ee` for cartesian-space keyboard control (and set `--robot.teleop_target_mode=cartesian`)
- `aic_spacemouse` for cartesian-space SpaceMouse control (and set `--robot.teleop_target_mode=cartesian`)
- `aic_keyboard_joint` for joint-space control (and set `--robot.teleop_target_mode=joint`)
- `cheatcode` for scripted cable-insertion trajectories with OU perturbations (and set `--robot.teleop_target_mode=pose`) — see [Automated Data Collection via CheatCode](#automated-data-collection-via-cheatcode) below.

Options for `--robot.teleop_frame_id` when `--robot.teleop_target_mode` is `cartesian`:
- `base_link` to send cartesian targets with respect to the robot's base link.
- `gripper/tcp` to send cartesian targets with respect to the `tcp` frame attached to the robot's gripper.

As an example,
```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true
```


:warning: Note: In addition to setting `--teleop.type` you must set `--robot.teleop_target_mode` because the `AICRobotAICController` class needs to know which type of actions to send to the controller and it doesn't have access to `--teleop.type`.

#### Cartesian space control

For cartesian control, in addition to setting `--teleop.type` and `--robot.teleop_target_mode`, you can also set `teleop_frame_id` (the reference frame used for cartesian control) which sets the reference frame. Set this to either the gripper TCP (`"gripper/tcp"`, the default) or the robot base link (`"base_link"`).

##### Keyboard

> Note on using the Shift+&lt;key&gt; commands: To stop, let go of &lt;key&gt; *before* letting go of Shift. Otherwise, the robot will continue rotating even after you let go of both Shift and &lt;key&gt;.

| Key     | Cartesian      |
| ------- | ---------- |
| w       | -linear y  |
| s       | +linear y  |
| a       | -linear x  |
| d       | +linear x  |
| r       | -linear z  |
| f       | +linear z  |
| q       | -angular z |
| e       | +angular z |
| shift+w | +angular x |
| shift+s | -angular x |
| shift+a | -angular y |
| shift+d | +angular y |

Press 't' to toggle between slow and fast mode.

View and edit key mappings and speed settings in `AICKeyboardJointTeleop` and `AICKeyboardJointTeleopConfig` in `aic_teleop.py`.

##### SpaceMouse

:warning: Note: In our experience, SpaceMouse teleoperation was laggier than keyboard teleoperation.

We used a 3Dconnexion SpaceMouse with the [pyspacemouse](https://github.com/JakubAndrysek/PySpaceMouse?tab=readme-ov-file#dependencies) library. To enable USB permissions, you may need to add the following to your `/etc/udev/rules.d/99-spacemouse.rules`:
``` bash
# Apply to all hidraw nodes for 3Dconnexion devices
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
# Apply to the USB device itself
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
```
and then run
``` bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
View and edit axis mappings and speed settings in `AICSpaceMouseTeleop` and `AICSpaceMouseTeleopConfig` in `aic_teleop.py`.

#### Joint space control

| Key | Joint          |
| --- | -------------- |
| q   | -shoulder_pan  |
| a   | +shoulder_pan  |
| w   | -shoulder_lift |
| s   | +shoulder_lift |
| e   | -elbow         |
| d   | +elbow         |
| r   | -wrist_1       |
| f   | +wrist_1       |
| t   | -wrist_2       |
| g   | +wrist_2       |
| y   | -wrist_3       |
| h   | +wrist_3       |

Press 'u' to toggle between slow and fast mode.

View and edit key mappings and speed settings in `AICKeyboardEETeleop` and `AICKeyboardEETeleopConfig` in `aic_teleop.py`.

### Recording Training Data

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --dataset.repo_id=<hf-repo> \
  --dataset.single_task=<task-prompt> \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

:warning: Note (same as with `lerobot-teleoperate` above): In addition to setting `--teleop.type` you must set `--robot.teleop_target_mode` because the `AICRobotAICController` class needs to know which type of actions to send to the controller and it doesn't have access to `--teleop.type`.

Upon starting the command, you may see `WARN   Watchdog Validator ThreadId(13) zenoh_shm::watchdog::periodic_task: Some("Watchdog Validator")` which is safe to ignore; just look for `INFO ... ls/utils.py:227 Recording episode 0`.

LeRobot recording keys:

| Key         | Command          |
| ----------- | ---------------- |
| Right Arrow | Next episode     |
| Left Arrow  | Cancel current episode and re-record |
| ESC         | Stop recording   |

<!-- TODO: lerobot-record doesn't load the hil processor to handle teleop events (lerobot bug?) -->

### Automated Data Collection via CheatCode

The `cheatcode` teleoperator replays the scripted approach→descend→hold
trajectory from `aic_example_policies/ros/CheatCode.py` as an in-process
action source for `lerobot-record`, with Ornstein–Uhlenbeck perturbations
on the commanded pose to reproduce teleop-style overshoot-and-correct
behavior. Actions are recorded as Cartesian **pose targets**
(`--robot.teleop_target_mode=pose`) using `MODE_POSITION`.

**Prerequisites:**
1. Bringup must be launched with ground-truth TFs on
   (`ground_truth:=true`) so `{cable}/{plug}_link` and
   `{task_board}/{port}_link` are published.
2. **Strongly recommended:** launch `episode_reset_node` with
   `randomize_start_pose:=true` so each episode starts from a different
   robot home pose (±6 cm XYZ). Without this, every episode has an
   identical initial observation and the dataset is useless for training.
3. The task (cable/plug/module/port names) must be passed to the teleop;
   pass them as CLI flags (defaults shown) or adjust for your task.

**Example** (use `aic-record`, **not** `lerobot-record` — `aic-record` is the
in-repo wrapper that calls `robot.reset()` between episodes, which is what
triggers `randomize_start_pose`):
```bash
cd ~/ws_aic/src/aic
pixi run aic-record \
  --robot.type=aic_controller --robot.id=aic \
  --robot.teleop_target_mode=pose \
  --teleop.type=cheatcode --teleop.id=aic \
  --teleop.cable_name=cable_0 --teleop.plug_name=sfp_tip \
  --teleop.target_module_name=nic_card_mount_0 --teleop.port_name=sfp_port_0 \
  --teleop.approach_noise_xyz_m=0.004 \
  --teleop.descent_noise_xyz_m=0.001 \
  --dataset.repo_id=local/aic_cheatcode \
  --dataset.single_task="insert sfp cable" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=25 \
  --dataset.push_to_hub=false \
  --play_sounds=false
```

The teleop runs a state machine: APPROACH (ramps to a pose ~20 cm above
the port over ~5 s) → DESCEND (z_offset decreases each tick) → HOLD
(maintains the final pose after insertion succeeds). Episode boundaries
are driven by `--dataset.episode_time_s`; size it to comfortably cover
approach + descend + a dwell margin.

Success detection (plug–port xy distance < 2 mm, |z| < 1 mm held for
~10 ticks) gates the transition to HOLD and is logged; it does not by
itself terminate the lerobot episode.

### Training

Once you have your LeRobot dataset, you can follow the [LeRobot tutorials](https://huggingface.co/docs/lerobot/en/index) for training.

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=your_policy_type \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```
