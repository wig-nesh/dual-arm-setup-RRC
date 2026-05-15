# Dual Arm Real Life Setup

This repository contains modular controllers and test scripts for a real-world robotic pipeline involving dual xArm7s, Dynamixel Grippers, RealSense RGB-D cameras, and remote SAM segmentation.

## Project Structure

- **`xarm/`**: xArm7 control and simulation.
  - `src/xarm_controller.py`: Main class for hardware interaction.
  - `src/xarm_sim.py`: PyBullet-based simulation for IK and trajectory visualization.
  - `xarm_test.py`: Validates motion in sim before executing on the real arm.
  - `dual_arm_test.py`: Simple script to move both arms up by 10cm simultaneously.
- **`gripper/`**: Dynamixel gripper control.
  - `src/gripper_controller.py`: Controller class for single/dual gripper setups.
  - `dual_gripper_test.py`: Interactive L/R toggle test (IDs 0 and 1).
- **`realsense/`**: Intel RealSense capture.
  - `src/realsense_camera.py`: RGB-D frame acquisition and alignment.
  - `realsense_test.py`: Live viewer and data capture (RGB, Depth, Intrinsics).
- **`client/`**: Remote inference clients.
  - `src/sam.py`: Client for remote SAM (Segment Anything Model) inference.
  - `webcam_sam_test.py`: Test script to run SAM on a local webcam feed.

## Hardware & 3D Printing

The gripper used in this project consists of custom 3D printed parts designed for Dynamixel X-series motors.

### Printing Instructions
The STL files are located in `gripper/3d_printed_parts/`. For a complete gripper, you need to print:
- **Base** (`base.stl`): The main chassis that mounts to the motor.
- **Gear** (`gear.stl`): The internal drive gear.
- **Left Tip** (`left_tip.stl`): The left-side finger.
- **Right Tip** (`right_tip.stl`): The right-side finger.
- **Cover** (`cover.stl` or `cover.3mf`): The protective housing.

## Setup Instructions

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for package management)

### Installation
1. Clone the repository and navigate to the root:
   ```bash
   git clone <repo-url>
   cd real_life_pipeline
   ```

2. Create environment and install dependencies:
   ```bash
   uv sync
   ```

### SAM Server (External)
The `SAMClient` requires a remote SAM server to be running. 
- The server implementation (e.g., using Flask + Ultralytics SAM) must be set up manually on a machine with appropriate GPU support.
- This repository **does not** manage server-side dependencies (like `ultralytics` or `torch`).

## Usage

### xArm Control
Run the single arm test with PyBullet visualization:
```bash
uv run xarm/xarm_test.py
```
Run the dual arm simultaneous movement test (no PyBullet yet):
```bash
uv run xarm/dual_arm_test.py
```

### Gripper Control
Test the dual gripper setup (Left: ID 0, Right: ID 1):
```bash
uv run gripper/dual_gripper_test.py
```
- `L`: Toggle Left
- `R`: Toggle Right

### RealSense Capture
Capture RGB-D data to the `data/` folder:
```bash
uv run realsense/realsense_test.py
```
- `SPACE`: Capture Frame
- `Q`: Quit

### SAM Client Test
Run interactive segmentation on your webcam feed:
```bash
uv run client/webcam_sam_test.py --url http://YOUR_SERVER_IP:8000 # dualarm@orion.rrcx.tk perhaps
```
- `Left Click`: Select point to segment.
- `Q`: Quit.

## Data Storage
Captured data is stored in the `data/` directory (git-ignored), organized by session name:
- `rgb/`: RGB images (.png)
- `depth/`: Raw depth maps (.npy)
- `intrinsics.json`: Camera calibration data.
