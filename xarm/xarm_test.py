import os
import sys
import math
import time
import numpy as np

# Add project root and current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xarm_controller import XArmController
from src.xarm_sim import XArmSim


def check_collision(pose, gripper_length=220.0, table_z=0.0):
    """
    Approximates the gripper as a line segment of `gripper_length`.
    Computes the lowest point (either flange or tip) and checks if it hits the table.
    pose: [x, y, z, roll, pitch, yaw] (in mm and degrees)
    Returns: (is_colliding, flange_z, tip_z, lowest_z)
    """
    if not pose or len(pose) < 6:
        return False, 0, 0, 0

    x, y, z, roll, pitch, yaw = pose[:6]

    # Convert degrees to radians
    r = math.radians(roll)
    p = math.radians(pitch)
    yw = math.radians(yaw)

    # Precompute trig functions
    sx, cx = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(yw), math.cos(yw)

    # Z-axis direction vector (based on Extrinsic XYZ / Intrinsic ZYX sequence)
    # This matches the xArm's rotation convention.
    z_dir = np.array([cy * sp * cx + sy * sx, sy * sp * cx - cy * sx, cp * cx])

    p_flange = np.array([x, y, z])
    p_tip = p_flange + z_dir * gripper_length

    # Treat gripper as a straight line, finding the lowest Z point
    min_z = min(p_flange[2], p_tip[2])
    is_colliding = min_z <= table_z

    return is_colliding, p_flange[2], p_tip[2], min_z


def main():
    ARM_IP = "192.168.1.242"
    URDF_PATH = (
        "/home/wignesh/Documents/rrc/test/gym-xarm/gym_xarm/envs/urdf/xarm7.urdf"
    )

    print(f"Connecting to xArm at {ARM_IP}...")
    arm = XArmController(ARM_IP)
    sim = XArmSim(urdf_path=URDF_PATH)

    # Robot-specific constants
    GRIPPER_LENGTH = 220.0
    TABLE_Z = 0.0

    try:
        # 1. Read real arm position
        curr_angles = arm.get_servo_angle()
        if curr_angles is None:
            print("Failed to get current angles from robot.")
            return

        print(f"Current Joint Angles: {curr_angles}")

        # Safety Check: Initial Pose
        curr_pose = arm.get_position()
        if curr_pose:
            is_col, f_z, t_z, min_z = check_collision(
                curr_pose, GRIPPER_LENGTH, TABLE_Z
            )
            print(f"[Initial Check] Flange Z: {f_z:.1f}, Tip Z: {t_z:.1f}")
            if is_col:
                print(
                    f"ERROR: Initial pose is colliding with table (Z <= {TABLE_Z}). Quitting."
                )
                return

        # 2. Set simulation to that position
        sim.set_joint_positions(curr_angles)

        # 3. Simulate movement (e.g., +10cm on global Z)
        final_angles = sim.simulate_trajectory(
            curr_angles, target_pos_offset=[0, 0, 0.1], steps=240
        )
        print("\nTrajectory simulated in PyBullet.")

        # Safety Check: Goal Pose
        # We can ask the arm API for the FK of the final simulated angles
        code, target_pose = arm.arm.get_forward_kinematics(final_angles)
        if code == 0:
            is_col, f_z, t_z, min_z = check_collision(
                target_pose, GRIPPER_LENGTH, TABLE_Z
            )
            print(f"[Goal Check] Flange Z: {f_z:.1f}, Tip Z: {t_z:.1f}")
            if is_col:
                print(
                    f"ERROR: Goal pose will collide with table (Z <= {TABLE_Z}). Quitting."
                )
                return
        else:
            print("Warning: Could not compute FK for final angles to check collision.")

        # 4. Ask user for permission to execute on real robot
        user_input = (
            input(
                "Simulation complete & safety checks passed. Execute on real robot? (yes/no): "
            )
            .strip()
            .lower()
        )

        if user_input == "yes":
            print("Executing on real robot...")
            arm.set_servo_angle(final_angles, speed=30, wait=True)
            print("Movement complete.")

            while True:
                final_choice = (
                    input("\n[r] → Return to HOME position | [q] → Quit: ")
                    .strip()
                    .lower()
                )
                if final_choice == "r":
                    # Safety check for HOME pose
                    code, home_pose = arm.arm.get_forward_kinematics(
                        [0, 0, 0, 0, 0, 0, 0]
                    )
                    if code == 0:
                        is_col, f_z, t_z, min_z = check_collision(
                            home_pose, GRIPPER_LENGTH, TABLE_Z
                        )
                        if is_col:
                            safe_z = (
                                TABLE_Z + GRIPPER_LENGTH + 20.0
                            )  # 20mm clearance above table
                            print(
                                f"Default home collides (Tip Z: {t_z:.1f}). Calculating safe home offset..."
                            )
                            print(
                                f"Moving to Safe Home (Cartesian Z = {safe_z:.1f})..."
                            )
                            safe_home_pose = list(home_pose[:6])
                            safe_home_pose[2] = safe_z
                            arm.set_position(*safe_home_pose, speed=30, wait=True)
                            break

                    print("Returning to default HOME position...")
                    arm.move_gohome(speed=30, wait=True)
                    break
                elif final_choice == "q":
                    break
        else:
            print("Execution cancelled.")

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        arm.disconnect()
        sim.close()


if __name__ == "__main__":
    main()
