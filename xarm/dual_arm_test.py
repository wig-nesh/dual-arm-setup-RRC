import os
import sys
import time
import math
import threading
import numpy as np

# Add project root and current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xarm_controller import XArmController


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
    z_dir = np.array([cy * sp * cx + sy * sx, sy * sp * cx - cy * sx, cp * cx])

    p_flange = np.array([x, y, z])
    p_tip = p_flange + z_dir * gripper_length

    # Treat gripper as a straight line, finding the lowest Z point
    min_z = min(p_flange[2], p_tip[2])
    is_colliding = min_z <= table_z

    return is_colliding, p_flange[2], p_tip[2], min_z


def safe_move_gohome(arm, gripper_length, table_z):
    """Helper to safely send an arm to its home position."""
    code, home_pose = arm.arm.get_forward_kinematics([0, 0, 0, 0, 0, 0, 0])
    if code == 0:
        is_col, f_z, t_z, min_z = check_collision(home_pose, gripper_length, table_z)
        if is_col:
            safe_z = table_z + gripper_length + 20.0  # 20mm clearance
            print(
                f"[{arm.ip}] Default home collides (Tip Z: {t_z:.1f}). Moving to Safe Home (Z = {safe_z:.1f})..."
            )
            safe_home_pose = list(home_pose[:6])
            safe_home_pose[2] = safe_z
            arm.set_position(*safe_home_pose, speed=30, wait=True)
            return

    print(f"[{arm.ip}] Returning to default HOME position...")
    arm.move_gohome(speed=30, wait=True)


def move_arm_relative_z(arm, offset_mm, speed=30):
    """Helper to move a single arm relative to its current Z position."""
    curr_pos = arm.get_position()
    if curr_pos:
        target_pos = list(curr_pos)
        target_pos[2] += offset_mm
        print(f"Arm at {arm.ip} moving to {target_pos[2]}mm (Z)")
        arm.set_position(*target_pos, speed=speed, wait=True)


def main():
    IP1 = "192.168.1.242"
    IP2 = "192.168.1.175"

    print(f"Connecting to arms: {IP1} and {IP2}...")
    arm1 = XArmController(IP1)
    arm2 = XArmController(IP2)

    GRIPPER_LENGTH = 220.0
    TABLE_Z = 0.0

    try:
        # Check initial poses for safety
        for arm in [arm1, arm2]:
            curr_pose = arm.get_position()
            if curr_pose:
                is_col, f_z, t_z, min_z = check_collision(
                    curr_pose, GRIPPER_LENGTH, TABLE_Z
                )
                if is_col:
                    print(
                        f"ERROR: Initial pose of arm {arm.ip} is colliding with table (Z={min_z:.1f} <= {TABLE_Z}). Quitting."
                    )
                    return

        # Ask for confirmation
        user_input = (
            input(
                "\nInitial safety checks passed. Move both arms up by 100mm? (yes/no): "
            )
            .strip()
            .lower()
        )
        if user_input != "yes":
            print("Operation cancelled.")
            return

        print("Executing simultaneous move...")

        # Use threads to start movement on both arms at the same time
        t1 = threading.Thread(target=move_arm_relative_z, args=(arm1, 100))
        t2 = threading.Thread(target=move_arm_relative_z, args=(arm2, 100))

        t1.start()
        t2.start()

        # Wait for both to finish
        t1.join()
        t2.join()

        print("Movement complete.")

        input("\nPress Enter to return both to HOME...")
        print("Returning to HOME...")

        t_home1 = threading.Thread(
            target=safe_move_gohome, args=(arm1, GRIPPER_LENGTH, TABLE_Z)
        )
        t_home2 = threading.Thread(
            target=safe_move_gohome, args=(arm2, GRIPPER_LENGTH, TABLE_Z)
        )

        t_home1.start()
        t_home2.start()

        t_home1.join()
        t_home2.join()

        print("Arms returned to HOME.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        arm1.disconnect()
        arm2.disconnect()


if __name__ == "__main__":
    main()
