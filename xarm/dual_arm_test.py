import os
import sys
import time
import threading

# Add project root and current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xarm_controller import XArmController


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

    try:
        # Ask for confirmation
        user_input = input("\nMove both arms up by 100mm? (yes/no): ").strip().lower()
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

        t_home1 = threading.Thread(target=arm1.move_gohome)
        t_home2 = threading.Thread(target=arm2.move_gohome)

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
