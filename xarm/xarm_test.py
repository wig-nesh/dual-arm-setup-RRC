import os
import sys
import time
import numpy as np

# Add project root and current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xarm_controller import XArmController
from src.xarm_sim import XArmSim


def main():
    ARM_IP = "192.168.1.242"
    URDF_PATH = (
        "/home/wignesh/Documents/rrc/test/gym-xarm/gym_xarm/envs/urdf/xarm7.urdf"
    )

    print(f"Connecting to xArm at {ARM_IP}...")
    arm = XArmController(ARM_IP)
    sim = XArmSim(urdf_path=URDF_PATH)

    try:
        # 1. Read real arm position
        curr_angles = arm.get_servo_angle()
        if curr_angles is None:
            print("Failed to get current angles from robot.")
            return

        print(f"Current Joint Angles: {curr_angles}")

        # 2. Set simulation to that position
        sim.set_joint_positions(curr_angles)

        # 3. Simulate movement (e.g., +10cm on global Z)
        # Note: offsets in sim are in meters
        final_angles = sim.simulate_trajectory(
            curr_angles, target_pos_offset=[0, 0, 0.1], steps=240
        )

        print("\nTrajectory simulated in PyBullet.")

        # 4. Ask user for permission to execute on real robot
        user_input = (
            input("Simulation complete. Execute on real robot? (yes/no): ")
            .strip()
            .lower()
        )

        if user_input == "yes":
            print("Executing on real robot...")
            # We move to the final joint configuration found by IK in simulation
            arm.set_servo_angle(final_angles, speed=30, wait=True)
            print("Movement complete.")

            while True:
                final_choice = (
                    input("\n[r] → Return to HOME position | [q] → Quit: ")
                    .strip()
                    .lower()
                )
                if final_choice == "r":
                    print("Returning to HOME position...")
                    arm.move_gohome(speed=30, wait=True)
                    break
                elif final_choice == "q":
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
