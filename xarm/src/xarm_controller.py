import numpy as np
from xarm.wrapper import XArmAPI


class XArmController:
    def __init__(self, ip):
        """
        Initialize the xArm controller.

        Args:
            ip (str): IP address of the xArm.
        """
        self.arm = XArmAPI(ip)
        self.ip = ip
        self.setup()

    def setup(self):
        """Initial setup for the arm."""
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(state=0)  # Ready state
        print(f"xArm at {self.ip} initialized.")

    def get_position(self):
        """
        Get current position.

        Returns:
            list: [x, y, z, roll, pitch, yaw]
        """
        code, pos = self.arm.get_position()
        if code == 0:
            return pos
        else:
            print(f"Error getting position: {code}")
            return None

    def set_position(self, x, y, z, roll, pitch, yaw, speed=30, wait=True, mvacc=1000):
        """
        Move the arm to a specific position.

        Args:
            x, y, z: Position in mm.
            roll, pitch, yaw: Orientation in degrees.
            speed: Movement speed (mm/s).
            wait: Whether to wait for motion to complete.
            mvacc: Acceleration (mm/s^2).
        """
        code = self.arm.set_position(
            x, y, z, roll, pitch, yaw, speed=speed, wait=wait, mvacc=mvacc
        )
        if code != 0:
            print(f"Error setting position: {code}")
        return code

    def get_servo_angle(self):
        """
        Get current joint angles.

        Returns:
            list: [j1, j2, j3, j4, j5, j6, j7] in degrees.
        """
        code, angles = self.arm.get_servo_angle()
        if code == 0:
            return angles
        else:
            print(f"Error getting servo angles: {code}")
            return None

    def set_servo_angle(self, angles, speed=30, wait=True):
        """Sets the servo angles."""
        # The SDK expects 'angle' as the parameter name for the list of angles
        code = self.arm.set_servo_angle(angle=angles, speed=speed, wait=wait)
        return code

    def move_gohome(self, speed=30, wait=True):
        """Move the arm to its home position."""
        code = self.arm.move_gohome(speed=speed, wait=wait)
        if code != 0:
            print(f"Error moving to home: {code}")
        return code

    def disconnect(self):
        """Stop motion and disconnect."""
        self.arm.disconnect()
        print(f"xArm at {self.ip} disconnected.")
