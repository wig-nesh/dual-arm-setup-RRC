#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Add the project root to sys.path to allow running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import *
from gripper.src.gripper_controller import DynamixelGripper

# ── Keyboard input setup ──────────────────────────────────────────────────────
if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


# ── Configuration ─────────────────────────────────────────────────────────────
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

TRAVEL_DEG = 40

def main():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        print("Failed to open port")
        return
    if not port_handler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        return

    # L is ID 0, R is ID 1
    left_gripper = DynamixelGripper(port_handler, packet_handler, dxl_id=0, travel_deg=TRAVEL_DEG)
    right_gripper = DynamixelGripper(port_handler, packet_handler, dxl_id=1, travel_deg=TRAVEL_DEG)

    try:
        print("Starting calibration...")
        left_gripper.calibrate(getch)
        right_gripper.calibrate(getch)

        print("\n── Dual Gripper Control ─────────────────────────────────────")
        print("  L    → Toggle Left Gripper (ID 0)")
        print("  R    → Toggle Right Gripper (ID 1)")
        print("  ESC  → Quit")
        print("─────────────────────────────────────────────────────────────")

        while True:
            key = getch()
            if key == chr(0x1B):
                break
            elif key.lower() == "l":
                left_gripper.toggle()
            elif key.lower() == "r":
                right_gripper.toggle()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        left_gripper.close()
        right_gripper.close()
        left_gripper.disable_torque()
        right_gripper.disable_torque()
        port_handler.closePort()
        print("Done")


if __name__ == "__main__":
    main()
