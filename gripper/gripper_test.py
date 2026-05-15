#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *

# ── Motor / protocol constants ────────────────────────────────────────────────
MY_DXL                      = 'X_SERIES'

ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_POSITION       = 132
ADDR_PROFILE_VELOCITY       = 112

DXL_MINIMUM_POSITION_VALUE  = 0
DXL_MAXIMUM_POSITION_VALUE  = 4095

PROTOCOL_VERSION            = 2.0
DXL_ID                      = 1
DEVICENAME                  = '/dev/ttyUSB0'
BAUDRATE                    = 57600

TORQUE_ENABLE               = 1
TORQUE_DISABLE              = 0
DXL_MOVING_STATUS_THRESHOLD = 20

# ── Unit helpers ──────────────────────────────────────────────────────────────

def deg_to_pos(deg: float) -> int:
    """Convert degrees [0, 360) to raw encoder position [0, 4095]."""
    return int((deg / 360.0) * DXL_MAXIMUM_POSITION_VALUE)

def pos_to_deg(pos: int) -> float:
    """Convert raw encoder position [0, 4095] to degrees [0, 360)."""
    return (pos / DXL_MAXIMUM_POSITION_VALUE) * 360.0

# ── Controller class ──────────────────────────────────────────────────────────

class DynamixelController:

    GRIPPER_TRAVEL_DEG = 60   # degrees rotated between open and closed

    def __init__(self):
        self.port   = PortHandler(DEVICENAME)
        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.home_position_deg: float | None = None
        self._gripper_open: bool | None = None   # None = unknown until calibrated

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _open_port(self) -> None:
        if not self.port.openPort():
            raise IOError("Failed to open port")
        print("Port opened")

    def _set_baudrate(self) -> None:
        if not self.port.setBaudRate(BAUDRATE):
            raise IOError("Failed to set baudrate")
        print("Baudrate set")

    def _set_torque(self, enable: bool) -> None:
        value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        result, error = self.packet.write1ByteTxRx(
            self.port, DXL_ID, ADDR_TORQUE_ENABLE, value
        )
        self._check_comm(result, error)
        print(f"Torque {'enabled' if enable else 'disabled'}")

    def _check_comm(self, result: int, error: int) -> bool:
        if result != COMM_SUCCESS:
            print(f"Comm error: {self.packet.getTxRxResult(result)}")
            return False
        if error != 0:
            print(f"Packet error: {self.packet.getRxPacketError(error)}")
            return False
        return True

    def read_position_deg(self) -> float | None:
        """Read the current motor position in degrees. Returns None on error."""
        pos, result, error = self.packet.read4ByteTxRx(
            self.port, DXL_ID, ADDR_PRESENT_POSITION
        )
        if not self._check_comm(result, error):
            return None
        return pos_to_deg(pos)

    # ── Startup ───────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open port and set baudrate."""
        self._open_port()
        self._set_baudrate()
        print("Motor connected and ready")

    def calibrate(self) -> float:
        """
        Record the current position as the home / zero reference.

        The motor should be manually placed at its desired home position
        before calling this. The recorded angle is stored in
        ``self.home_position_deg`` and returned.
        """
        print("\n── Calibration ──────────────────────────────────────────────")
        self._set_torque(False)
        print("Move the motor to the HOME (closed) position, then press any key...")
        getch()
        self._set_torque(True)

        home = self.read_position_deg()
        if home is None:
            raise RuntimeError("Could not read position during calibration")

        self.home_position_deg = home
        self._gripper_open = False   # home = closed position
        print(f"Home position recorded: {home:.2f}°  (gripper: CLOSED)")
        print("─────────────────────────────────────────────────────────────\n")
        return home

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_to_position(self, goal_deg: float, velocity: int = 140) -> bool:
        """
        Move to ``goal_deg`` at the given profile velocity.

        Returns True if the goal was reached, False otherwise.
        """
        self.packet.write4ByteTxRx(
            self.port, DXL_ID, ADDR_PROFILE_VELOCITY, velocity
        )

        goal_pos = deg_to_pos(goal_deg)
        self.packet.write4ByteTxRx(
            self.port, DXL_ID, ADDR_GOAL_POSITION, goal_pos
        )

        stuck       = 0
        last_pos_deg = None

        while True:
            current_pos_deg = self.read_position_deg()
            if current_pos_deg is None:
                print("Error reading position")
                break

            print(f"current: {current_pos_deg:.2f}°  goal: {goal_deg:.2f}°")

            # Stuck detection — motor hasn't moved between polls
            if last_pos_deg is not None:
                if abs(current_pos_deg - last_pos_deg) < 1e-3:
                    stuck += 1
                    print(f"  stuck count: {stuck}")
                else:
                    stuck = 0          # reset if it moves again

            if stuck >= 5:
                print("Stopping — stuck condition detected")
                break

            last_pos_deg = current_pos_deg

            if abs(current_pos_deg - goal_deg) < 1.0:
                print("Reached goal position")
                break

            time.sleep(0.1)

        # Final position check
        final_deg = self.read_position_deg()
        if final_deg is not None and abs(final_deg - goal_deg) < 5.0:
            print(f"Motion complete  ({final_deg:.2f}°)")
            return True
        else:
            print(f"Motion incomplete — final pos: {final_deg}°, goal: {goal_deg}°")
            return False

    def move_to_home(self, velocity: int = 140) -> bool:
        """Return the motor to the calibrated home position."""
        if self.home_position_deg is None:
            raise RuntimeError("No home position set — run calibrate() first")
        print(f"Returning to home ({self.home_position_deg:.2f}°)")
        return self.move_to_position(self.home_position_deg, velocity)

    def open_gripper(self, velocity: int = 140) -> None:
        """Rotate clockwise by GRIPPER_TRAVEL_DEG to open. No-op if already open."""
        if self._gripper_open is None:
            raise RuntimeError("Gripper state unknown — run calibrate() first")
        if self._gripper_open:
            print("Gripper already open — skipping")
            return
        if self.home_position_deg is None:
            raise RuntimeError("No home position set — run calibrate() first")
        open_deg = (self.home_position_deg - self.GRIPPER_TRAVEL_DEG) % 360.0
        print(f"Opening gripper → {open_deg:.2f}°")
        success = self.move_to_position(open_deg, velocity)
        if success:
            self._gripper_open = True

    def close_gripper(self, velocity: int = 140) -> None:
        """Rotate anti-clockwise by GRIPPER_TRAVEL_DEG to close. No-op if already closed."""
        if self._gripper_open is None:
            raise RuntimeError("Gripper state unknown — run calibrate() first")
        if not self._gripper_open:
            print("Gripper already closed — skipping")
            return
        print(f"Closing gripper → {self.home_position_deg:.2f}°")
        success = self.move_to_home(velocity)
        if success:
            self._gripper_open = False

    def control_loop(self) -> None:
        """
        Interactive key loop after calibration.
          o  — open gripper
          c  — close gripper
          ESC — quit
        """
        print("\n── Gripper control ──────────────────────────────────────────")
        print("  o  → open   |  c  → close   |  ESC → quit")
        print("─────────────────────────────────────────────────────────────")
        while True:
            key = getch()
            if key == chr(0x1b):
                break
            elif key == 'o':
                self.open_gripper()
            elif key == 'c':
                self.close_gripper()
            else:
                print(f"Unknown key '{key}' — use o / c / ESC")
                continue
            state = "OPEN" if self._gripper_open else "CLOSED"
            print(f"  gripper state: {state}\n")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def disconnect(self) -> None:
        """Disable torque and close the port."""
        self._set_torque(False)
        self.port.closePort()
        print("Port closed")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    motor = DynamixelController()
    motor.connect()
    motor.calibrate()
    motor.control_loop()
    motor.close_gripper()   # ensure closed before power-down
    motor.disconnect()
