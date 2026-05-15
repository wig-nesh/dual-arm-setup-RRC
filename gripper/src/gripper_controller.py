#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from dynamixel_sdk import *

# ── Motor / protocol constants ────────────────────────────────────────────────
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

DXL_MINIMUM_POSITION_VALUE = 0
DXL_MAXIMUM_POSITION_VALUE = 4095

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# ── Unit helpers ──────────────────────────────────────────────────────────────


def deg_to_pos(deg: float) -> int:
    """Convert degrees [0, 360) to raw encoder position [0, 4095]."""
    return int((deg / 360.0) * DXL_MAXIMUM_POSITION_VALUE)


def pos_to_deg(pos: int) -> float:
    """Convert raw encoder position [0, 4095] to degrees [0, 360)."""
    return (pos / DXL_MAXIMUM_POSITION_VALUE) * 360.0


# ── Controller class ──────────────────────────────────────────────────────────


class DynamixelGripper:
    def __init__(
        self,
        port_handler,
        packet_handler,
        dxl_id,
        travel_deg: float = 30.0,
        device_name: str = "/dev/ttyUSB0",
    ):
        self.port = port_handler
        self.packet = packet_handler
        self.dxl_id = dxl_id
        self.travel_deg = travel_deg
        self.device_name = device_name
        self.home_position_deg: float | None = None
        self._gripper_open: bool | None = None

    def _set_torque(self, enable: bool) -> None:
        value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        result, error = self.packet.write1ByteTxRx(
            self.port, self.dxl_id, ADDR_TORQUE_ENABLE, value
        )
        self._check_comm(result, error)

    def _check_comm(self, result: int, error: int) -> bool:
        if result != COMM_SUCCESS:
            print(f"[ID:{self.dxl_id}] Comm error: {self.packet.getTxRxResult(result)}")
            return False
        if error != 0:
            print(
                f"[ID:{self.dxl_id}] Packet error: {self.packet.getRxPacketError(error)}"
            )
            return False
        return True

    def read_position_deg(self) -> float | None:
        """Read the current motor position in degrees [0, 360)."""
        pos, result, error = self.packet.read4ByteTxRx(
            self.port, self.dxl_id, ADDR_PRESENT_POSITION
        )
        if not self._check_comm(result, error):
            return None
        return pos_to_deg(pos)

    def calibrate(self, getch_func) -> float:
        """
        Record current position as home (CLOSED).
        """
        print(f"\n── Calibrating Gripper ID: {self.dxl_id} ──")
        self._set_torque(False)
        print(f"Move gripper {self.dxl_id} to CLOSED position, then press any key...")
        getch_func()

        self._set_torque(True)
        time.sleep(0.1)

        home = self.read_position_deg()
        if home is None:
            raise RuntimeError(f"Could not read position for ID {self.dxl_id}")

        self.home_position_deg = home
        self._gripper_open = False
        print(f"ID {self.dxl_id} home recorded at: {home:.2f}° (CLOSED)")
        return home

    def move_to_position(self, goal_deg: float, velocity: int = 140) -> bool:
        self.packet.write4ByteTxRx(
            self.port, self.dxl_id, ADDR_PROFILE_VELOCITY, velocity
        )

        goal_pos = deg_to_pos(goal_deg)
        curr = self.read_position_deg()
        print(
            f"[ID:{self.dxl_id}] MOVE: current={curr:.2f}°, goal={goal_deg:.2f}°, pos={goal_pos}"
        )

        self.packet.write4ByteTxRx(self.port, self.dxl_id, ADDR_GOAL_POSITION, goal_pos)

        stuck = 0
        last_pos_deg = None

        while True:
            current_pos_deg = self.read_position_deg()
            if current_pos_deg is None:
                break

            if last_pos_deg is not None:
                if abs(current_pos_deg - last_pos_deg) < 1e-3:
                    stuck += 1
                else:
                    stuck = 0

            if stuck >= 5:
                print(f"[ID:{self.dxl_id}] Stopped - stuck")
                break

            last_pos_deg = current_pos_deg
            if abs(current_pos_deg - goal_deg) < 1.0:
                break
            time.sleep(0.1)

        final_deg = self.read_position_deg()
        return final_deg is not None and abs(final_deg - goal_deg) < 5.0

    def open(self, velocity: int = 140) -> None:
        if self._gripper_open:
            return
        if self.home_position_deg is None:
            raise RuntimeError(f"ID {self.dxl_id} not calibrated")

        open_deg = (self.home_position_deg - self.travel_deg) % 360.0

        if self.move_to_position(open_deg, velocity):
            self._gripper_open = True
            print(f"[ID:{self.dxl_id}] OPEN")

    def close(self, velocity: int = 140) -> None:
        if self._gripper_open == False:
            return
        if self.home_position_deg is None:
            raise RuntimeError(f"ID {self.dxl_id} not calibrated")

        if self.move_to_position(self.home_position_deg, velocity):
            self._gripper_open = False
            print(f"[ID:{self.dxl_id}] CLOSED")

    def toggle(self):
        if self._gripper_open:
            self.close()
        else:
            self.open()

    def disable_torque(self):
        self._set_torque(False)
