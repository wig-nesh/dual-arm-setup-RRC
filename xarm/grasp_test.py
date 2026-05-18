import os
import sys
import math
import time
import threading
import numpy as np
import argparse
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xarm_controller import XArmController
from dynamixel_sdk import PortHandler, PacketHandler
from gripper.src.gripper_controller import DynamixelGripper

# ── Keyboard input ─────────────────────────────────────────────────────────────
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


# ── Config ─────────────────────────────────────────────────────────────────────
IP1 = "192.168.1.242"  # xArm7 (Blue)
IP2 = "192.168.1.175"  # xArm6 (Red)

# ── Base to Board Transforms (Update these!) ───────────────────────────────────
# Transform from the Arm Base to the Board (Object) frame in mm.
# The measurements were provided in cm, so we multiply by 10.
XARM7_BASE_TO_BOARD_X = 320.0
XARM7_BASE_TO_BOARD_Y = -297.5

XARM6_BASE_TO_BOARD_X = 322.5
XARM6_BASE_TO_BOARD_Y = 295.0


def get_T_base_board(x_offset_mm, y_offset_mm):
    T = np.eye(4)
    T[0, 3] = x_offset_mm
    T[1, 3] = y_offset_mm
    return T


GRIPPER_DEVICENAME = "/dev/ttyUSB0"
GRIPPER_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
TRAVEL_DEG = 40

# Pre-grasp: how far back along EEF -Z to step (mm)
PREGRASP_OFFSET_MM = 100.0

# Primitive distances in global Z
LIFT_MM = 100.0

SPEED = 30  # mm/s
APPROACH_STEPS = 20  # interpolation steps for approach / retreat

POSITION_THRESHOLD_MM = 5.0

# ── Collision config ───────────────────────────────────────────────────────────
GRIPPER_LENGTH_MM = 220.0
TABLE_Z_MM = 0.0
COLLISION_CLEARANCE_MM = 5.0

# ── Rotation / frame helpers ──────────────────────────────────────────────────


def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def matrix_to_pose(T):
    """Convert 4x4 matrix to [x, y, z, roll, pitch, yaw] (mm, degrees)."""
    x, y, z = T[:3, 3]
    sy = math.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = math.atan2(T[1, 0], T[0, 0])
    else:
        roll = math.atan2(-T[1, 2], T[1, 1])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = 0
    return [x, y, z, math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def eef_z_axis(pose):
    R = rotation_matrix(pose[3], pose[4], pose[5])
    return R[:, 2]


def compute_flange_pose(tip_pose, gripper_length=GRIPPER_LENGTH_MM):
    """
    The model outputs the grasp pose for the TIP of the gripper.
    We need to offset this backwards along the local Z axis by the gripper length
    to find the pose the arm's FLANGE needs to reach.
    """
    z_ax = eef_z_axis(tip_pose)
    flange = list(tip_pose)
    flange[0] -= z_ax[0] * gripper_length
    flange[1] -= z_ax[1] * gripper_length
    flange[2] -= z_ax[2] * gripper_length
    return flange


def compute_pregrasp(flange_pose, offset_mm=PREGRASP_OFFSET_MM):
    z_ax = eef_z_axis(flange_pose)
    pre = list(flange_pose)
    pre[0] -= z_ax[0] * offset_mm
    pre[1] -= z_ax[1] * offset_mm
    pre[2] -= z_ax[2] * offset_mm
    return pre


# ── Collision checking ────────────────────────────────────────────────────────


def check_table_collision(
    pose,
    gripper_length=GRIPPER_LENGTH_MM,
    table_z=TABLE_Z_MM,
    clearance=COLLISION_CLEARANCE_MM,
):
    if pose is None or len(pose) < 6:
        return False, 0.0, 0.0, 0.0

    x, y, z, roll, pitch, yaw = pose[:6]
    z_dir = eef_z_axis(pose)

    p_flange = np.array([x, y, z])
    p_tip = p_flange + z_dir * gripper_length

    flange_z = p_flange[2]
    tip_z = p_tip[2]
    min_z = min(flange_z, tip_z)

    safe_z = table_z + clearance
    is_colliding = min_z <= safe_z

    return is_colliding, flange_z, tip_z, min_z


def assert_safe(arm_label, pose):
    is_col, f_z, t_z, min_z = check_table_collision(pose)
    if is_col:
        raise CollisionError(
            f"[{arm_label}] Table collision detected! "
            f"Flange Z={f_z:.1f}mm  Tip Z={t_z:.1f}mm  "
            f"Min Z={min_z:.1f}mm  ≤  safe limit={TABLE_Z_MM + COLLISION_CLEARANCE_MM:.1f}mm"
        )


class CollisionError(Exception):
    pass


# ── Motion helpers ────────────────────────────────────────────────────────────


def move_both(arm1, arm2, pose1, pose2, speed=SPEED, wait=True):
    assert_safe(arm1.ip, pose1)
    assert_safe(arm2.ip, pose2)

    def _move(arm, pose):
        arm.set_position(*pose[:6], speed=speed, wait=wait)

    t1 = threading.Thread(target=_move, args=(arm1, pose1))
    t2 = threading.Thread(target=_move, args=(arm2, pose2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def set_manual_mode(arm, enable=True):
    if enable:
        arm.arm.set_mode(2)
        arm.arm.set_state(0)
        print(f"[{arm.ip}] Manual mode ON")
    else:
        arm.arm.set_mode(0)
        arm.arm.set_state(0)
        print(f"[{arm.ip}] Manual mode OFF — position control")


def pose_distance(current_pose, target_pose):
    return float(np.linalg.norm(np.array(current_pose[:3]) - np.array(target_pose[:3])))


# ── Pose streamer ─────────────────────────────────────────────────────────────


class SingleArmStreamer:
    def __init__(self, arm, target_pose, arm_name):
        self.arm = arm
        self.target = target_pose
        self.arm_name = arm_name
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        tx, ty, tz = self.target[:3]
        while not self._stop.is_set():
            p = self.arm.get_position()
            if p:
                cx, cy, cz = p[:3]
                ex = tx - cx
                ey = ty - cy
                ez = tz - cz
                dist = math.sqrt(ex**2 + ey**2 + ez**2)

                print(
                    f"\r[{self.arm_name}] "
                    f"CUR:({cx:6.1f}, {cy:6.1f}, {cz:6.1f}) | "
                    f"TGT:({tx:6.1f}, {ty:6.1f}, {tz:6.1f}) | "
                    f"ERR:(X:{ex:6.1f}, Y:{ey:6.1f}, Z:{ez:6.1f}) | "
                    f"Δ:{dist:5.1f}mm    ",
                    end="",
                    flush=True,
                )
            time.sleep(0.1)


# ── Grasp primitive ────────────────────────────────────────────────────────────


def execute_grasp_primitive(
    arm1, arm2, grasp1, grasp2, pre1, pre2, left_gripper, right_gripper
):
    print("\n[Primitive] Moving to exact PRE-GRASP...")
    move_both(arm1, arm2, pre1, pre2)

    print("\n[Primitive] Waiting 3 seconds...")
    time.sleep(3.0)

    print("[Primitive] Opening grippers...")
    left_gripper.toggle()
    right_gripper.toggle()
    time.sleep(1.0)

    print("[Primitive] Moving FORWARD to grasp pose...")
    move_both(arm1, arm2, grasp1, grasp2)

    print("[Primitive] Closing grippers...")
    left_gripper.toggle()
    right_gripper.toggle()
    time.sleep(1.0)

    print(f"[Primitive] Lifting +{LIFT_MM:.0f}mm...")
    p1 = arm1.get_position()
    p2 = arm2.get_position()
    lift1 = list(p1)
    lift1[2] += LIFT_MM
    lift2 = list(p2)
    lift2[2] += LIFT_MM
    move_both(arm1, arm2, lift1, lift2)

    print("[Primitive] Holding for 2 seconds...")
    time.sleep(2.0)

    print("[Primitive] Lowering back down...")
    move_both(arm1, arm2, p1, p2)

    print("[Primitive] Opening grippers...")
    left_gripper.toggle()
    right_gripper.toggle()

    print("[Primitive] Done.")


# ── Loading and Setup ──────────────────────────────────────────────────────────


def visualize_grasps(run_dir, grasp1_idx, grasp2_idx, all_grasps_scaled):
    print("Visualizing selected grasps in Open3D...")
    pcd_path = os.path.join(run_dir, "pcd_scaled.ply")
    if not os.path.exists(pcd_path):
        print(f"No pointcloud found at {pcd_path}")
        return

    pcd = o3d.io.read_point_cloud(pcd_path)
    geometries = [pcd]

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)  # scaled
    geometries.append(base_frame)

    # Helper to draw a frame
    def make_grasp_mesh(T_scaled, color):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        mesh.transform(T_scaled)
        # Give it a block representing the gripper
        box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.5)
        box.translate([-0.1, -0.1, -0.25])
        box.transform(T_scaled)
        box.paint_uniform_color(color)
        return [mesh, box]

    # Blue for xArm7 (grasp1)
    geometries.extend(make_grasp_mesh(all_grasps_scaled[grasp1_idx], [0, 0, 1]))
    # Red for xArm6 (grasp2)
    geometries.extend(make_grasp_mesh(all_grasps_scaled[grasp2_idx], [1, 0, 0]))

    o3d.visualization.draw_geometries(
        geometries, window_name="Assigned Grasps (Blue=xArm7, Red=xArm6)"
    )


def main():
    parser = argparse.ArgumentParser(description="Execute grasps from a run directory")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to run_data directory"
    )
    args = parser.parse_args()

    npz_path = os.path.join(args.run_dir, "grasps.npz")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path, allow_pickle=True)
    # Extract the top scoring pair
    if "refined_grasp_pairs" in data and len(data["refined_grasp_pairs"]) > 0:
        top_pair = data["refined_grasp_pairs"][0]
        grasps_scaled = [top_pair[0], top_pair[1]]
    elif "single_grasps" in data and len(data["single_grasps"]) >= 2:
        # Fallback to top 2 single grasps if no pairs exist
        grasps_scaled = [data["single_grasps"][0], data["single_grasps"][1]]
    else:
        print("Not enough grasps found in the npz file (need at least 2).")
        return

    grasps_scaled = np.array(grasps_scaled)

    # 1. Convert to board frame in mm
    grasps_board_mm = []
    for g in grasps_scaled:
        g_board_m = g.copy()
        g_board_m[:3, 3] /= 8.0
        g_board_mm = g_board_m.copy()
        g_board_mm[:3, 3] *= 1000.0
        grasps_board_mm.append(g_board_mm)

    # 2. Assign based on distance to base
    T_base7_board = get_T_base_board(XARM7_BASE_TO_BOARD_X, XARM7_BASE_TO_BOARD_Y)
    T_base6_board = get_T_base_board(XARM6_BASE_TO_BOARD_X, XARM6_BASE_TO_BOARD_Y)

    g0_mm = grasps_board_mm[0]
    g1_mm = grasps_board_mm[1]

    # Distance of Grasp 0 to both arms
    dist_g0_to_7 = np.linalg.norm((T_base7_board @ g0_mm)[:3, 3])
    dist_g0_to_6 = np.linalg.norm((T_base6_board @ g0_mm)[:3, 3])

    # Distance of Grasp 1 to both arms
    dist_g1_to_7 = np.linalg.norm((T_base7_board @ g1_mm)[:3, 3])
    dist_g1_to_6 = np.linalg.norm((T_base6_board @ g1_mm)[:3, 3])

    # Assign arms to minimize total distance
    if (dist_g0_to_7 + dist_g1_to_6) < (dist_g1_to_7 + dist_g0_to_6):
        idx_7, idx_6 = 0, 1
        dist_7, dist_6 = dist_g0_to_7, dist_g1_to_6
    else:
        idx_7, idx_6 = 1, 0
        dist_7, dist_6 = dist_g1_to_7, dist_g0_to_6

    print(f"Assigned Grasp {idx_7} to xArm7 (Blue, {IP1}) - Dist: {dist_7:.1f}mm")
    print(f"Assigned Grasp {idx_6} to xArm6 (Red, {IP2}) - Dist: {dist_6:.1f}mm")

    # 3. Visualize
    visualize_grasps(args.run_dir, idx_7, idx_6, grasps_scaled)

    # 4. Generate the exact base poses
    T_base7_grasp_tip = T_base7_board @ grasps_board_mm[idx_7]
    T_base6_grasp_tip = T_base6_board @ grasps_board_mm[idx_6]

    grasp1_tip = matrix_to_pose(T_base7_grasp_tip)
    grasp2_tip = matrix_to_pose(T_base6_grasp_tip)

    grasp1 = compute_flange_pose(grasp1_tip)
    grasp2 = compute_flange_pose(grasp2_tip)

    pre1 = compute_pregrasp(grasp1)
    pre2 = compute_pregrasp(grasp2)

    for label, pose in [
        ("Arm1 grasp", grasp1),
        ("Arm1 pregrasp", pre1),
        ("Arm2 grasp", grasp2),
        ("Arm2 pregrasp", pre2),
    ]:
        is_col, f_z, t_z, min_z = check_table_collision(pose)
        status = "⚠ COLLIDES" if is_col else "OK"
        print(
            f"  {label:20s}: flange_z={f_z:.1f}  tip_z={t_z:.1f}  min_z={min_z:.1f}  [{status}]"
        )
        if is_col:
            print(f"  ERROR: {label} collides with table. Aborting.")
            return

    print("=" * 70)
    print(f"  Grasp Arm1  (own):   {[f'{v:.1f}' for v in grasp1]}")
    print(f"  Pre-grasp Arm1 (own):{[f'{v:.1f}' for v in pre1]}")
    print(f"  Grasp Arm2  (own):   {[f'{v:.1f}' for v in grasp2]}")
    print(f"  Pre-grasp Arm2 (own):{[f'{v:.1f}' for v in pre2]}")
    print("=" * 70)

    # ── Connect arms ──
    print(f"\nConnecting to arms {IP1} and {IP2}...")
    arm1 = XArmController(IP1)
    arm2 = XArmController(IP2)

    for arm in [arm1, arm2]:
        curr = arm.get_position()
        if curr:
            is_col, f_z, t_z, min_z = check_table_collision(curr)
            if is_col:
                print(
                    f"  ERROR: [{arm.ip}] current pose already collides with table. Aborting."
                )
                arm1.disconnect()
                arm2.disconnect()
                return

    # ── Connect grippers ──
    port_handler = PortHandler(GRIPPER_DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    if not port_handler.openPort() or not port_handler.setBaudRate(GRIPPER_BAUDRATE):
        print("Failed to open gripper port or set baudrate")
        return

    left_gripper = DynamixelGripper(
        port_handler, packet_handler, dxl_id=0, travel_deg=TRAVEL_DEG
    )
    right_gripper = DynamixelGripper(
        port_handler, packet_handler, dxl_id=1, travel_deg=TRAVEL_DEG
    )

    print("Calibrating grippers...")
    left_gripper.calibrate(getch)
    right_gripper.calibrate(getch)

    try:
        # --- Arm 1 Manual Mode ---
        set_manual_mode(arm1, enable=True)
        print("\n── Manual Mode: Arm 1 (Blue / xArm7) ─────────────────────────")
        print("  Guide Arm 1 to its pre-grasp pose.")
        print("  Press [N] to lock Arm 1 and move to Arm 2.   Press [ESC] to abort.")
        print("──────────────────────────────────────────────────────────────────\n")

        streamer1 = SingleArmStreamer(arm1, pre1, "Arm 1")
        streamer1.start()

        while True:
            key = getch()
            if key.lower() == "n":
                streamer1.stop()
                set_manual_mode(arm1, enable=False)
                print("\n[Arm 1 locked.]")
                break
            elif key == chr(0x1B):
                streamer1.stop()
                print("\nESC — aborting.")
                return

        time.sleep(1)

        # --- Arm 2 Manual Mode ---
        set_manual_mode(arm2, enable=True)
        print("\n── Manual Mode: Arm 2 (Red / xArm6) ──────────────────────────")
        print("  Guide Arm 2 to its pre-grasp pose.")
        print("  Press [P] to execute grasp primitive.   Press [ESC] to abort.")
        print("──────────────────────────────────────────────────────────────────\n")

        streamer2 = SingleArmStreamer(arm2, pre2, "Arm 2")
        streamer2.start()

        while True:
            key = getch()
            if key.lower() == "p":
                streamer2.stop()
                set_manual_mode(arm2, enable=False)
                print("\n[P] pressed — switching to position control...")
                break
            elif key == chr(0x1B):
                streamer2.stop()
                print("\nESC — aborting.")
                return

        time.sleep(0.5)

        execute_grasp_primitive(
            arm1, arm2, grasp1, grasp2, pre1, pre2, left_gripper, right_gripper
        )

    except CollisionError as e:
        print(f"\n\n  *** COLLISION ABORT *** {e}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        set_manual_mode(arm1, enable=False)
        set_manual_mode(arm2, enable=False)
        left_gripper.close()
        right_gripper.close()
        left_gripper.disable_torque()
        right_gripper.disable_torque()
        port_handler.closePort()
        arm1.disconnect()
        arm2.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
