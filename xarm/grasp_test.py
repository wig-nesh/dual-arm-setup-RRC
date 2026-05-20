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

SPEED = 40  # mm/s
APPROACH_STEPS = 20  # interpolation steps for approach / retreat

POSITION_THRESHOLD_MM = 5.0

# Transform from the saved/model board frame into the robot execution board frame.
# This must be the single source of truth for both visualization and execution.
OBJECT_ROTATION_Z_DEG = -90.0

# ── Collision config ───────────────────────────────────────────────────────────
GRIPPER_LENGTH_MM = 220.0  # Total length from flange to very tip for collision checks
FLANGE_TO_MODEL_BASE_MM = 150.0  # Distance from arm flange to the center base of gripper fingers (model output pose)
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


def get_T_robot_board_model_board():
    """
    Map saved/model board coordinates into the robot board coordinates used for
    base calibration, visualization, and execution.
    """
    R_z = rotation_matrix(0, 0, OBJECT_ROTATION_Z_DEG)
    T_rot_z = np.eye(4)
    T_rot_z[:3, :3] = R_z

    T_flip_y = np.eye(4)
    T_flip_y[1, 1] = -1.0

    return T_flip_y @ T_rot_z


def get_T_model_grasp_robot_grasp():
    """
    Model grasps use local Y as the approach axis. xArm execution uses local Z.
    This is a local-frame correction, so it is post-multiplied onto grasp poses.
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(-90, 0, 0)
    return T


def ensure_right_handed_pose(T):
    """
    A single-axis board flip mirrors coordinates and can make grasp rotations
    left-handed. The robot can only execute proper rotations, so preserve the
    local approach axis (Z) and flip local X if needed.
    """
    T_fixed = T.copy()
    if np.linalg.det(T_fixed[:3, :3]) < 0:
        T_fixed[:3, 0] *= -1.0
    return T_fixed


def model_grasp_to_robot_board_mm(T_model_board_grasp_scaled):
    """
    Convert a saved model grasp into the exact board frame used by robot
    execution. Input translation is scaled by 8; output translation is mm.
    """
    T_model_board_grasp_mm = T_model_board_grasp_scaled.copy()
    T_model_board_grasp_mm[:3, 3] /= 8.0
    T_model_board_grasp_mm[:3, 3] *= 1000.0

    T_robot_board_grasp = (
        get_T_robot_board_model_board()
        @ T_model_board_grasp_mm
        @ get_T_model_grasp_robot_grasp()
    )
    return ensure_right_handed_pose(T_robot_board_grasp)


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


def pose_to_matrix(pose):
    """Convert [x, y, z, roll, pitch, yaw] back to a 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T


def assert_pose_round_trip(label, T):
    pose = matrix_to_pose(T)
    T_round_trip = pose_to_matrix(pose)
    pos_err = np.linalg.norm(T[:3, 3] - T_round_trip[:3, 3])
    rot_err = np.linalg.norm(T[:3, :3] - T_round_trip[:3, :3])
    if pos_err > 1e-6 or rot_err > 1e-6:
        raise RuntimeError(
            f"{label} cannot be represented as an xArm pose "
            f"(pos_err={pos_err:.6f}mm, rot_err={rot_err:.6f})"
        )
    return pose


def eef_z_axis(pose):
    R = rotation_matrix(pose[3], pose[4], pose[5])
    return R[:, 2]


def compute_flange_pose(model_pose, offset=FLANGE_TO_MODEL_BASE_MM):
    """
    The model outputs the grasp pose for the BASE of the fingers.
    We need to offset this backwards along the local Z axis by the distance
    from the flange to the base of the fingers.
    """
    z_ax = eef_z_axis(model_pose)
    flange = list(model_pose)
    flange[0] -= z_ax[0] * offset
    flange[1] -= z_ax[1] * offset
    flange[2] -= z_ax[2] * offset
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


def visualize_scene(
    run_dir,
    T_robot_board_model_board,
    T_board_base7,
    T_board_base6,
    T_board_tip7,
    T_board_flange7,
    T_board_pre7,
    T_board_tip6,
    T_board_flange6,
    T_board_pre6,
):
    print("Visualizing scene in Open3D...")
    pcd_path = os.path.join(run_dir, "pcd_scaled.ply")
    if not os.path.exists(pcd_path):
        print(f"No pointcloud found at {pcd_path}")
        return

    pcd = o3d.io.read_point_cloud(pcd_path)
    # The saved pcd is scaled by 8 (1 unit = 0.125m). We want it in mm.
    # So 1 unit = 125mm.
    pcd.scale(125.0, center=(0, 0, 0))
    pcd.transform(T_robot_board_model_board)

    geometries = [pcd]

    board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    geometries.append(board_frame)

    def make_frame(T, size, color=None):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(T)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size / 4.0)
        sphere.transform(T)
        if color:
            sphere.paint_uniform_color(color)
        return [mesh, sphere]

    def make_line(T1, T2, color):
        points = [T1[:3, 3], T2[:3, 3]]
        lines = [[0, 1]]
        colors = [color]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    # Base 7 (Blue)
    geometries.extend(make_frame(T_board_base7, size=100.0, color=[0, 0, 1]))
    # Base 6 (Red)
    geometries.extend(make_frame(T_board_base6, size=100.0, color=[1, 0, 0]))

    # Grasp 7 & Pre 7 (Blue)
    geometries.extend(make_frame(T_board_flange7, size=30.0, color=[0, 0, 0.8]))
    geometries.extend(make_frame(T_board_pre7, size=30.0, color=[0.5, 0.5, 1]))
    geometries.append(make_line(T_board_pre7, T_board_tip7, [0, 0, 1]))

    # Grasp 6 & Pre 6 (Red)
    geometries.extend(make_frame(T_board_flange6, size=30.0, color=[0.8, 0, 0]))
    geometries.extend(make_frame(T_board_pre6, size=30.0, color=[1, 0.5, 0.5]))
    geometries.append(make_line(T_board_pre6, T_board_tip6, [1, 0, 0]))

    # Camera Frame (if available)
    marker_path = os.path.join(run_dir, "marker_transform.npy")
    if os.path.exists(marker_path):
        # marker_transform maps camera-frame points into the saved board frame:
        # P_board = T_board_camera @ P_camera. That matrix is already the
        # camera pose in board coordinates, so do not invert it for drawing.
        T_board_camera = np.load(marker_path)
        T_board_camera_mm = T_board_camera.copy()
        T_board_camera_mm[:3, 3] *= 1000.0
        T_board_camera_mm = T_robot_board_model_board @ T_board_camera_mm
        T_board_camera_mm = ensure_right_handed_pose(T_board_camera_mm)

        geometries.extend(
            make_frame(T_board_camera_mm, size=80.0, color=[0, 1, 0])
        )  # Green for Camera
        print("Included Camera Frame (Green) in visualization.")
    else:
        print(
            "No camera transform (marker_transform.npy) found in this run. Run main_pipeline again to save it."
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Scene Visualization (Blue=xArm7, Red=xArm6, Green=Camera)",
    )


def main():
    parser = argparse.ArgumentParser(description="Execute grasps from a run directory")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to run_data directory"
    )
    parser.add_argument(
        "--direct", action="store_true", help="Skip manual mode and execute directly"
    )
    parser.add_argument(
        "--grasp-idx",
        type=int,
        default=0,
        help="Index of the grasp pair to execute (default: 0)",
    )
    args = parser.parse_args()

    npz_path = os.path.join(args.run_dir, "grasps.npz")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path, allow_pickle=True)
    # Extract the requested grasp pair
    if "refined_grasp_pairs" in data and len(data["refined_grasp_pairs"]) > 0:
        pairs = data["refined_grasp_pairs"]
        if args.grasp_idx >= len(pairs) or args.grasp_idx < 0:
            print(
                f"Error: Requested grasp index {args.grasp_idx} is out of bounds (max {len(pairs) - 1})."
            )
            return
        top_pair = pairs[args.grasp_idx]
        grasps_scaled = [top_pair[0], top_pair[1]]
        print(f"Using Refined Grasp Pair Index: {args.grasp_idx} out of {len(pairs)}")
    elif "single_grasps" in data and len(data["single_grasps"]) >= 2:
        # Fallback to top 2 single grasps if no pairs exist
        grasps_scaled = [data["single_grasps"][0], data["single_grasps"][1]]
    else:
        print("Not enough grasps found in the npz file (need at least 2).")
        return

    grasps_scaled = np.array(grasps_scaled)

    # 1. Convert to the robot board frame in mm.
    T_robot_board_model_board = get_T_robot_board_model_board()
    grasps_board_mm = []
    for g in grasps_scaled:
        grasps_board_mm.append(model_grasp_to_robot_board_mm(g))

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

    # 3. Compute all grasp transformations in the robot board frame.
    T_model_to_flange = np.eye(4)
    T_model_to_flange[2, 3] = -FLANGE_TO_MODEL_BASE_MM
    T_flange_to_pre = np.eye(4)
    T_flange_to_pre[2, 3] = -PREGRASP_OFFSET_MM

    T_board_model7 = grasps_board_mm[idx_7]
    T_board_model6 = grasps_board_mm[idx_6]

    # Apply a rotation around the local Z (approach) axis to correct final "roll"
    # Note: Since model approach was Y, and we mapped it to Z via R_local_corr,
    # the approach axis is now Z in this local frame.

    # Arm-specific roll correction around local approach Z.
    R_approach_ax7 = np.eye(4)
    R_approach_ax7[:3, :3] = rotation_matrix(0, 0, 90.0)
    T_board_model7 = T_board_model7 @ R_approach_ax7

    R_approach_ax6 = np.eye(4)
    R_approach_ax6[:3, :3] = rotation_matrix(0, 0, 90.0)
    T_board_model6 = T_board_model6 @ R_approach_ax6

    T_board_flange7 = T_board_model7 @ T_model_to_flange
    T_board_pre7 = T_board_flange7 @ T_flange_to_pre

    T_board_flange6 = T_board_model6 @ T_model_to_flange
    T_board_pre6 = T_board_flange6 @ T_flange_to_pre

    # 4. Generate the exact base poses for execution.
    T_base7_flange = T_base7_board @ T_board_flange7
    T_base7_pre = T_base7_board @ T_board_pre7
    T_base6_flange = T_base6_board @ T_board_flange6
    T_base6_pre = T_base6_board @ T_board_pre6

    # 5. Reconstruct the visualization frames from the execution frames. This
    # makes Open3D show exactly what the xArm commands below will request.
    T_board_base7 = np.linalg.inv(T_base7_board)
    T_board_base6 = np.linalg.inv(T_base6_board)

    T_vis_flange7 = T_board_base7 @ T_base7_flange
    T_vis_pre7 = T_board_base7 @ T_base7_pre
    T_vis_flange6 = T_board_base6 @ T_base6_flange
    T_vis_pre6 = T_board_base6 @ T_base6_pre

    for label, expected, actual in [
        ("Arm1 flange", T_board_flange7, T_vis_flange7),
        ("Arm1 pregrasp", T_board_pre7, T_vis_pre7),
        ("Arm2 flange", T_board_flange6, T_vis_flange6),
        ("Arm2 pregrasp", T_board_pre6, T_vis_pre6),
    ]:
        if not np.allclose(expected, actual, atol=1e-6):
            delta = np.linalg.norm(expected[:3, 3] - actual[:3, 3])
            raise RuntimeError(
                f"{label} visualization/execution mismatch: {delta:.6f}mm"
            )

    visualize_scene(
        args.run_dir,
        T_robot_board_model_board,
        T_board_base7,
        T_board_base6,
        T_board_model7,
        T_vis_flange7,
        T_vis_pre7,
        T_board_model6,
        T_vis_flange6,
        T_vis_pre6,
    )

    grasp1 = assert_pose_round_trip("Arm1 grasp", T_base7_flange)
    pre1 = assert_pose_round_trip("Arm1 pregrasp", T_base7_pre)
    grasp2 = assert_pose_round_trip("Arm2 grasp", T_base6_flange)
    pre2 = assert_pose_round_trip("Arm2 pregrasp", T_base6_pre)

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
