import os
import sys
import cv2
import numpy as np
import open3d as o3d
import argparse
import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realsense.src.realsense_camera import RealSenseCamera
from client.src.sam import SAMClient
from client.src.model import ModelClient

# ArUco marker configuration (3x4 board with 12 markers)
# Layout:
#  0  1  2
#  3  4  5
#  6  7  8
#  9 10 11
MARKER_LENGTH = 0.056  # 5.6cm markers
BOARD_WIDTH = 0.197  # edge-to-edge (3 markers wide)
BOARD_HEIGHT = 0.259  # edge-to-edge (4 markers tall)

# Convert edge-to-edge to center-to-center spacing
spacing_x = (BOARD_WIDTH - MARKER_LENGTH) / 2  # 3 markers = 2 intervals
spacing_y = (BOARD_HEIGHT - MARKER_LENGTH) / 3  # 4 markers = 3 intervals

# Build marker world positions (all 12 markers centered around origin)
half_extent_x = (BOARD_WIDTH - MARKER_LENGTH) / 2
half_extent_y = (BOARD_HEIGHT - MARKER_LENGTH) / 2

MARKER_WORLD = {}
for row in range(4):  # 4 rows
    for col in range(3):  # 3 columns
        marker_id = row * 3 + col  # 0-11
        x = -half_extent_x + col * spacing_x
        y = -half_extent_y + row * spacing_y
        MARKER_WORLD[marker_id] = np.array([x, y, 0.0])

# Global state for SAM selection
state = {
    "mode": "box",  # default
    "points": [],
    "box": None,
    "drawing": False,
}


def mouse_callback(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        if state["mode"] == "point":
            state["points"] = [(x, y)]
        elif state["mode"] == "box":
            state["drawing"] = True
            state["box"] = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"] and state["mode"] == "box":
            state["box"][2] = x
            state["box"][3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if state["drawing"] and state["mode"] == "box":
            state["drawing"] = False
            state["box"][2] = x
            state["box"][3] = y


def depth_to_pcd(
    color_image, depth_image, intrinsics, mask=None, transform=None, camera=None
):
    if camera is not None and getattr(camera, "use_ffs", False):
        # --- FFS MANUAL ALIGNMENT PATH ---
        # depth_image is float32 (meters) in Left IR frame
        # color_image is BGR in RGB frame
        # mask is in RGB frame

        K_ir = camera.K_ir
        K_rgb = camera.K_rgb
        R_ir2rgb = camera.R_ir_to_rgb
        t_ir2rgb = camera.t_ir_to_rgb

        H, W = depth_image.shape
        v, u = np.indices((H, W))

        # Valid depth mask
        valid = depth_image > 0
        u = u[valid]
        v = v[valid]
        z = depth_image[valid]

        # 1. Deproject to Left IR 3D frame
        fx, fy = K_ir[0, 0], K_ir[1, 1]
        cx, cy = K_ir[0, 2], K_ir[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_ir = np.stack((x, y, z), axis=-1)

        # 2. Transform 3D points from Left IR frame to RGB frame
        pts_rgb_frame = pts_ir @ R_ir2rgb.T + t_ir2rgb

        # 3. Project 3D points into RGB image to get UV coordinates for color sampling
        fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]
        cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]

        u_rgb = np.round(
            (pts_rgb_frame[:, 0] * fx_rgb / pts_rgb_frame[:, 2]) + cx_rgb
        ).astype(int)
        v_rgb = np.round(
            (pts_rgb_frame[:, 1] * fy_rgb / pts_rgb_frame[:, 2]) + cy_rgb
        ).astype(int)

        # Filter bounds
        H_c, W_c = color_image.shape[:2]
        in_bounds = (u_rgb >= 0) & (u_rgb < W_c) & (v_rgb >= 0) & (v_rgb < H_c)

        pts_rgb_frame = pts_rgb_frame[in_bounds]
        u_rgb = u_rgb[in_bounds]
        v_rgb = v_rgb[in_bounds]

        # 4. If SAM mask is provided, filter points
        # The mask is in the RGB frame, so we use u_rgb, v_rgb
        if mask is not None:
            in_mask = mask[v_rgb, u_rgb] > 0
            pts_rgb_frame = pts_rgb_frame[in_mask]
            u_rgb = u_rgb[in_mask]
            v_rgb = v_rgb[in_mask]

        if len(pts_rgb_frame) == 0:
            return np.array([]), np.array([])

        # 5. Sample colors
        colors_bgr = color_image[v_rgb, u_rgb]
        colors_rgb = colors_bgr[:, ::-1] / 255.0  # Normalize to 0-1 float

        # 6. Apply ArUco transform to World Frame
        # The ArUco detection ran on the RGB image, so `transform` maps from RGB frame to World frame.
        if transform is not None:
            pts_h = np.hstack((pts_rgb_frame, np.ones((pts_rgb_frame.shape[0], 1))))
            pts_w = (transform @ pts_h.T).T[:, :3]
            points = pts_w
        else:
            points = pts_rgb_frame

        return points, colors_rgb

    # --- NATIVE DEPTH (OPEN3D) PATH ---
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["ppx"],
        cy=intrinsics["ppy"],
    )

    if mask is not None:
        depth_image = depth_image.copy()
        depth_image[mask == 0] = 0

    # If depth image is float32, it's likely from FFS and already in meters.
    # Open3D's RGBDImage uses depth_scale=1000.0 by default, which divides the values by 1000.
    # To fix this, if it's float32, we convert it to uint16 millimeters before passing.
    if depth_image.dtype == np.float32:
        depth_image = (depth_image * 1000.0).astype(np.uint16)

    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Transform points from camera frame to world frame if transform provided
    if transform is not None:
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        points_w = (transform @ points_h.T).T[:, :3]
        points = points_w

    return points, colors


def visualize_grasps(
    pcd_xyz, pcd_rgb, grasp_pairs_np, pair_scores_np, gripper_mesh_path
):
    if grasp_pairs_np is None or grasp_pairs_np.shape[0] == 0:
        print("No grasp pairs to visualize.")
        return

    # Sort pairs by score (highest first)
    if pair_scores_np is not None:
        sort_idx = np.argsort(pair_scores_np)[::-1]
        grasp_pairs_np = grasp_pairs_np[sort_idx]
        pair_scores_np = pair_scores_np[sort_idx]

    # Load base meshes
    base_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1
    )  # Fallback
    if os.path.exists(gripper_mesh_path):
        base_gripper = o3d.io.read_triangle_mesh(gripper_mesh_path)
        base_gripper.compute_vertex_normals()
        base_gripper.scale(8.0, center=(0, 0, 0))
    else:
        print(
            f"Warning: Gripper mesh not found at {gripper_mesh_path}. Using coordinate frame fallback."
        )

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    g1 = o3d.geometry.TriangleMesh(base_gripper)
    g1.paint_uniform_color([0.8, 0.2, 0.2])
    g2 = o3d.geometry.TriangleMesh(base_gripper)
    g2.paint_uniform_color([0.2, 0.2, 0.8])

    f1 = o3d.geometry.TriangleMesh(base_frame)
    f2 = o3d.geometry.TriangleMesh(base_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    vis_state = {"idx": 0}

    def update_meshes(vis):
        idx = vis_state["idx"]
        pair = grasp_pairs_np[idx]
        score_str = (
            f" | Score: {pair_scores_np[idx]:.4f}" if pair_scores_np is not None else ""
        )
        print(f"Showing Grasp Pair {idx + 1}/{len(grasp_pairs_np)}{score_str}")

        g1.vertices = base_gripper.vertices
        g1.transform(pair[0])
        g1.compute_vertex_normals()

        g2.vertices = base_gripper.vertices
        g2.transform(pair[1])
        g2.compute_vertex_normals()

        f1.vertices = base_frame.vertices
        f1.transform(pair[0])
        f1.compute_vertex_normals()

        f2.vertices = base_frame.vertices
        f2.transform(pair[1])
        f2.compute_vertex_normals()

        vis.update_geometry(g1)
        vis.update_geometry(g2)
        vis.update_geometry(f1)
        vis.update_geometry(f2)

    def next_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] + 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    def prev_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] - 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dual Grasp Viewer (Right/Left Arrow to cycle)")

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(g1)
    vis.add_geometry(g2)
    vis.add_geometry(f1)
    vis.add_geometry(f2)

    # GLFW Key codes: 262 is Right Arrow, 263 is Left Arrow
    vis.register_key_callback(262, next_grasp)
    vis.register_key_callback(263, prev_grasp)

    print("\nOpening Open3D Visualization...")
    print("Red Gripper: Arm 1 | Blue Gripper: Arm 2")
    print("Use RIGHT and LEFT arrow keys to cycle through grasp pairs.")

    update_meshes(vis)
    vis.run()
    vis.destroy_window()


def visualize_single_grasps(
    pcd_xyz, pcd_rgb, single_grasps_np, grasp_scores_np, gripper_mesh_path
):
    """Visualize single grasps when no valid pairs are available."""
    if single_grasps_np is None or len(single_grasps_np) == 0:
        print("No single grasps to visualize.")
        return

    # Sort by score
    if grasp_scores_np is not None:
        sort_idx = np.argsort(grasp_scores_np)[::-1]
        single_grasps_np = single_grasps_np[sort_idx]
        grasp_scores_np = grasp_scores_np[sort_idx]

    # Load gripper mesh
    base_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    if os.path.exists(gripper_mesh_path):
        base_gripper = o3d.io.read_triangle_mesh(gripper_mesh_path)
        base_gripper.compute_vertex_normals()
        base_gripper.scale(8.0, center=(0, 0, 0))
    else:
        print(f"Warning: Gripper mesh not found, using coordinate frame fallback.")

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    gripper_mesh = o3d.geometry.TriangleMesh(base_gripper)
    gripper_mesh.paint_uniform_color([0.8, 0.2, 0.2])
    frame_mesh = o3d.geometry.TriangleMesh(base_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    vis_state = {"idx": 0}

    def update_mesh(vis):
        idx = vis_state["idx"]
        grasp = single_grasps_np[idx]
        score_str = (
            f" | Score: {grasp_scores_np[idx]:.4f}"
            if grasp_scores_np is not None
            else ""
        )
        print(f"Showing Single Grasp {idx + 1}/{len(single_grasps_np)}{score_str}")

        gripper_mesh.vertices = base_gripper.vertices
        gripper_mesh.transform(grasp)
        gripper_mesh.compute_vertex_normals()

        frame_mesh.vertices = base_frame.vertices
        frame_mesh.transform(grasp)
        frame_mesh.compute_vertex_normals()

        vis.update_geometry(gripper_mesh)
        vis.update_geometry(frame_mesh)

    def next_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] + 1) % len(single_grasps_np)
        update_mesh(vis)
        return False

    def prev_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] - 1) % len(single_grasps_np)
        update_mesh(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Single Grasp Viewer (Right/Left Arrow to cycle)")

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(gripper_mesh)
    vis.add_geometry(frame_mesh)

    vis.register_key_callback(262, next_grasp)
    vis.register_key_callback(263, prev_grasp)

    print("\nOpening Open3D Visualization...")
    print("Red: Single Grasp")
    print("Use RIGHT and LEFT arrow keys to cycle through grasps.")

    update_mesh(vis)
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline (No Robot Execution)"
    )
    parser.add_argument(
        "--sam-url",
        type=str,
        default="http://dualarm@orion.rrcx.tk:8000",
        help="SAM Server URL",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default="http://localhost:8000",
        help="Grasp Model Server URL",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="client/gripper.obj",
        help="Path to gripper mesh for vis",
    )
    parser.add_argument(
        "--ffs-url",
        type=str,
        default=None,
        help="FFS Server URL for depth. If omitted, uses native RealSense depth.",
    )
    args = parser.parse_args()

    # 1. RealSense Capture
    use_ffs = args.ffs_url is not None
    ffs_status = f"FFS at {args.ffs_url}" if use_ffs else "Native Depth"
    print(f"Initializing RealSense (using {ffs_status} at 640x480)...")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("run_data", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving run data to: {run_dir}")

    camera = RealSenseCamera(
        width=640, height=480, use_ffs=use_ffs, ffs_url=args.ffs_url
    )
    try:
        camera.start()
    except Exception as e:
        print(f"Failed to start RealSense: {e}")
        return

    print("Streaming RealSense... Press [SPACE] to capture a frame.")
    color_frame = None
    depth_frame = None
    intrinsics = None
    marker_transform = None  # Camera to world transform from ArUco

    while True:
        color_frame, depth_frame = camera.get_frames()
        if color_frame is None:
            continue

        # Detect markers for visualization overlay
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = camera.aruco_detector.detectMarkers(gray)

        vis_frame = color_frame.copy()

        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)

            # Try to draw frame axes for pose visualization
            K, dist = camera.get_camera_matrix()
            try:
                # Use marker_world for PnP to get pose
                half_marker = MARKER_LENGTH / 2
                marker_corners_local = np.array(
                    [
                        [-half_marker, -half_marker, 0],
                        [half_marker, -half_marker, 0],
                        [half_marker, half_marker, 0],
                        [-half_marker, half_marker, 0],
                    ],
                    dtype=np.float32,
                )

                obj_points = []
                img_points = []
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in MARKER_WORLD:
                        center = MARKER_WORLD[marker_id]
                        world_corners = marker_corners_local + center
                        obj_points.append(world_corners)
                        img_points.append(corners[i][0])

                if len(obj_points) >= 2:
                    obj_pts = np.vstack(obj_points).astype(np.float32)
                    img_pts = np.vstack(img_points).astype(np.float32)

                    # Refine corners
                    cv2.cornerSubPix(
                        gray,
                        img_pts.reshape(-1, 1, 2),
                        (5, 5),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )

                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, dist, cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        cv2.drawFrameAxes(vis_frame, K, dist, rvec, tvec, 0.1)
            except Exception as e:
                pass  # Skip axis drawing if it fails

        cv2.imshow("RealSense", vis_frame)
        key = cv2.waitKey(1)

        if key == ord(" "):
            # First capture: try to detect ArUco markers
            print("Detecting ArUco markers...")
            intrinsics = camera.get_intrinsics()

            success, marker_transform, num_markers = camera.detect_aruco_markers(
                color_frame, MARKER_WORLD, MARKER_LENGTH
            )

            if success:
                print(f"Marker pose detected using {num_markers} markers!")
                if num_markers < 4:
                    print(
                        "WARNING: Only a few markers detected. Pose may be less accurate."
                    )
                print(f"Translation: {marker_transform[:3, 3]}")
                R = marker_transform[:3, :3]
                print("Rotation matrix detected.")
            else:
                if num_markers > 0:
                    print(
                        f"Only {num_markers} markers detected (need at least 2). Using camera frame (no transform)."
                    )
                else:
                    print("No markers detected. Using camera frame (no transform).")

            break
        elif key == ord("q"):
            camera.stop()
            cv2.destroyAllWindows()
            return

    # Now show RGB + Depth view
    print("Streaming RGB+Depth... Press [SPACE] to capture.")
    while True:
        color_frame, depth_frame = camera.get_frames()
        if color_frame is None:
            continue

        # Normalize depth for display
        depth_vis = depth_frame.astype(np.float32)
        depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Stack RGB and Depth side by side
        combined = np.hstack([color_frame, depth_vis])
        cv2.putText(
            combined, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            combined,
            "Depth",
            (color_frame.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
        )

        cv2.imshow("RealSense", combined)
        key = cv2.waitKey(1)
        if key == ord(" "):
            break

    print("Refreshing ArUco pose on final RGBD capture...")
    success, refreshed_marker_transform, num_markers = camera.detect_aruco_markers(
        color_frame, MARKER_WORLD, MARKER_LENGTH
    )
    if success:
        marker_transform = refreshed_marker_transform
        print(f"Updated marker pose using {num_markers} markers.")
        print(f"Translation: {marker_transform[:3, 3]}")
    else:
        if marker_transform is not None:
            print(
                f"Could not refresh marker pose on final capture ({num_markers} markers). "
                "Using earlier marker pose."
            )
        elif num_markers > 0:
            print(
                f"Only {num_markers} markers detected on final capture. "
                "Using camera frame (no transform)."
            )
        else:
            print("No markers detected on final capture. Using camera frame (no transform).")

    camera.stop()
    cv2.destroyAllWindows()

    # Save raw captured data
    print("Saving raw frames and intrinsics...")
    cv2.imwrite(os.path.join(run_dir, "rgb.png"), color_frame)
    np.save(os.path.join(run_dir, "depth.npy"), depth_frame)
    with open(os.path.join(run_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=4)

    # 2. SAM Interaction
    print("\nCapture successful!")
    print("Select target using SAM:")
    print("  [B] to switch to Bounding Box mode (click and drag)")
    print("  [P] to switch to Point mode (click once)")
    print("  [SPACE] to confirm selection and query SAM")

    cv2.namedWindow("SAM Selection")
    cv2.setMouseCallback("SAM Selection", mouse_callback)

    while True:
        vis = color_frame.copy()

        if state["mode"] == "point" and len(state["points"]) > 0:
            cv2.circle(vis, state["points"][0], 5, (0, 255, 0), -1)
        elif state["mode"] == "box" and state["box"] is not None:
            x1, y1, x2, y2 = state["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("SAM Selection", vis)
        key = cv2.waitKey(10)

        if key == ord("p"):
            state["mode"] = "point"
            state["points"] = []
            state["box"] = None
            print("Mode: POINT")
        elif key == ord("b"):
            state["mode"] = "box"
            state["points"] = []
            state["box"] = None
            print("Mode: BOX")
        elif key == ord(" "):
            if (state["mode"] == "point" and len(state["points"]) > 0) or (
                state["mode"] == "box" and state["box"] is not None
            ):
                break
            else:
                print("Make a selection first!")

    cv2.destroyAllWindows()

    # Query SAM
    print(f"\nSending {state['mode']} prompt to SAM server at {args.sam_url}...")
    sam_client = SAMClient(args.sam_url)
    try:
        if state["mode"] == "point":
            mask = sam_client.predict_from_point(color_frame, state["points"][0])
        else:
            x1, y1, x2, y2 = state["box"]
            bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            mask = sam_client.predict_from_box(color_frame, bbox)

        # Preview Mask
        vis_mask = color_frame.copy()
        vis_mask[mask > 0] = [0, 255, 0]
        cv2.imshow("SAM Mask", vis_mask)
        cv2.imwrite(os.path.join(run_dir, "sam_mask.png"), mask)
        cv2.imwrite(os.path.join(run_dir, "sam_mask_vis.png"), vis_mask)
        print("Displaying mask. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"SAM prediction failed: {e}")
        return

    # 3. Project to PointCloud
    print("\nDeprojecting depth to 3D point cloud...")
    if marker_transform is not None:
        print(
            "Transforming points from camera frame to world frame using marker pose..."
        )
    xyz, rgb = depth_to_pcd(
        color_frame, depth_frame, intrinsics, mask, marker_transform, camera
    )
    print(f"Generated masked point cloud with {len(xyz)} points.")

    if len(xyz) == 0:
        print("Error: Point cloud is empty (check mask and depth alignment).")
        return

    # --- Debugging Stats ---
    unique_pts = np.unique(xyz, axis=0)
    print(f"\n--- Point Cloud Stats ---")
    print(f"Unique points: {len(unique_pts)} / {len(xyz)}")
    if len(xyz) > 0:
        print(f"XYZ Min : {xyz.min(axis=0)}")
        print(f"XYZ Max : {xyz.max(axis=0)}")
        print(f"XYZ Mean: {xyz.mean(axis=0)}")

    if mask is not None:
        masked_depths = depth_frame[mask > 0]
        if len(masked_depths) > 0:
            print(
                f"Depth values in mask - Min: {masked_depths.min()}, Max: {masked_depths.max()}, Mean: {masked_depths.mean():.2f}"
            )
        else:
            print("No valid depth values found inside the mask!")
    print(f"-------------------------\n")

    # Filter points within 0.5m radius of board center (in world frame)
    if marker_transform is not None:
        dist = np.linalg.norm(xyz, axis=1)
        radius_mask = dist < 0.5
        xyz = xyz[radius_mask]
        rgb = rgb[radius_mask]
        print(f"After 0.5m radius filter: {len(xyz)} points")

    if len(xyz) == 0:
        print("Error: No points within 0.5m of board center.")
        return

    # Downsample to 2048 points for model input
    print("Downsampling to 2048 points for model...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Remove statistical outliers (noise)
    print("Removing statistical outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    print(f"After outlier removal: {len(pcd.points)} points")

    # Voxel downsample first to even out density (optional, but good before FPS)
    # pcd = pcd.voxel_down_sample(voxel_size=0.005)

    current_n = len(pcd.points)
    if current_n >= 2048:
        # Farthest Point Sampling (FPS)
        print("Applying Farthest Point Sampling (FPS)...")
        pcd = pcd.farthest_point_down_sample(2048)
    else:
        # Random sample with replacement to get exactly 2048
        print("Not enough points for FPS, padding with random replacement...")
        idx = np.random.choice(current_n, 2048, replace=True)
        pcd = pcd.select_by_index(idx)

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    print(f"Downsampled to {len(xyz)} points.")

    print("Previewing 3D Pointcloud before sending...")

    # Scale points by 8x to match model expectations (MUST match what's sent to server)
    xyz_scaled = xyz * 8.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_scaled)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save pointcloud
    print("Saving processed pointcloud...")
    o3d.io.write_point_cloud(os.path.join(run_dir, "pcd_scaled.ply"), pcd)

    # Save marker transform if available
    if marker_transform is not None:
        np.save(os.path.join(run_dir, "marker_transform.npy"), marker_transform)

    # Add coordinate frame at origin (board center after points transformed to world frame)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    world_frame.scale(8.0, center=(0, 0, 0))

    if marker_transform is not None:
        print("Showing object frame (from marker transform) - scaled 8x")
    else:
        print("Showing camera frame - scaled 8x")

    o3d.visualization.draw_geometries([pcd, world_frame], window_name="Filtered PCD")

    # 4. Send to Grasp Model (already scaled)
    platform_height_meters = 0.08
    platform_height_scaled = (
        platform_height_meters * 8.0
    )  # Scale to match the 8x pointcloud
    print(
        f"\nSending pointcloud to Grasp Model Server at {args.model_url} with platform_height={platform_height_scaled} (scaled 8x)..."
    )
    model_client = ModelClient(args.model_url, timeout=120.0)
    try:
        results = model_client.predict(
            xyz_scaled, platform_height=platform_height_scaled
        )

        print("Saving grasp results...")
        np.savez(os.path.join(run_dir, "grasps.npz"), **results)

        if "refined_grasp_pairs" in results:
            grasp_pairs = results["refined_grasp_pairs"]
            pair_scores = results.get("pair_scores", None)
            print(f"Received {grasp_pairs.shape[0]} refined grasp pairs.")
            visualize_grasps(xyz_scaled, rgb, grasp_pairs, pair_scores, args.gripper)
        elif "single_grasps" in results:
            # Fallback: show single grasps when no pairs available
            single_grasps = results["single_grasps"]
            grasp_scores = results.get("grasp_scores", None)
            print(f"Received {single_grasps.shape[0]} single grasps (no valid pairs).")
            visualize_single_grasps(
                xyz_scaled, rgb, single_grasps, grasp_scores, args.gripper
            )
        else:
            print("Server returned success but no grasps in response.")
            print(f"Keys found: {list(results.keys())}")

    except Exception as e:
        print(f"Model prediction failed: {e}")


if __name__ == "__main__":
    main()
