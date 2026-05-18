import os
import sys
import cv2
import numpy as np
import open3d as o3d
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.realsense_camera import RealSenseCamera

# Match the board config from main_pipeline.py
MARKER_LENGTH = 0.056
BOARD_WIDTH = 0.197  # edge-to-edge
BOARD_HEIGHT = 0.259  # edge-to-edge

# Center-to-center spacing
spacing_x = (BOARD_WIDTH - MARKER_LENGTH) / 2
spacing_y = (BOARD_HEIGHT - MARKER_LENGTH) / 3

# Build marker world positions (all 12 markers centered around origin)
half_extent_x = (BOARD_WIDTH - MARKER_LENGTH) / 2
half_extent_y = (BOARD_HEIGHT - MARKER_LENGTH) / 2

MARKER_WORLD = {}
for row in range(4):
    for col in range(3):
        marker_id = row * 3 + col
        x = -half_extent_x + col * spacing_x
        y = -half_extent_y + row * spacing_y
        MARKER_WORLD[marker_id] = np.array([x, y, 0.0])


def create_pointcloud(color, depth, intrinsics):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["ppx"]
    cy = intrinsics["ppy"]
    h, w = depth.shape

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth / 1000.0
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    mask = (pts[:, 2] > 0) & (pts[:, 2] < 2.5)
    return pts[mask], colors[mask]


def main():
    print("Initializing RealSense...")
    camera = RealSenseCamera()
    camera.start()

    try:
        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            # Detect markers
            success, T, num_markers = camera.detect_aruco_markers(
                color, MARKER_WORLD, MARKER_LENGTH
            )

            vis = color.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = camera.aruco_detector.detectMarkers(gray)

            # Depth visualization
            depth_vis = depth.astype(np.float32)
            depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Draw on RGB
            if ids is not None:
                # Draw markers
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                if success:
                    # Draw axes
                    K, dist = camera.get_camera_matrix()
                    # Need original solvePnP result for drawFrameAxes (world→cam)
                    obj_pts = []
                    img_pts = []
                    half = MARKER_LENGTH / 2
                    local_corners = np.array(
                        [
                            [-half, -half, 0],
                            [half, -half, 0],
                            [half, half, 0],
                            [-half, half, 0],
                        ],
                        dtype=np.float32,
                    )

                    for i, mid in enumerate(ids.flatten()):
                        if mid in MARKER_WORLD:
                            center = MARKER_WORLD[mid]
                            obj_pts.append(local_corners + center)
                            img_pts.append(corners[i][0])

                    if len(obj_pts) >= 2:
                        obj_pts = np.vstack(obj_pts).astype(np.float32)
                        img_pts = np.vstack(img_pts).astype(np.float32)
                        cv2.cornerSubPix(
                            gray,
                            img_pts.reshape(-1, 1, 2),
                            (5, 5),
                            (-1, -1),
                            (
                                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                30,
                                0.001,
                            ),
                        )
                        _, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts, K, dist, cv2.SOLVEPNP_ITERATIVE
                        )
                        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.1)

                    info = f"Pose OK! Markers: {num_markers}"
                else:
                    info = f"Markers: {num_markers} (not enough for pose)"
            else:
                info = "No markers detected"

            # Stack RGB and Depth side by side
            combined = np.hstack([vis, depth_vis])
            cv2.putText(
                combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                combined,
                "RGB",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined,
                "Depth",
                (vis.shape[1] + 10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("ArUco Pose Test", combined)

            key = cv2.waitKey(1)
            if key == ord(" ") and success:
                intrinsics = camera.get_intrinsics()
                break
            elif key == ord("q"):
                camera.stop()
                cv2.destroyAllWindows()
                return

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print(f"\nBoard pose detected with {num_markers} markers!")
    print(f"Camera→World translation: {T[:3, 3]}")

    # Build pointcloud and visualize
    print("Building pointcloud...")
    pts, colors = create_pointcloud(color, depth, intrinsics)

    # Transform to world frame
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_w = (T @ pts_h.T).T[:, :3]

    # Filter points within 0.5m radius of board center
    colors = colors  # keep original colors reference
    dist = np.linalg.norm(pts_w, axis=1)
    radius_mask = dist < 0.5
    pts_w = pts_w[radius_mask]
    colors = colors[radius_mask]
    print(f"After 0.5m radius filter: {len(pts_w)} points")

    # Scale by 8x
    pts_scaled = pts_w * 8.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_scaled)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Coordinate frame at board origin, scaled 8x
    board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    board_frame.scale(8.0, center=(0, 0, 0))

    print("\nVisualizing pointcloud in board frame:")
    print(f"- Pointcloud at board origin (after camera→world transform)")
    print(f"- RGB axes at board center (origin of board coordinate frame)")
    print(f"- Z-up (blue) should point perpendicular to board surface")
    print(f"- Red = X, Green = Y, Blue = Z")
    o3d.visualization.draw_geometries([pcd, board_frame], window_name="Board Frame PCD")


if __name__ == "__main__":
    main()
