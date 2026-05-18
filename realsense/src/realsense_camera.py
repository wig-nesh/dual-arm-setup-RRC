import pyrealsense2 as rs
import numpy as np
import json
import os
import cv2


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, use_ffs=False, ffs_url=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width = width
        self.height = height
        self.fps = fps
        self.use_ffs = use_ffs

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        if self.use_ffs:
            self.config.enable_stream(
                rs.stream.infrared, 1, width, height, rs.format.y8, fps
            )
            self.config.enable_stream(
                rs.stream.infrared, 2, width, height, rs.format.y8, fps
            )
            # Try to disable emitter later in start() if we can
            if ffs_url:
                import sys
                import os

                sys.path.append(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
                )
                from client.src.ffs import FFSClient

                self.ffs_client = FFSClient(ffs_url)
            else:
                self.ffs_client = None
        else:
            self.config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, fps
            )
            self.align = rs.align(rs.stream.color)

        # ArUco detector setup
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

    def start(self):
        self.profile = self.pipeline.start(self.config)

        if self.use_ffs:
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 0)

            # Allow auto-exposure to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()

            left_stream = self.profile.get_stream(
                rs.stream.infrared, 1
            ).as_video_stream_profile()
            right_stream = self.profile.get_stream(
                rs.stream.infrared, 2
            ).as_video_stream_profile()
            color_stream = self.profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()

            intr = left_stream.get_intrinsics()
            self.K_ir = np.array(
                [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                dtype=np.float32,
            )
            extrinsics = left_stream.get_extrinsics_to(right_stream)
            self.baseline = abs(extrinsics.translation[0])

            # Get extrinsics from Left IR to RGB
            extr_ir_to_rgb = left_stream.get_extrinsics_to(color_stream)
            self.R_ir_to_rgb = np.array(extr_ir_to_rgb.rotation).reshape(3, 3)
            self.t_ir_to_rgb = np.array(extr_ir_to_rgb.translation).reshape(3)

            intr_color = color_stream.get_intrinsics()
            self.K_rgb = np.array(
                [
                    [intr_color.fx, 0, intr_color.ppx],
                    [0, intr_color.fy, intr_color.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        if self.use_ffs:
            color_frame = frames.get_color_frame()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not color_frame or not left_frame or not right_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            left_ir = np.asanyarray(left_frame.get_data())
            right_ir = np.asanyarray(right_frame.get_data())

            if self.ffs_client:
                try:
                    depth_image = self.ffs_client.predict_depth(
                        left_ir, right_ir, color_image, self.K_ir, self.baseline
                    )
                except Exception as e:
                    print(f"FFS prediction failed: {e}")
                    depth_image = np.zeros((self.height, self.width), dtype=np.float32)
            else:
                depth_image = np.zeros((self.height, self.width), dtype=np.float32)

            return color_image, depth_image

        else:
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image

    def get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }

    def get_camera_matrix(self):
        """Returns camera matrix K and distortion coefficients."""
        intr = self.get_intrinsics()
        K = np.array(
            [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]]
        )
        dist = np.array(intr["coeffs"])
        return K, dist

    def detect_aruco_markers(
        self, color_image, marker_world_positions, marker_length=0.05
    ):
        """
        Detect ArUco markers and estimate camera pose relative to marker board.

        Args:
            color_image: BGR image from camera
            marker_world_positions: dict mapping marker_id -> world position (np.array)
            marker_length: side length of markers in meters (default 0.05 = 5cm)

        Returns:
            (success, T, num_markers) where T is 4x4 transformation matrix (camera to world)
            If failed, returns (False, None, 0)
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return False, None, 0

        ids = ids.flatten()
        half_marker = marker_length / 2

        # Local corner positions (same for all markers)
        marker_corners_local = np.array(
            [
                [-half_marker, -half_marker, 0],
                [half_marker, -half_marker, 0],
                [half_marker, half_marker, 0],
                [-half_marker, half_marker, 0],
            ],
            dtype=np.float32,
        )

        # Build PnP data
        obj_points = []
        img_points = []

        for i, marker_id in enumerate(ids):
            if marker_id not in marker_world_positions:
                continue

            center = marker_world_positions[marker_id]
            world_corners = marker_corners_local + center
            img_corners = corners[i][0]

            obj_points.append(world_corners)
            img_points.append(img_corners)

        num_markers = len(obj_points)

        if num_markers < 2:
            return False, None, num_markers

        obj_points = np.vstack(obj_points).astype(np.float32)
        img_points = np.vstack(img_points).astype(np.float32)

        # Refine corners
        all_corners = img_points.reshape(-1, 1, 2)
        cv2.cornerSubPix(
            gray,
            all_corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        img_points = all_corners.reshape(-1, 2)

        # Solve PnP
        K, dist = self.get_camera_matrix()
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False, None, num_markers

        # Sanity check on translation
        tnorm = np.linalg.norm(tvec)
        if tnorm < 0.05 or tnorm > 5:
            return False, None, num_markers

        R, _ = cv2.Rodrigues(rvec)

        # Camera→world: P_world = R^T * P_cam - R^T * tvec
        T_cw = np.eye(4)
        T_cw[:3, :3] = R.T
        T_cw[:3, 3] = -R.T @ tvec.flatten()

        # The board's Z axis from solvePnP points toward camera.
        # We want world Z to point UP from the board surface (toward camera).
        # If solvePnP returns Z pointing INTO the table, flip it.
        # Check if camera is below board (negative Z in world):
        # If so, flip the Z axis.
        cam_z_world = T_cw[2, 3]
        if cam_z_world < 0:
            print(
                f"  Z appears inverted (camera Z in world = {cam_z_world:.3f}). Flipping Z axis."
            )
            # Apply Z-flip: [x, y, z] → [x, y, -z]
            F_z = np.eye(4)
            F_z[2, 2] = -1
            T = F_z @ T_cw
        else:
            T = T_cw

        print(f"Camera→World translation: {T[:3, 3]}")

        return True, T, num_markers

    def save_intrinsics(self, path):
        intrinsics = self.get_intrinsics()
        with open(path, "w") as f:
            json.dump(intrinsics, f, indent=4)
