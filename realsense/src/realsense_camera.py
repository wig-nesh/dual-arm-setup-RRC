import pyrealsense2 as rs
import numpy as np
import json
import os


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width = width
        self.height = height
        self.fps = fps

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = None
        self.align = rs.align(rs.stream.color)

    def start(self):
        self.profile = self.pipeline.start(self.config)

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
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

    def save_intrinsics(self, path):
        intrinsics = self.get_intrinsics()
        with open(path, "w") as f:
            json.dump(intrinsics, f, indent=4)
