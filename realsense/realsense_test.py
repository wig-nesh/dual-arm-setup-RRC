import os
import sys
import cv2
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realsense.src.realsense_camera import RealSenseCamera


def main():
    # Setup directories
    session_name = input("Enter capture session name: ").strip()
    base_dir = os.path.join("data", session_name)
    rgb_dir = os.path.join(base_dir, "rgb")
    depth_dir = os.path.join(base_dir, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    cam = RealSenseCamera()
    cam.start()

    # Save intrinsics once
    cam.save_intrinsics(os.path.join(base_dir, "intrinsics.json"))

    print("\nControls:")
    print("SPACE → capture frame")
    print("Q     → quit")

    frame_id = 0

    try:
        while True:
            rgb, depth = cam.get_frames()
            if rgb is None:
                continue

            # Depth visualization for display
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            # Combine RGB and Depth side-by-side
            combined = np.hstack((rgb, depth_vis))

            cv2.imshow("RealSense Test (SPACE to capture, Q to quit)", combined)

            # Key handling - works when the OpenCV window is focused
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                rgb_path = os.path.join(rgb_dir, f"{frame_id:06d}.png")
                depth_path = os.path.join(depth_dir, f"{frame_id:06d}.npy")

                cv2.imwrite(rgb_path, rgb)
                np.save(depth_path, depth)

                print(f"Captured frame {frame_id}")
                frame_id += 1

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
