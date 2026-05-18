import cv2
import numpy as np
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from realsense.src.realsense_camera import RealSenseCamera


def main():
    parser = argparse.ArgumentParser(description="Test FFS Server with RealSense")
    parser.add_argument(
        "--ffs-url",
        type=str,
        default=None,
        help="FFS Server URL. If omitted, uses native RealSense depth.",
    )
    args = parser.parse_args()

    use_ffs = args.ffs_url is not None
    ffs_status = f"FFS at {args.ffs_url}" if use_ffs else "Native Depth"

    print(f"Initializing RealSense (using {ffs_status} at 640x480)...")

    # Initialize camera with FFS
    cam = RealSenseCamera(width=640, height=480, use_ffs=use_ffs, ffs_url=args.ffs_url)

    try:
        cam.start()
        print("\nCamera started successfully!")
        print("Streaming RGB and FFS Depth. Press 'q' or 'ESC' to quit.")

        while True:
            color, depth = cam.get_frames()

            if color is None or depth is None:
                continue

            # Normalize depth for visualization (ignore zeros which are invalid/background)
            depth_vis = depth.astype(np.float32)
            max_depth = depth_vis.max()
            if max_depth > 0:
                # Optional: clip very far distances for better visualization contrast
                clip_max = (
                    np.percentile(depth_vis[depth_vis > 0], 95)
                    if np.any(depth_vis > 0)
                    else max_depth
                )
                depth_vis = np.clip(depth_vis, 0, clip_max)
                depth_vis = (depth_vis / clip_max * 255).astype(np.uint8)
            else:
                depth_vis = depth_vis.astype(np.uint8)

            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Since 640x480 side-by-side is 1280px wide, we don't necessarily need to resize
            # but we keep the structure just in case.
            disp_color = cv2.resize(color, (640, 480))
            disp_depth = cv2.resize(depth_vis_color, (640, 480))

            combined = np.hstack((disp_color, disp_depth))

            # Add labels
            cv2.putText(
                combined,
                "RGB (640x480)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined,
                f"FFS Depth (Max disp: {max_depth:.2f}m)",
                (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("FFS Live Test", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
