import cv2
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.src.sam import SAMClient


def main():
    parser = argparse.ArgumentParser(description="Webcam SAM Client Test")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="SAM Server URL"
    )
    args = parser.parse_args()

    # Initialize SAM Client
    client = SAMClient(args.url)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nControls:")
    print("LEFT CLICK → Get SAM mask at point")
    print("Q          → Quit")

    last_mask = None
    point_to_predict = None

    def on_mouse(event, x, y, flags, param):
        nonlocal point_to_predict
        if event == cv2.EVENT_LBUTTONDOWN:
            point_to_predict = (x, y)

    cv2.namedWindow("Webcam SAM Test")
    cv2.setMouseCallback("Webcam SAM Test", on_mouse)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Handle SAM request if a point was clicked
            if point_to_predict is not None:
                print(f"Requesting SAM mask for point: {point_to_predict}...")
                try:
                    # Send point to server
                    mask = client.predict_from_point(frame, point_to_predict)
                    last_mask = mask
                except Exception as e:
                    print(f"SAM Error: {e}")
                finally:
                    point_to_predict = None

            # Visualization
            display_frame = frame.copy()

            if last_mask is not None:
                # Apply green mask overlay
                mask_bool = last_mask > 0
                display_frame[mask_bool] = display_frame[mask_bool] // 2 + np.array(
                    [0, 128, 0], dtype=np.uint8
                )

                # Optional: draw contour
                contours, _ = cv2.findContours(
                    last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)

            cv2.imshow("Webcam SAM Test", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np  # Needed for the overlay math

    main()
