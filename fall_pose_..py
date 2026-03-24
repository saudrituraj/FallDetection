# python3 fall_pose.py Timeline10_00149015.mp4 /home/pguha6/uma/codes/interns/fall/fall1.mp4



import cv2
import numpy as np
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO


def main(input_video, output_video):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # History buffers
    height_hist = defaultdict(lambda: deque(maxlen=20))
    center_hist = defaultdict(lambda: deque(maxlen=20))
    ratio_hist = defaultdict(lambda: deque(maxlen=20))

    # State management
    fall_state = defaultdict(lambda: "STAND")
    recovery_counter = defaultdict(int)

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            iou=0.5,
            verbose=False,
            classes=[0]  # person class
        )

        if results[0].boxes is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id

            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)

                for i, box in enumerate(boxes):

                    if i >= len(track_ids):
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if h <= 0:
                        continue

                    ratio = w / h
                    track_id = track_ids[i]

                    # Store history
                    height_hist[track_id].append(h)
                    center_hist[track_id].append(cy)
                    ratio_hist[track_id].append(ratio)

                    heights = list(height_hist[track_id])
                    centers = list(center_hist[track_id])
                    ratios = list(ratio_hist[track_id])

                    # Need enough history
                    if len(heights) >= 10:

                        max_height = max(heights)
                        current_height = heights[-1]
                        prev_height = heights[-5]

                        current_ratio = ratios[-1]
                        prev_ratio = ratios[-5]

                        # --- FALL CONDITIONS ---
                        height_drop = current_height < 0.6 * max_height
                        vertical_motion = centers[-1] - centers[-5] > 25
                        horizontal_pose = current_ratio > 0.8

                        # --- STATE MACHINE ---
                        if fall_state[track_id] == "STAND":

                            if height_drop and vertical_motion and horizontal_pose:
                                fall_state[track_id] = "FALL"
                                recovery_counter[track_id] = 0

                        elif fall_state[track_id] == "FALL":

                            # Recovery checks
                            upright_pose = current_ratio < 0.6
                            height_recovered = current_height > 0.8 * max_height

                            if upright_pose and height_recovered:
                                recovery_counter[track_id] += 1
                            else:
                                recovery_counter[track_id] = 0

                            # Require stable upright frames
                            if recovery_counter[track_id] > 8:
                                fall_state[track_id] = "STAND"

                    # Draw results
                    is_fall = fall_state[track_id] == "FALL"
                    color = (0, 0, 255) if is_fall else (0, 255, 0)
                    thickness = 4 if is_fall else 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    cv2.putText(
                        frame,
                        f"ID:{track_id} {fall_state[track_id]}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output saved to:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Fall Detection System")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    args = parser.parse_args()

    main(args.input, args.output)
