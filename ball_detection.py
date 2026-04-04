import cv2
import numpy as np
from ultralytics import YOLO


# ----- Paths and constants -----
input_video_path = "test_video1.mp4"
output_video_path = "output_ball.mp4"

# COCO "sports ball" in pretrained YOLO models.
SPORTS_BALL_CLASS_ID = 32

# Ignore weak YOLO guesses (0.0–1.0). Raise if you miss detections; lower if noisy.
YOLO_MIN_CONF = 0.25

# HSV range for orange (OpenCV H is 0–179). Tune if lighting differs.
HSV_ORANGE_LOWER = np.array([8, 80, 80], dtype=np.uint8)
HSV_ORANGE_UPPER = np.array([28, 255, 255], dtype=np.uint8)

# Ignore tiny blobs in the HSV mask (noise).
MIN_ORANGE_AREA = 400

# Load YOLOv8 once (downloads weights on first run).
model = YOLO("yolov8x.pt")

def xyxy_to_xywh(x1, y1, x2, y2):
    """Convert YOLO corner box to OpenCV tracker format (x, y, width, height)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def draw_circle_from_box(frame_img, bx, by, bw, bh):
    """Draw a green circle inside the box (no text)."""
    cx = int(bx + bw / 2)
    cy = int(by + bh / 2)
    r = int(min(bw, bh) / 2)
    if r < 1:
        return
    cv2.circle(frame_img, (cx, cy), r, (0, 255, 0), 2)


def yolo_best_sports_ball(frame):
    """Return (x, y, w, h) for best sports-ball detection, or None."""
    results = model(frame, verbose=False)
    result = results[0]
    best_xywh = None
    best_conf = -1.0
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id != SPORTS_BALL_CLASS_ID:
            continue
        conf = float(box.conf[0].item())
        if conf < YOLO_MIN_CONF:
            continue
        if conf > best_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, w, h = xyxy_to_xywh(x1, y1, x2, y2)
            if w >= 2 and h >= 2:
                best_xywh = (x1, y1, w, h)
                best_conf = conf
    return best_xywh


def hsv_orange_bbox(frame):
    """Largest orange blob → bounding box (x, y, w, h), or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_ORANGE_LOWER, HSV_ORANGE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_ORANGE_AREA:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def find_ball_init_scan(cap, use_yolo):
    """
    Read cap forward frame by frame until a ball box is found or video ends.
    use_yolo True → YOLO; False → HSV orange fallback.
    Returns (frame_index, (x,y,w,h)) or (None, None).
    """
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return None, None
        if use_yolo:
            box = yolo_best_sports_ball(frame)
        else:
            box = hsv_orange_bbox(frame)
        if box is not None:
            return index, box
        index += 1


def create_csrt():
    return cv2.legacy.TrackerCSRT_create()


# ----- Pass 1: find first frame where we can init the tracker -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open input video: {input_video_path}")
    raise SystemExit

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=True)
cap.release()

if init_frame_index is None:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not reopen video for HSV fallback.")
        raise SystemExit
    init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=False)
    cap.release()

# ----- Pass 2: write full output video -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video for output pass.")
    raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

tracker = None
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if init_bbox is None:
        # Neither YOLO nor HSV ever found a ball: copy frames unchanged.
        out.write(frame)
        frame_number += 1
        continue

    if frame_number < init_frame_index:
        # Before we have a box: no circle yet.
        out.write(frame)
    elif frame_number == init_frame_index:
        tracker = create_csrt()
        x, y, w, h = init_bbox
        if not tracker.init(frame, (x, y, w, h)):
            print("Warning: CSRT init failed; output without tracking.")
            tracker = None
            out.write(frame)
        else:
            draw_circle_from_box(frame, x, y, w, h)
            out.write(frame)
    else:
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                bx, by, bw, bh = bbox
                draw_circle_from_box(frame, bx, by, bw, bh)
        out.write(frame)

    frame_number += 1

cap.release()
out.release()

if init_bbox is None:
    print("Warning: No ball found (YOLO or HSV). Saved video with no circles.")
else:
    print(f"Done! Ball tracking from frame {init_frame_index}. Saved: {output_video_path}")
