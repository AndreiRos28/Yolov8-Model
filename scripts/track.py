from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
import torch
import torch.nn as nn
import json
import os
from collections import deque
import time

YOLO_WEIGHTS = "C:/Downloads/Human.v1i.yolov8/runs/detect/train/weights/best.pt"
CONF_THRESHOLD = 0.75
BOX_COLOR = (0, 255, 0)
UID_COLOR = (0, 0, 255)
SIM_THRESHOLD = 0.95
EMBEDDING_HISTORY = 5
PERSIST_FILE = "uid_embeddings.json"
MAX_AGE = 30

# Your real height in cm for reference
REFERENCE_HEIGHT_CM = 172  
LOITER_THRESHOLD_SEC = 30
STEP_THRESHOLD = 20
FAST_MOVE_THRESHOLD = 50
CROWD_THRESHOLD = 3  # number of people considered as a crowd

model = YOLO(YOLO_WEIGHTS)
tracker = DeepSort(max_age=MAX_AGE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_model = nn.Sequential(
    nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(32, 128)
).to(device)
feature_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

geofence = None
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, geofence
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        geofence = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        geofence = (ix, iy, x, y)

cv2.namedWindow("YOLOv8 + DeepSORT")
cv2.setMouseCallback("YOLOv8 + DeepSORT", draw_rectangle)

sticky_uids = {}
all_uids = set()
last_positions = {}
uid_geofence_status = {}
embedding_store = {}
embedding_history = {}
uid_entry_time = {}
step_count = {}
next_uid = 1
reference_pixels = None

if os.path.exists(PERSIST_FILE):
    try:
        with open(PERSIST_FILE, "r") as f:
            data = json.load(f)
            embedding_store = {uid: np.array(emb) for uid, emb in data.get("embeddings", {}).items()}
            next_uid = data.get("next_uid", 1)
    except json.JSONDecodeError:
        pass

def inside_geofence(cx, cy, gf):
    if gf is None:
        return True
    x1, y1, x2, y2 = gf
    return min(x1,x2) <= cx <= max(x1,x2) and min(y1,y2) <= cy <= max(y1,y2)

def get_embedding(cropped_img):
    img_tensor = preprocess(cropped_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = feature_model(img_tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)

def average_embedding(deque_emb):
    arr = np.array(deque_emb)
    avg = np.mean(arr, axis=0)
    return avg / np.linalg.norm(avg)

def assign_persistent_uid(avg_emb):
    global next_uid
    best_uid = None
    best_sim = 0
    for uid, stored_emb in embedding_store.items():
        sim = np.dot(avg_emb, stored_emb)
        if sim > best_sim:
            best_sim = sim
            best_uid = uid
    if best_sim >= SIM_THRESHOLD:
        embedding_store[best_uid] = avg_emb
        return best_uid
    new_uid = str(next_uid)
    next_uid += 1
    embedding_store[new_uid] = avg_emb
    return new_uid

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if geofence:
        x1, y1, x2, y2 = geofence
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    results = model(frame, conf=CONF_THRESHOLD, iou=0.6)
    detections = []
    for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                         results[0].boxes.conf.cpu().numpy()):
        l, t, r, b = map(int, box)
        detections.append([[l, t, r, b], float(conf)])

    tracks = tracker.update_tracks(detections, frame=frame)

    in_geofence_uids = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        ds_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l+r)//2, (t+b)//2
        cropped = frame[t:b, l:r]
        if cropped.size == 0:
            continue
        emb = get_embedding(cropped)
        if ds_id not in embedding_history:
            embedding_history[ds_id] = deque(maxlen=EMBEDDING_HISTORY)
        embedding_history[ds_id].append(emb)
        avg_emb = average_embedding(embedding_history[ds_id])
        assigned_uid = assign_persistent_uid(avg_emb)
        sticky_uids[ds_id] = assigned_uid
        all_uids.add(assigned_uid)
        last_pos = last_positions.get(assigned_uid, (cx, cy))
        last_positions[assigned_uid] = (cx, cy)
        uid_geofence_status[assigned_uid] = inside_geofence(cx, cy, geofence)

        if uid_geofence_status[assigned_uid]:
            in_geofence_uids.append(assigned_uid)

        person_pixel_height = b - t
        if reference_pixels is None:
            reference_pixels = person_pixel_height
        estimated_height = person_pixel_height / reference_pixels * REFERENCE_HEIGHT_CM
        estimated_height = max(140, min(210, estimated_height))

        # Step count
        if assigned_uid not in step_count:
            step_count[assigned_uid] = 0
        dx = cx - last_pos[0]
        dy = cy - last_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance > STEP_THRESHOLD:
            step_count[assigned_uid] += 1

        # Loitering detection
        if uid_geofence_status[assigned_uid]:
            if assigned_uid not in uid_entry_time:
                uid_entry_time[assigned_uid] = time.time()
        else:
            uid_entry_time.pop(assigned_uid, None)

        loitering = False
        if assigned_uid in uid_entry_time:
            duration = time.time() - uid_entry_time[assigned_uid]
            if duration >= LOITER_THRESHOLD_SEC:
                loitering = True

        # Abnormal movement detection
        abnormal_movement = distance > FAST_MOVE_THRESHOLD and uid_geofence_status[assigned_uid]

        # Draw box and label
        label = f"UID:{assigned_uid} H:{int(estimated_height)}cm S:{step_count[assigned_uid]}"
        if loitering:
            label += " [LOITERING]"
        if abnormal_movement:
            label += " [FAST MOVE]"
        if uid_geofence_status[assigned_uid]:
            cv2.rectangle(frame, (l, t), (r, b), BOX_COLOR, 2)
            cv2.putText(frame, label, (l, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, UID_COLOR, 2)

    # Crowd detection
    if len(in_geofence_uids) >= CROWD_THRESHOLD:
        cv2.putText(frame, f"CROWD ALERT! Count: {len(in_geofence_uids)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    in_frame_count = sum(uid_geofence_status.values())
    cv2.putText(frame, f"In-Frame Count (GF): {in_frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

json_data = {
    "next_uid": next_uid,
    "embeddings": {uid: emb.tolist() for uid, emb in embedding_store.items()}
}
with open(PERSIST_FILE, "w") as f:
    json.dump(json_data, f, indent=2)

for uid in sorted(all_uids, key=int):
    print(f" - {uid}")

