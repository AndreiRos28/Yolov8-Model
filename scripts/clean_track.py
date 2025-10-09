from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Config
YOLO_WEIGHTS = "C:/Downloads/Human.v1i.yolov8/weights_backup/best.pt"
CONF_THRESHOLD = 0.65  # Lowered to detect harder poses
BOX_COLOR = (0, 255, 0)

# Init YOLO + DeepSORT
model = YOLO(YOLO_WEIGHTS)
model.fuse()

tracker = DeepSort(
    max_age=15,
    n_init=2,
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    nn_budget=100
)

# Advanced Responsive Box Stabilizer
class ResponsiveBoxFilter:
    def __init__(self, ema_alpha=0.3, deadzone=8, expansion_sensitivity=1.3):
        """
        ema_alpha: Base smoothing (0.1-0.5)
        deadzone: Ignore small movements
        expansion_sensitivity: How quickly box expands (1.0-2.0, higher = more responsive)
        """
        self.base_alpha = ema_alpha
        self.deadzone = deadzone
        self.expansion_sensitivity = expansion_sensitivity
        
        self.smoothed = {}
        self.velocity = {}
        self.history = {}
        self.size_history = {}
        
    def filter(self, track_id, box):
        box = np.array(box, dtype=float)
        
        # Initialize if new track
        if track_id not in self.smoothed:
            self.smoothed[track_id] = box
            self.velocity[track_id] = np.zeros(4)
            self.history[track_id] = [box]
            self.size_history[track_id] = [self._get_box_size(box)]
            return box.astype(int)
        
        prev_box = self.smoothed[track_id]
        
        # Calculate size change (width + height)
        prev_size = self._get_box_size(prev_box)
        curr_size = self._get_box_size(box)
        size_change_ratio = curr_size / (prev_size + 1e-6)
        
        # Detect expansion (arm stretch, movement)
        is_expanding = size_change_ratio > 1.05  # Growing by 5%+
        
        # Dynamic alpha: more responsive when expanding
        if is_expanding:
            # Use higher alpha for faster expansion response
            dynamic_alpha = min(self.base_alpha * self.expansion_sensitivity, 0.7)
        else:
            # Use lower alpha for stability when shrinking/stable
            dynamic_alpha = self.base_alpha * 0.6
        
        # Calculate change
        raw_diff = box - prev_box
        
        # Smart deadzone: reduce deadzone when expanding
        current_deadzone = self.deadzone if not is_expanding else self.deadzone * 0.3
        
        for i in range(4):
            if abs(raw_diff[i]) < current_deadzone:
                raw_diff[i] = 0
        
        # Outlier detection with expansion awareness
        history = self.history[track_id]
        if len(history) >= 3:
            recent_boxes = np.array(history[-3:])
            avg_box = np.mean(recent_boxes, axis=0)
            std_box = np.std(recent_boxes, axis=0)
            
            # More lenient with outliers when expanding
            threshold = 4 if is_expanding else 3
            
            for i in range(4):
                if std_box[i] > 0:
                    deviation = abs(box[i] - avg_box[i]) / (std_box[i] + 1)
                    if deviation > threshold:
                        box[i] = avg_box[i] + np.sign(box[i] - avg_box[i]) * threshold * std_box[i]
        
        # Velocity calculation with expansion boost
        velocity_damping = 0.6 if is_expanding else 0.4
        new_velocity = raw_diff * velocity_damping + self.velocity[track_id] * (1 - velocity_damping)
        self.velocity[track_id] = new_velocity
        
        # Apply dynamic EMA smoothing
        prediction = prev_box + new_velocity
        self.smoothed[track_id] = (
            dynamic_alpha * box +
            (1 - dynamic_alpha) * prediction
        )
        
        # Update histories
        self.history[track_id].append(self.smoothed[track_id].copy())
        if len(self.history[track_id]) > 5:
            self.history[track_id].pop(0)
        
        self.size_history[track_id].append(curr_size)
        if len(self.size_history[track_id]) > 5:
            self.size_history[track_id].pop(0)
        
        return self.smoothed[track_id].astype(int)
    
    def _get_box_size(self, box):
        """Calculate box size (width + height)"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width + height
    
    def cleanup(self, active_ids):
        self.smoothed = {k: v for k, v in self.smoothed.items() if k in active_ids}
        self.velocity = {k: v for k, v in self.velocity.items() if k in active_ids}
        self.history = {k: v for k, v in self.history.items() if k in active_ids}
        self.size_history = {k: v for k, v in self.size_history.items() if k in active_ids}

# Create responsive stabilizer
stabilizer = ResponsiveBoxFilter(
    ema_alpha=0.25,           # Base smoothing
    deadzone=10,              # Ignore small movements
    expansion_sensitivity=1.5  # Boost response when expanding (1.0-2.0)
)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(
        frame, 
        conf=CONF_THRESHOLD,
        iou=0.5,
        imgsz=640,
        verbose=False
    )

    # Collect person detections
    detections = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # DEBUG: Show all detections with confidence
        for box, conf, cls in zip(boxes, confs, classes):
            if int(cls) == 0:  # only "person"
                l, t, r, b = map(int, box)
                detections.append(([l, t, r-l, b-t], float(conf), 'person'))
                print(f"Detection confidence: {conf:.2f}")  # DEBUG LINE

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    
    active_ids = []
    
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        active_ids.append(track_id)
        
        l, t, r, b = map(int, track.to_ltrb())
        
        # Apply responsive stabilization
        stable_box = stabilizer.filter(track_id, (l, t, r, b))
        l, t, r, b = stable_box

        # Draw responsive box
        cv2.rectangle(frame, (l, t), (r, b), BOX_COLOR, 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

    stabilizer.cleanup(active_ids)

    cv2.imshow("YOLOv8 + DeepSORT (Responsive)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()