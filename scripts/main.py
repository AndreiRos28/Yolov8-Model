from ultralytics import YOLO

# Load your last trained weights as pretrained starting point
model = YOLO("C:/Downloads/Human.v1i.yolov8/runs/detect/train/weights/last.pt")

# Start a NEW training run (fresh logs, fresh optimizer)
model.train(
    data="C:/Downloads/Human.v1i.yolov8/data/data.yaml",
    epochs=100,     # train up to 100 epochs (full run)
    resume=True    # <- drop resume, start new run
)
