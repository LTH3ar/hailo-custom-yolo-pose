from ultralytics import YOLO
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(PROJECT_ROOT)

TIMESTAMP = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
print(TIMESTAMP)

MODEL_NAME = "yolov8n-pose" # select pretrain model, name only: yolov8n/s/m-pose, yolov11n/s/m-pose, yolo26n/s/m-pose

model = YOLO(f"{MODEL_NAME}.pt")  # pretrained pose model

RUN_INSTANCE = f"upper-body-pose_{MODEL_NAME}_{TIMESTAMP}"

results = model.train(
    data="config/upper-body-pose.yaml", # path to yaml model config file
    # epochs=100,            # default
    epochs=1,            # test only
    imgsz=640,             # default
    batch=16,              # default (use -1 for auto GPU memory)
    project=str(PROJECT_ROOT / "runs"),
    name=RUN_INSTANCE,

    # ── Optimizer & LR (all defaults) ───────────
    optimizer="auto",      # auto-selects SGD for YOLO
    lr0=0.01,              # initial learning rate
    lrf=0.01,              # final LR = lr0 * lrf (i.e. 0.0001)
    momentum=0.937,        # SGD momentum
    weight_decay=0.0005,   # L2 regularization
    warmup_epochs=3.0,     # warmup from low LR
    warmup_momentum=0.8,   # warmup starting momentum
    warmup_bias_lr=0.1,    # warmup bias LR

    # ── Loss weights (defaults) ─────────────────
    box=7.5,               # bounding box loss weight
    cls=0.5,               # classification loss weight
    dfl=1.5,               # distribution focal loss weight
    pose=12.0,             # keypoint loss weight
    kobj=1.0,              # keypoint objectness loss weight

    # ── Training behavior (defaults) ────────────
    patience=100,          # early stopping patience
    cos_lr=False,          # linear LR decay (not cosine)
    close_mosaic=10,       # disable mosaic last 10 epochs
    amp=True,              # mixed precision
    cache=False,           # no image caching
    workers=8,             # dataloader workers
    seed=0,                # reproducibility
    deterministic=True,    # deterministic mode

    # ── Augmentation (defaults) ─────────────────
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,

    # ── Saving & logging (defaults) ─────────────
    save=True,
    save_period=-1,        # only save best & last
    plots=True,
    val=True,
)

# ── Video Inference ─────────────────────────────
# video_path = PROJECT_ROOT / "example.mp4"
best_weights = PROJECT_ROOT / "runs" / RUN_INSTANCE / "weights" / "best.pt"

# trained_model = YOLO(str(best_weights))

# trained_model.predict(
#     source=str(video_path),
#     save=True,
#     project=str(PROJECT_ROOT / "runs" / RUN_INSTANCE),
#     name=f"result_{video_path.stem}",
#     conf=0.25,             # confidence threshold
#     imgsz=640,
# )

# Convert to onnx
# 1. Load the PyTorch model
model = YOLO(best_weights)

# 2. Export the model to ONNX format
success = model.export(format="onnx", opset=19)

print("Export successful:", success)