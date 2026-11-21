"""Train an upper-body (13-keypoint) YOLO pose model with Ultralytics, then export it to ONNX.

Run from the repository root. Defaults are the settings used for the thesis models
(100 epochs, 640 px, seed 0, deterministic, Ultralytics' default augmentation and
loss weights). The thesis used batch 16 for n and m, 32 for s.

usage:
  python yolo/scripts/03-train.py --model yolov8n-pose
  python yolo/scripts/03-train.py --model yolov8s-pose --batch 32
  python yolo/scripts/03-train.py --model yolov8n-pose --epochs 1      # smoke test
"""
import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n-pose",
                    help="pretrained model name without .pt: yolov8{n,s,m}-pose, yolo11{n,s,m}-pose, ... (default yolov8n-pose)")
    ap.add_argument("--data", default=str(REPO / "yolo" / "config" / "upper-body-pose.yaml"),
                    help="dataset YAML (default yolo/config/upper-body-pose.yaml)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto from GPU memory (thesis: 16 for n/m, 32 for s)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", help="run name (default upper-body-pose_<model>_<timestamp>)")
    ap.add_argument("--no-export", action="store_true", help="skip the ONNX export after training")
    ap.add_argument("--opset", type=int, default=19, help="ONNX opset for the export (default 19)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run = args.name or f"upper-body-pose_{args.model}_{stamp}"
    print(f"repo {REPO}\nrun  {run}")

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(REPO / "runs"),
        name=run,
        # optimizer and learning rate: Ultralytics defaults. With optimizer="auto",
        # Ultralytics picks the optimizer and learning rate itself and ignores lr0/momentum.
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # loss weights (defaults)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        # training behaviour (defaults)
        patience=100,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        cache=False,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        # augmentation (defaults; flip_idx in the dataset YAML swaps left/right keypoints)
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
        # saving and logging
        save=True,
        save_period=-1,
        plots=True,
        val=True,
    )

    best = REPO / "runs" / run / "weights" / "best.pt"
    if not args.no_export:
        out = YOLO(str(best)).export(format="onnx", opset=args.opset)
        print("exported", out)


if __name__ == "__main__":
    main()
