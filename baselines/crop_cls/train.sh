#!/usr/bin/env bash
# Train the crop baseline (YOLOv8s-cls) on every dataset folder make_crops.py wrote (ubuntu3060, GPU).
# usage: baselines/crop_cls/train.sh ~/hcpose/crop_cls [runs...]   (default: fold0..fold4 all scb_only cla_only)
# Each run ends with weights/last.pt, which predict.py and clips.py use. Settings are fixed here so every
# run is the same; change them only for all runs together.
set -euo pipefail
ROOT=$(realpath "$1"); shift
RUNS=${*:-"fold0 fold1 fold2 fold3 fold4 all scb_only cla_only"}
for r in $RUNS; do
  [ -d "$ROOT/ds/$r/train" ] || { echo "skip $r: no $ROOT/ds/$r/train"; continue; }
  if [ -f "$ROOT/runs/$r/weights/last.pt" ] && [ -f "$ROOT/runs/$r/results.csv" ] && \
     [ "$(tail -n +2 "$ROOT/runs/$r/results.csv" | wc -l)" -ge "${EPOCHS:-50}" ]; then
    echo "done already: $r"; continue
  fi
  start=$(date +%s)
  yolo classify train model=yolov8s-cls.pt data="$ROOT/ds/$r" imgsz=224 epochs="${EPOCHS:-50}" batch=128 \
       seed=0 deterministic=True workers=8 project="$ROOT/runs" name="$r" exist_ok=True plots=False
  echo "$r: $(( ($(date +%s) - start) / 60 )) min" | tee -a "$ROOT/runs/times.txt"
done
