#!/usr/bin/env bash
# Compile one or more models with several calibration seeds, one build after another.
#
# usage (from the repository root, inside the Hailo venv):
#   yolo/scripts/compile_seeds.sh OUT_DIR CALIB_DIR "SEEDS" NAME=ONNX:ALLS [NAME=ONNX:ALLS ...]
# e.g.
#   yolo/scripts/compile_seeds.sh build data/images/train2017 "1 2 3 4" \
#     upper_body_n_dfc3331=onnx/n.onnx:yolo/config/yolov8s_pose.alls \
#     upper_body_s_dfc3331=onnx/s.onnx:yolo/config/yolov8s_pose.alls \
#     upper_body_m_dfc3331=onnx/m.onnx:yolo/config/yolov8m_pose.alls
#
# Each build writes OUT_DIR/NAME_seedK.{har,_optimized.har,hef,build.json} and a log NAME_seedK.log.
# A build whose HEF already exists is skipped, so an interrupted sweep can be restarted.
# env: PY (default: python), IMAGES (default: 1024)
set -uo pipefail
[ $# -ge 4 ] || { sed -n 2,15p "$0"; exit 2; }
OUT=$1; CALIB=$2; SEEDS=$3; shift 3
PY=${PY:-python}; IMAGES=${IMAGES:-1024}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for spec in "$@"; do
  NAME=${spec%%=*}; rest=${spec#*=}; ONNX=${rest%%:*}; ALLS=${rest#*:}
  for k in $SEEDS; do
    N="${NAME}_seed${k}"
    if [ -s "$OUT/$N.hef" ]; then echo "== $N: exists, skipped"; continue; fi
    echo "== $N ($(date +%H:%M))"
    if "$PY" "$HERE/04-pytorch-to-hailo.py" "$ONNX" "$CALIB" "$ALLS" --seed "$k" --images "$IMAGES" \
        --name "$N" --out-dir "$OUT" > "$OUT/$N.log" 2>&1; then
      tail -n 1 "$OUT/$N.log"
    else
      echo "   FAILED, see $OUT/$N.log"
    fi
  done
done
