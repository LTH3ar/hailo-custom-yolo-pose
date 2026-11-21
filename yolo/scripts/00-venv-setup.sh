#!/bin/bash
# Create the two Python 3.10 environments (run from the repository root).
#   .venv_train  PyTorch + Ultralytics (training, ONNX export)
#   .venv_hailo  Hailo Dataflow Compiler + Model Zoo (HEF compilation)
# The Hailo wheels must be downloaded by hand from the Hailo Developer Zone. Point at them with
#   DFC_WHL=/path/to/hailo_dataflow_compiler-*.whl MZ_WHL=/path/to/hailo_model_zoo-*.whl bash yolo/scripts/00-venv-setup.sh
# The thesis models were compiled with DFC 3.33.1 + Model Zoo 2.18 (HailoRT 4.23 on the Pi).
# A newer DFC may produce HEFs that need a newer HailoRT on the device.
set -euo pipefail
PY="${PYTHON:-python3.10}"
DFC_WHL="${DFC_WHL:-./hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl}"
MZ_WHL="${MZ_WHL:-./hailo_model_zoo-2.18.0-py3-none-any.whl}"

"$PY" -m venv .venv_train
.venv_train/bin/pip install torch torchvision torchaudio onnx onnxslim colorama ml_dtypes onnxruntime-gpu ultralytics

"$PY" -m venv .venv_hailo
.venv_hailo/bin/pip install "$DFC_WHL" "$MZ_WHL"
