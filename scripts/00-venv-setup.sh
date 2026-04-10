#!/bin/bash
source .venv_train/bin/activate
pip3 install torch torchvision torchaudio onnx onnxslim colorama ml_dtypes onnxruntime-gpu ultralytics 

source .venv_hailo/bin/activate
pip install ./hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl 
pip install ./hailo_model_zoo-2.18.0-py3-none-any.whl