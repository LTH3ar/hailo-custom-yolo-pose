# Hailo Custom YOLO Pose

A toolkit for training custom **YOLO Pose** models using Ultralytics and converting them into **HEF** (Hailo Executable Format) for deployment on Hailo-8/Hailo-8L platforms.

## 🐍 Python Environment Setup
Due to version conflicts between the Hailo Dataflow Compiler and the latest PyTorch/Ultralytics versions, you **must** use two separate virtual environments. Both use Python **3.10.***.

### 1. Training Environment (PyTorch + Ultralytics)
Used for dataset preparation, verification, and model training. This setup assumes **CUDA 13.1** (adjust if your NVIDIA driver differs).

```bash
# Create and activate training env
python3.10 -m venv .venv_train
source .venv_train/bin/activate

# Install dependencies
pip3 install torch torchvision torchaudio onnx onnxslim colorama ml_dtypes onnxruntime-gpu ultralytics
```

### 2. Hailo Environment (Compiler + Model Zoo)
Used for converting the trained ONNX model to Hailo's HEF format.
> **Note:** You must register at the [Hailo Developer Zone](https://hailo.ai/developer-zone/) to download the `.whl` files.

```bash
# Create and activate hailo env
python3.10 -m venv .venv_hailo
source .venv_hailo/bin/activate

# Install downloaded Hailo packages
pip install ./hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl 
pip install ./hailo_model_zoo-2.18.0-py3-none-any.whl
```

---

## 🚀 Quick Start Guide

### Step 1: Basic Environment Setup
Run the setup script to initialize your environment. 
> **Note:** If your dataset directory contains a `downloads` subdirectory with a `.zip` file inside, the script will automatically skip the download phase to save time.

```bash
# Usage: bash path/to/01-basic-setup.sh path/to/dataset
bash ./scripts/01-basic-setup.sh ./data
```

### Step 2: Dataset Preparation & Verification
Prepare the dataset structure for YOLO training and verify that the labels are correctly formatted.

```bash
# Setup the dataset
python ./scripts/02-setup-dataset.py ./data/

# Verify labels and data integrity
python ./scripts/02.5-verify_label.py ./data/
```

### Step 3: Training the Model
Before running this step, ensure you have a `config/` directory containing your `.yaml` configuration file. 
> **Important:** You must manually edit `03-train.py` to point to your specific model version and parameters before running.

```bash
python ./scripts/03-train.py
```

### Step 4: Convert to Hailo (HEF)
Once training is complete, convert your best Pytorch weights into a Hailo-compatible format. This script handles the ONNX export and the Hailo optimization flow.

```bash
# Usage: python 04-pytorch-to-hailo.py <onnx_path> <calib_data_dir> <hailo_alls_config>
python ./scripts/04-pytorch-to-hailo.py weights/best.onnx data/images/train2017 config/yolov8s_pose.alls
```

---

## 🛠️ Project Structure
* `config/`: Contains `.yaml` for training and `.alls` for Hailo optimization scripts.
* `weights/`: Default output location for trained models.
* `data/`: Recommended directory for your datasets.

---

## ⚠️ Troubleshooting & Known Bugs

### Hailo/XLA: libdevice.10.bc not found
If you encounter a JIT compilation error during conversion that looks like this:
`Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice.error: libdevice not found at ./libdevice.10.bcUNKNOWN: JIT compilation failed.`

**The Fix:**
You need to manually link your system's CUDA `libdevice` into your virtual environment and set the XLA flag.

1. **Find the file on your system:**
   ```bash
   find /usr /usr/local -type f -name "libdevice.10.bc" 2>/dev/null
   ```
2. **Create the expected directory and symlink it:**
   ```bash
   mkdir -p ./.venv_hailo/nvvm/libdevice
   ln -s /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ./.venv_hailo/nvvm/libdevice/libdevice.10.bc
   ```
3. **Export the XLA flag:**
   ```bash
   export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(pwd)/.venv_hailo
   ```

---

---

## 📜 License & Attribution
This project is licensed under the **GNU AGPLv3**.

### Citation
If you use this code or guide in your project, please cite the author:
> **Author:** LTH3ar  
> **Project:** hailo-custom-yolo-pose (2026)  
> **Link:** [https://github.com/LTH3ar/hailo-custom-yolo-pose](https://github.com/LTH3ar/hailo-custom-yolo-pose)

### Dependencies
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
* [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/)
