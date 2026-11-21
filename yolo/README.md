# YOLO pose: training and Hailo compilation

Train custom **YOLO Pose** models with Ultralytics and convert them to **HEF** (Hailo Executable Format) for Hailo-8/Hailo-8L. Run every command from the **repository root**.

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

Or both environments in one go, pointing at wherever the wheels are:

```bash
DFC_WHL=path/to/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl \
MZ_WHL=path/to/hailo_model_zoo-2.18.0-py3-none-any.whl bash yolo/scripts/00-venv-setup.sh
```

> **Versions.** The thesis models were compiled with DFC 3.33.1 + Model Zoo 2.18 and run on a Pi with HailoRT 4.23. A HEF from a newer compiler (e.g. DFC 3.34) may need a newer HailoRT on the device: check that it loads (`hailortcli parse-hef`) before relying on it.

---

## 🚀 Quick Start Guide

### Step 1: Basic Environment Setup
Run the setup script to initialize your environment. 
> **Note:** If your dataset directory contains a `downloads` subdirectory with the COCO `.zip` files inside (`train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip`), the script skips the download, so copies from another machine can be dropped in there.

```bash
# Usage: bash yolo/scripts/01-basic-setup.sh path/to/dataset
bash yolo/scripts/01-basic-setup.sh ./data
```

### Step 2: Dataset Preparation & Verification
Prepare the dataset structure for YOLO training and verify that the labels are correctly formatted.

```bash
# Setup the dataset
python yolo/scripts/02-setup-dataset.py ./data/

# Verify labels and data integrity
python yolo/scripts/02.5-verify_label.py ./data/
```

### Step 3: Training the Model
The dataset is described by `yolo/config/upper-body-pose.yaml` (13 upper-body keypoints, COCO 0-12). The defaults are the thesis settings: 100 epochs, 640 px, seed 0, deterministic, Ultralytics' default augmentation. The thesis used batch 16 for n and m, 32 for s. The best weights are exported to ONNX (opset 19) at the end.

```bash
python yolo/scripts/03-train.py --model yolov8n-pose
python yolo/scripts/03-train.py --model yolov8s-pose --batch 32
python yolo/scripts/03-train.py --model yolov8n-pose --epochs 1   # smoke test
```

### Step 4: Convert to Hailo (HEF)
Once training is complete, convert your best Pytorch weights into a Hailo-compatible format. This script handles the ONNX export and the Hailo optimization flow.

```bash
# Usage: python yolo/scripts/04-pytorch-to-hailo.py <onnx_path> <calib_data_dir> <hailo_alls_config> [--seed 0]
python yolo/scripts/04-pytorch-to-hailo.py weights/best.onnx data/images/train2017 yolo/config/yolov8s_pose.alls
```

Calibration draws 1,024 random images with `--seed` (default 0). The same seed always draws the same images; the list is saved next to the cache. The compiler uses 64 of them for the quantisation statistics and all 1,024 for fine-tuning. The thesis models were compiled before the seed existed.

**Check every HEF.** The optimisation step varies from run to run, so the same seed doesn't give a byte-identical HEF. Calibration seeds and repeated runs move COCO keypoint AP only a little:

| Model | Val AP over five seeds (mean ± SD, min-max) |
|---|---|
| n | 0.430 ± 0.007 (0.418-0.437) |
| s | 0.508 ± 0.011 (0.490-0.517) |
| m | 0.570 ± 0.006 (0.562-0.576) |

Once in eight s runs, though, optimisation failed outright. That was the s HEF used in the thesis: AP 0.260. Re-optimising with the very same calibration images gave 0.487. The failed run's output SNR was low on every output (4-11 dB), but SNR doesn't rank healthy builds. So score each new HEF on a small set of images that are *not* in the test set (1,000 COCO train2017 images not used for calibration take under a minute on the Pi), compare a few seeds, and keep the best. Ranking on the test set itself would flatter the result.

`--name` and `--out-dir` name the outputs, so builds don't overwrite each other. Every build writes `<name>.build.json` next to its HEF, with:
- the ONNX and model-script checksums, and the script text
- the seed and calibration-image list
- the DFC and Model Zoo versions
- the HEF checksum and the time each stage took

Several seeds in one go, skipping builds already done:

```bash
yolo/scripts/compile_seeds.sh build data/images/train2017 "0 1 2 3 4" \
  upper_body_n=runs/<n run>/weights/best.onnx:yolo/config/yolov8s_pose.alls \
  upper_body_m=runs/<m run>/weights/best.onnx:yolo/config/yolov8m_pose.alls
# -> build/upper_body_n_seed0.hef ... with .build.json and .log for each
```

**Model scripts.** `yolov8s_pose.alls` is used for both n and s, `yolov8m_pose.alls` for m. They refer to layers by position (`conv71`, `output_layer3`, ...): the sigmoid goes on the three class-score outputs, and 16-bit precision on the three keypoint outputs. Positions can differ between models, so after compiling a new model check with `python yolo/scripts/hef_reader.py <model>_optimized.har --alls <the .alls>` that the 16-bit outputs are the keypoint ones (`cv4`) and the sigmoids sit on the class scores (`cv3`). If they aren't, the keypoints end up in 8 bits and lose accuracy.

---

## 🛠️ Structure
* `yolo/config/`: the dataset `.yaml` and the `.alls` Hailo model scripts.
* `yolo/scripts/`: setup, dataset preparation, training, compilation, HAR inspection.
* `runs/`: training runs (Ultralytics output), at the repository root.
* `data/`: datasets, at the repository root.

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

### Hailo Refuses to Use GPU
If Hailo is falling back to the CPU and refusing to utilize your GPU, it is usually caused by missing or incorrect versions of the GPU-enabled runtime and TensorFlow libraries. 

**The Fix:**
Force install the specific GPU-compatible versions of `onnxruntime` and `tensorflow` via pip:

```bash
pip install onnxruntime-gpu==1.18.0
pip install "tensorflow[and-cuda]==2.18.0"
```

> **Note:** You might see dependency incompatibility warnings during installation regarding `onnxruntime`. This is expected and normal—you can safely ignore these warnings as your setup will still work properly.
---

---
