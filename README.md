# Hailo Custom YOLO Pose

A toolkit for training custom **YOLO Pose** models using Ultralytics and converting them into **HEF** (Hailo Executable Format) for deployment on Hailo-8/Hailo-8L platforms.

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
