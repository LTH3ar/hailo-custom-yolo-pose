# Hailo Custom YOLO Pose

A comprehensive guide and toolkit for training, converting, and deploying custom **YOLO Pose** models specifically for the **Hailo** AI processor.

## 🚀 Overview
This repository provides a streamlined workflow to bridge the gap between Ultralytics YOLO training and Hailo's high-performance inference hardware. It includes:
* **Training Scripts:** Custom training pipelines using `ultralytics`.
* **Conversion Tools:** Simplified steps to export to ONNX and compile for Hailo (HEF).
* **Inference Examples:** Reference code for running your pose models on-device.

## 🛠️ Requirements
* Python 3.10+
* Hailo Dataflow Compiler (DFC)
* Ultralytics YOLOv8/v11

## 📜 License & Attribution
This project is licensed under the **GNU AGPLv3**. 

### Citation
If you use this code or guide in your project, please cite the author:
> **Author:** LTH3ar  
> **Project:** hailo-custom-yolo-pose (2026)  
> **Link:** [https://github.com/LTH3ar/hailo-custom-yolo-pose](https://github.com/LTH3ar/hailo-custom-yolo-pose)

### Dependencies
This work utilizes the following excellent open-source libraries:
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
* [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) (MIT)