import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image
from hailo_sdk_client import ClientRunner

# 1. Setup Paths & Automatic Naming
if len(sys.argv) < 4:
    print("Usage: python convert.py <model.onnx> <calib_dir> <model_script.alls>")
    sys.exit(1)

ONNX_FILE = Path(sys.argv[1])
CALIBRATION_IMAGES_PATH = Path(sys.argv[2])
MODEL_SCRIPT = Path(sys.argv[3])

# Derive output names from ONNX path
# All outputs will live in the same folder as the input ONNX
model_name = ONNX_FILE.stem
parent_dir = ONNX_FILE.parent

STANDARD_HAR = parent_dir / f"{model_name}.har"
OPTIMIZED_HAR = parent_dir / f"{model_name}_optimized.har"
HEF_OUTPUT = parent_dir / f"{model_name}.hef"

print(f"[info] ONNX Source: {ONNX_FILE}")
print(f"[info] Target Dir: {parent_dir}")

# 2. Translation Stage
runner = ClientRunner(hw_arch="hailo8")
hn, _ = runner.translate_onnx_model(
    ONNX_FILE,
    model_name,
    start_node_names=["images"],
    end_node_names=[
        "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv4.0/cv4.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv4.1/cv4.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv", "/model.22/cv4.2/cv4.2.2/Conv"
    ],
    net_input_shapes={"images": [1, 3, 640, 640]}
)

runner.save_har(str(STANDARD_HAR))
print(f"[info] Standard HAR saved: {STANDARD_HAR}")

# 3. Calibration Dataset Preparation
MAX_IMAGES = 1024
# Pathlib makes directory scanning cleaner
images_list = [img for img in CALIBRATION_IMAGES_PATH.iterdir() if img.suffix.lower() in ('.jpg', '.jpeg', '.png')][:MAX_IMAGES]

calib_dataset = np.zeros((len(images_list), 640, 640, 3), dtype=np.float32)

print(f"[info] Preparing calibration dataset with {len(images_list)} images...")
for idx, img_path in enumerate(sorted(images_list)):
    img = Image.open(img_path).convert('RGB').resize((640, 640))
    calib_dataset[idx] = np.array(img)

# 4. Optimization Stage (Quantization)
runner.load_model_script(str(MODEL_SCRIPT))
runner.optimize(calib_dataset)
runner.save_har(str(OPTIMIZED_HAR))
print(f"[info] Optimized HAR saved: {OPTIMIZED_HAR}")

# 5. Compilation Stage (HEF generation)
print("[info] Compiling to HEF...")
hef_binary = runner.compile()

with open(HEF_OUTPUT, "wb") as f:
    f.write(hef_binary)

print(f"[success] All files created in {parent_dir}:")
print(f"  - {STANDARD_HAR.name}")
print(f"  - {OPTIMIZED_HAR.name}")
print(f"  - {HEF_OUTPUT.name}")