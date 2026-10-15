# Terratorch Object Detection — Status & Next Steps

## Overview & Context

This document tracks the object detection model development in terratorch.odapp for mobile deployment on the **Google Pixel 9** (Tensor G4 EdgeTPU / NPU).

The original baseline was **RF-DETR** (33M params, 133 MB ONNX), which could **not run on the NPU at all** due to unsupported ops (GridSample, LayerNormalization, ~850 dynamic shape ops), running at ~2.5 s / inference on CPU.

A replacement NPU-friendly architecture was trained, exported, and deployed:
- **Architecture**: **FCOS + EfficientNet-Lite0** (timm_efficientnet_lite0), 4.8M parameters, 5.1 MB int8 per-tensor ONNX (`fcos_detector_int8.onnx`).
- **Hardware Acceleration**: Successfully runs on the **Tensor G4 EdgeTPU** via ONNX Runtime's **NNAPI permissive** execution provider.
- **Measured Latency**: **9–13 ms** raw model inference on the EdgeTPU (TPU power rail active at 100–450 mW), ~56 ms end-to-end including frame extraction and Kotlin FCOS decoding + NMS.

---

## The Problem: Real-World Model Quality / Performance

While the model achieved high offline test metrics (**test mAP 0.9793** on the 98 trainod test set tiles), **in live testing on the Pixel 9 the model performs poorly**:

1. **High False Positive Rate / Background Sensitivity**: In live camera feeds and varied environments, the model produces dozens of spurious low-confidence bounding boxes (~50+ candidate boxes across the frame).
2. **Domain Shift & Lack of Generalization**: The training dataset (`/home/romeokienzler/Downloads/trainod/`) consists of 256x256 synthetic/overhead tiles. Training transforms only had horizontal/vertical flips. Real phone camera feeds have:
   - Variations in lighting, contrast, exposure, white balance.
   - Handheld camera motion blur and focus variations.
   - Scale variations and perspective/oblique camera angles (vs. perpendicular training crops).
3. **Dense Anchor False Positives**: FCOS generates **5,440 anchor locations** across 4 levels (Level 0 has 4,096 anchors at stride 4 alone). Without strong negative suppression or domain augmentation, many level-0 background patches score slightly above the 0.2–0.3 detection threshold.
4. **Limited Training**: Trained for only ~25 epochs with cosine annealing (`best-epoch=23.ckpt`), small batch size (16).
5. **Backbone Capacity**: EfficientNet-Lite0 is ultra-compact (4.8M params total) and uses depthwise-separable convolutions. While extremely fast on NPU, feature representation capacity for diverse backgrounds is limited compared to ResNet18 or larger backbones.

---

## Environment & Working Directories

| Resource | Path | Notes |
|---|---|---|
| Terratorch repo (branch `odapp`) | `/home/romeokienzler/gitco/terratorch.odapp` | Branch `odapp`, base `a8b8de36` |
| Python Environment | `/home/romeokienzler/tmp/.venv/bin/python` | Python 3.12, torch 2.11, torchvision 0.26, timm 1.0.26, onnx, onnxruntime |
| Training Dataset | `/home/romeokienzler/Downloads/trainod/` | `train_images/` (683 tiles), `val_images/` (195), `test_images/` (98) |
| Current Best Checkpoint | `/home/romeokienzler/Downloads/trainod/output/checkpoints_fcos_npu/best-epoch=23.ckpt` | val_map = 0.9864 |
| Training Config | `examples/object_detection/object_detection_trainod_fcos_npu.yaml` | FCOS + EfficientNet-Lite0 |
| Alternative Config | `examples/object_detection/object_detection_trainod_fcos_npu_r18.yaml` | FCOS + ResNet18 (drafted, not yet trained) |
| Export / Eval Scripts | `tools/npu/` | `export_trained_fcos.py`, `quantize_fcos_int8.py`, `eval_onnx_map.py` |
| Companion Android App | `/home/romeokienzler/gitco/odapp/` | `app/src/main/assets/fcos_detector_int8.onnx` |

**IMPORTANT**: Always run commands with:
```bash
export PYTHONPATH=/home/romeokienzler/gitco/terratorch.odapp
export PY=/home/romeokienzler/tmp/.venv/bin/python
```
Otherwise Python will load the un-editable site-packages copy of terratorch.

---

## Key Constraints for NPU Compatibility (Do Not Break These!)

Any new model or training change **must satisfy these rules** to retain EdgeTPU/NNAPI acceleration on the Pixel 9:

1. **`head_norm: batchnorm`**: GroupNorm in dense heads exports as `Reshape + InstanceNormalization`, which has NO NNAPI operator on EdgeTPU. BatchNorm folds into the preceding convolution and vanishes completely.
2. **Backbone must have standard NNAPI ops**: Standard convolutions, Relu/Clip, MaxPool, Add, Concat. No deformable attention (`GridSample`), no `LayerNorm`, no dynamic shapes.
3. **uint8 QDQ Per-Tensor Quantization**:
   - The Tensor G4 EdgeTPU requires **per-tensor uint8 QDQ**. Per-channel weights cause NNAPI to drop from 13 ms to 91 ms.
   - Do NOT quantize the head's float tail (`Reshape`/`Concat`/`Transpose` nodes). Use the verified `tools/npu/quantize_fcos_int8.py`.
4. **Input Size & Normalization**:
   - Current input is statically `[1, 3, 256, 256]` (or whatever resolution is exported).
   - Input pixels are raw **0–255 float** (`constant_scale=1`, `ToTensorV2`, no mean/std subtraction). Preprocessing must match this exactly.

---

## Actionable Plan for the Next Agent to Improve the Model

Here are the highest-impact improvements, ordered by priority:

### 1. Augmentation & Domain Randomization (Highest Impact)
The training data currently only uses `HorizontalFlip` and `VerticalFlip`. Update `train_transform` in `examples/object_detection/object_detection_trainod_fcos_npu.yaml` with realistic mobile camera degradations:
- `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)`
- `ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7)`
- `MotionBlur(blur_limit=5, p=0.3)` or `GaussianBlur(blur_limit=(3, 5), p=0.3)`
- `GaussNoise(var_limit=(10.0, 50.0), p=0.3)`
- `CoarseDropout` / `Cutout` (simulating partial occlusions)

### 2. Train FCOS with ResNet-18 Backbone
The config `examples/object_detection/object_detection_trainod_fcos_npu_r18.yaml` is already prepared.
- **Why**: ResNet-18 has 11M parameters (vs. 4.8M) with pure 3x3/1x1 standard convolutions (no depthwise convolutions). It has significantly higher feature capacity and handles background discrimination much better.
- **NPU Compatibility**: ResNet-18 + FPN + `head_norm: batchnorm` contains only Conv/Relu/Add/MaxPool — 100% supported by NNAPI/EdgeTPU.
- **How to train**:
  ```bash
  $PY -m terratorch fit -c examples/object_detection/object_detection_trainod_fcos_npu_r18.yaml \
      --trainer.accelerator cpu --data.init_args.num_workers 4
  ```

### 3. False Positive Suppression in Loss & Post-Processing
- **Focal Loss Gamma**: In FCOS classification, increase focal loss `gamma` (e.g. from 2.0 to 2.5 or 3.0) or tune the classification loss weight to more aggressively down-weight easy background examples.
- **Centerness Branch Tuning**: In `OnnxObjectDetector.kt`, the score is computed as `sqrt(sigmoid(cls_logits) * sigmoid(bbox_ctrness))`. If the centerness branch predicts ~0.5 on background, background false alarms can leak through. Check whether clamping or squaring centerness suppresses spurious detections.
- **Confidence Threshold**: For live camera usage, raise the deployment confidence threshold from 0.30 to 0.40 or 0.50, and evaluate precision vs. recall on real phone-captured images.

### 4. Background / Negative Mining
- Add pure background negative images (images with 0 bounding boxes) into the training split. FCOS will compute classification loss over all 5,440 anchors against label 0 (background), heavily penalizing false activations on non-target textures.

### 5. Multi-Scale Training / Resolution
- If target objects appear small on a phone screen, consider increasing training resolution to 384x384 (tested earlier, still runs fast on EdgeTPU at ~25 ms).

---

## How to Export, Quantize, and Validate

Once a new checkpoint is trained:

```bash
export PYTHONPATH=/home/romeokienzler/gitco/terratorch.odapp
export PY=/home/romeokienzler/tmp/.venv/bin/python

# 1. Export checkpoint to FP32 ONNX
$PY tools/npu/export_trained_fcos.py \
    --ckpt /path/to/best.ckpt \
    --size 256 \
    --out /home/romeokienzler/tmp/new_fcos_fp32.onnx

# 2. Quantize to INT8 per-tensor (verified EdgeTPU recipe)
$PY tools/npu/quantize_fcos_int8.py \
    --model /home/romeokienzler/tmp/new_fcos_fp32.onnx \
    --out /home/romeokienzler/tmp/new_fcos_int8.onnx \
    --size 256

# 3. Evaluate mAP against test set
$PY tools/npu/eval_onnx_map.py \
    --ckpt /path/to/best.ckpt \
    --onnx /home/romeokienzler/tmp/new_fcos_int8.onnx

# 4. Copy to Android app and test
cp /home/romeokienzler/tmp/new_fcos_int8.onnx /home/romeokienzler/gitco/odapp/app/src/main/assets/fcos_detector_int8.onnx
cd /home/romeokienzler/gitco/odapp
JAVA_HOME=.jdks/temurin-21 ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```