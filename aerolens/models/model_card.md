# Model Card: AeroLens Sentinel Edge Defect Detector

## Model Details
- **Architecture**: YOLOv8n (optimized for edge devices)
- **Model Size/Params**: ~3.2M parameters
- **Format**: ONNX, optimized separately to OpenVINO FP32/FP16 (RPi target) and TensorRT FP16 (Jetson target)
- **Input Resolution**: 640x640 pixels
- **Supported Classes**: 
  - `crack` (class index 0)
  - `corrosion` (class index 1)
  - `dent` (class index 2)
  - `paint_peel` (class index 3)

## Training Configuration
- **Dataset**: Merged Aircraft-Surface-Damage and aircraft_skin_defects base corpus (~1,572 images).
- **Quantization Level**: INT8 / FP16 calibration.
- **Data Augmentation**: Geometric transforms, contrast adjustments, and custom synthetic overlay for paint peel surface indicators.
- **Class Mapping**:
  - `scratch` merged into `paint_peel`.
  - `missing-head` dropped (out of scope for visual skin panel defects).

## Intended Use
- **Primary Use Case**: On-device real-time visual inspection of aircraft skin panels.
- **Deployment hardware**: Jetson Orin Nano (TensorRT) and Raspberry Pi 5 (OpenVINO CPU/fallback).
- **Latency target**: Sub-200ms per frame.
