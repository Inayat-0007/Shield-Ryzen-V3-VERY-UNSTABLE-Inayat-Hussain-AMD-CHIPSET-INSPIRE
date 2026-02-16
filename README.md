# 🛡️ Shield-Ryzen V1 (Diamond Tier)

**Advanced Deepfake Detection Engine for AMD Ryzen AI NPUs**

![Status](https://img.shields.io/badge/Status-Diamond%20Tier-00e6e6)
![Precision](https://img.shields.io/badge/Precision-INT8%20Quantized-blue)
![Platform](https://img.shields.io/badge/Platform-AMD%20Ryzen%20AI-red)

## 🚀 Project Overview
Shield-Ryzen is a high-performance, privacy-focused specific deepfake detection system designed for the **AMD Slingshot 2026** competition. It leverages a custom XceptionNet backbone optimized for the **AMD Ryzen AI NPU** via INT8 quantization, achieving real-time inference with military-grade security logic.

## 💎 Diamond Tier Features
- **Universal ONNX Engine**: Silicon-agnostic architecture (Run on NVIDIA, AMD, Intel).
- **INT8 Quantization**: Compressed **20.49 MB** logic (vs 79 MB original) for maximum NPU efficiency.
- **Security Mode**:
  - **89% Confidence Rule**: strict threshold for verification.
  - **Liveness Detection**: EAR-based blink tracking prevents photo spoofing.
  - **Texture Guard**: Laplacian variance analysis detects smoothing artifacts.

## 🛠️ Tech Stack
- **Core AI**: PyTorch ➡ ONNX ➡ INT8 Quantization (QDQ)
- **Vision**: OpenCV + MediaPipe FaceLandmarker
- **Hardware Acceleration**: CUDA (Dev) / AMD XDNA (Target)

## 📂 Repository Structure
```
Shield-Ryzen-V1/
├── shield_ryzen_int8.onnx   # 💎 The Diamond Tier NPU Engine (20 MB)
├── shield_ryzen_v2.onnx     # 🚀 FP32 Universal Engine (79 MB)
├── v3_int8_engine.py        # 🖥️ Deployment Script (Run this!)
├── quantize_int8.py         # ⚙️ Quantization Pipeline
├── docs/                    # 📄 Architecture constraints
└── ...
```

## ⚡ Quick Start
1. **Install Dependencies**:
   ```bash
   pip install numpy opencv-python mediapipe onnxruntime-gpu
   ```
2. **Run the Engine**:
   ```bash
   python v3_int8_engine.py
   ```

---
*Developed by Inayat-Builder for AMD Slingshot 2026*
