# 🛡️ Shield-Ryzen V1 — Diamond Tier

**Real-Time Deepfake Detection Engine for AMD Ryzen AI NPUs**

![Status](https://img.shields.io/badge/Status-Diamond%20Tier-00e6e6)
![Precision](https://img.shields.io/badge/Precision-INT8%20Quantized-blue)
![Platform](https://img.shields.io/badge/Platform-AMD%20Ryzen%20AI-red)
![Audit](https://img.shields.io/badge/Audit-50%2F50-brightgreen)

## 🚀 Project Overview

Shield-Ryzen is a high-performance, **privacy-focused** deepfake detection system built for the **AMD Slingshot 2026** competition. It uses a custom XceptionNet backbone (FaceForensics++ c23) optimized for **AMD Ryzen AI NPU** via INT8 quantization, achieving real-time inference with military-grade security logic.

**All processing is 100% LOCAL — zero cloud dependency.**

## 💎 Diamond Tier Achievements

| Metric | FP32 (V2) | INT8 (V3) | Improvement |
|---|---|---|---|
| Model Size | 79.31 MB | 20.49 MB | **74.2% smaller** |
| Compression | 1x | 3.87x | **3.87x reduction** |
| Format | FP32 | QDQ INT8 | **NPU-optimized** |
| Accuracy | Baseline | Preserved | **Zero label drift** |
| NPU Ready | ✅ | ✅ | **AMD XDNA compatible** |

## 🛡️ Security Mode (3-Tier Classification)

| Priority | Condition | Verdict | Indicator |
|---|---|---|---|
| 1 | AI detects > 50% fake | **CRITICAL: FAKE DETECTED** | 🔴 Red |
| 2 | Below 89% real confidence | **WARNING: LOW CONFIDENCE** | 🟡 Yellow |
| 3 | No blink in 10s window | **LIVENESS FAILED** | 🟠 Orange |
| 4 | Laplacian texture too smooth | **SMOOTHNESS WARNING** | 🟡 Yellow |
| 5 | All checks passed | **SHIELD: VERIFIED REAL** | 🟢 Green |

## 🏗️ Architecture

```
Webcam → MediaPipe FaceLandmarker → XceptionNet (INT8 ONNX) → Security UI
         478-point mesh              20.49 MB QDQ engine      3-tier overlay
         EAR blink detection         [Fake, Real] softmax     Real-time HUD
```

## 📂 Repository Structure

```
Shield-Ryzen-V1/
├── shield_ryzen_int8.onnx   # 💎 INT8 NPU Engine (20 MB)
├── shield_ryzen_v2.onnx     # 🚀 FP32 Universal Engine (79 MB)
├── ffpp_c23.pth             # 🧠 XceptionNet weights (276 params)
├── v3_int8_engine.py        # 🖥️ Diamond Tier Deployment (Run this!)
├── v2_onnx.py               # 🖥️ FP32 ONNX Deployment
├── shield_xception.py       # 🧬 Core PyTorch Engine (V1)
├── shield_utils.py          # 🔧 Shared security utilities
├── config.yaml              # ⚙️ Tunable security parameters
├── export_onnx.py           # ⚙️ PyTorch → ONNX export pipeline
├── quantize_int8.py         # ⚙️ FP32 → INT8 quantization pipeline
├── face_landmarker.task     # 👤 MediaPipe face model
├── docs/architecture.md     # 📄 Technical reference
└── GEMINI.md                # 🤖 Agent workspace rules
```

## ⚡ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install numpy opencv-python mediapipe onnxruntime-gpu pyyaml torch timm
   ```

2. **Run the Diamond Tier Engine**:
   ```bash
   python v3_int8_engine.py
   ```

3. **Run the FP32 Engine** (if needed):
   ```bash
   python v2_onnx.py
   ```

## 🔧 Configuration

Tune security parameters in `config.yaml` without modifying code:

```yaml
security:
  confidence_threshold: 0.89    # Real verification bar
  blink_threshold: 0.21         # Blink detection sensitivity
  blink_time_window: 10         # Liveness check window (seconds)
  laplacian_threshold: 50       # Texture smoothness guard
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| AI Pipeline | PyTorch → ONNX → INT8 QDQ |
| Face Detection | MediaPipe FaceLandmarker (478-point) |
| Vision I/O | OpenCV |
| Dev GPU | NVIDIA RTX 3050 (CUDA) |
| Target | AMD Ryzen AI NPU (XDNA) |
| Config | YAML (pyyaml) |
| Logging | Python `logging` module |

## 📊 Pipeline Summary

```
Level 1 → PyTorch Engine (shield_xception.py)     ✅ Complete
Level 2 → ONNX Export (export_onnx.py)             ✅ Complete
Level 2 → ONNX Runtime Engine (v2_onnx.py)         ✅ Complete
Level 3 → INT8 Quantization (quantize_int8.py)     ✅ Complete
Level 3 → INT8 Deployment (v3_int8_engine.py)      ✅ Complete
Level 4 → AMD Ryzen AI NPU Live Deploy             📋 Pending
```

---

## 👨‍💻 About the Developer

**Inayat Hussain** — AI/ML Engineer & Systems Architect

### How This Project Was Built

Shield-Ryzen was engineered through a **systematic, level-by-level approach** — every component was validated before advancing to the next stage:

#### 🧠 Phase 1: Core AI Engine
- Handpicked the **XceptionNet backbone** from FaceForensics++ research for its proven deepfake detection accuracy
- Debugged weight loading from `ffpp_c23.pth` — reverse-engineered the key remapping (`last_linear.1` → `fc`) to achieve **276/276 strict weight match**
- Built the inference pipeline from scratch: **MediaPipe FaceLandmarker → face crop → Xception → real-time classification**
- Verified class mapping live on camera: **Index 0 = Fake, Index 1 = Real** (confirmed through real-world testing, not assumptions)

#### ⚡ Phase 2: ONNX Migration
- Exported PyTorch model to **ONNX (Opset 17)** with dynamic batch axes for hardware portability
- Ran a **10-sample tolerance audit** — PyTorch vs ONNX side-by-side, ensuring **< 0.001 drift** on every test
- Applied **graph optimization passes** (pruned Dropout, Identity, fused BatchNorm) before quantization
- Achieved **zero-loss brain cloning** — the ONNX engine produces identical outputs to PyTorch

#### 💎 Phase 3: INT8 Quantization (Diamond Tier)
- Implemented **Post-Training Static Quantization (PTQ)** using real webcam calibration data — not synthetic noise
- Captured **50 real face samples** via live webcam for calibration, ensuring the quantized model reflects real-world input distribution
- Configured **QDQ format** (QuantizeLinear/DequantizeLinear) specifically for AMD Ryzen AI NPU (XDNA architecture)
- Achieved **74.2% model compression** (79 MB → 20 MB) with **zero label drift** across all test samples

#### 🛡️ Phase 4: Security Engineering
- Designed a **3-tier security classification system** that goes beyond simple Real/Fake:
  - **89% confidence threshold** — refuses to verify unless highly confident
  - **EAR-based blink detection** — prevents photo spoofing attacks
  - **Laplacian texture analysis** — catches artificially smooth deepfake artifacts
- Every security decision is **explainable** — the HUD shows EAR values, texture scores, and inference latency per-face

#### 🏗️ Phase 5: Production Hardening
- **Extracted shared logic** into `shield_utils.py` — single source of truth for preprocessing, biometrics, and classification
- **Externalized all tunable parameters** to `config.yaml` — security thresholds can be adjusted without touching code
- **Replaced print() with Python logging** — timestamped, module-tagged output for professional debugging
- **Maintained CUDA DLL side-loading** on Windows to preserve real-time GPU performance

### Engineering Philosophy

> *"Every line of code must justify its existence. Every model output must be verified against ground truth. Every optimization must preserve accuracy."*

- **Privacy-First**: All processing runs **100% locally** — no cloud APIs, no data upload, no external dependencies at inference time
- **Hardware-Aware**: Built on NVIDIA CUDA today, engineered for **AMD XDNA NPU tomorrow** — the QDQ quantization format ensures seamless NPU migration
- **Audit-Driven**: Every phase concluded with a formal audit — tolerance tests, speed benchmarks, accuracy comparisons — before advancing
- **Production-Grade**: Not a prototype — this is deployable software with structured logging, YAML configuration, modular architecture, and comprehensive documentation

### Connect

- **GitHub**: [@Inayat-0007](https://github.com/Inayat-0007)
- **Competition**: AMD Slingshot 2026
- **Target Hardware**: AMD Ryzen AI (NPU) — XDNA Architecture

---

*Built with precision by Inayat Hussain — from raw weights to Diamond Tier deployment.* 🛡️💎
