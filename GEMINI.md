# 🛡️ Shield-Xception — Workspace Rules for AI Agent

## Project Identity
- **Project Name:** Shield-Xception (Shield-Ryzen V1)
- **Competition:** AMD Slingshot 2026
- **Developer:** Inayat Hussain
- **Status:** Active Development (Target: March 1, 2026)

---

## 🚨 CORE PILLARS — DO NOT MODIFY WITHOUT DEVELOPER APPROVAL

### 1. Hardware Target
- **Development GPU:** NVIDIA RTX 3050 Laptop GPU
- **All inference MUST use CUDA acceleration** (`torch.device('cuda')`)
- **Future Target:** AMD Ryzen AI NPU (via ONNX export)

### 2. Architecture
- **Core Engine:** XceptionNet (via `timm` library)
- **Weights:** `ffpp_c23.pth` (FaceForensics++ compressed, quality c23)
- **Class:** `ShieldXception` in `shield_xception.py` — **ALWAYS PRESERVE this class structure**
- **Output:** Single sigmoid neuron (1 = Fake, 0 = Real)

### 3. Preprocessing Pipeline
- **Face Detection:** MediaPipe `FaceDetection` (model_selection=0, lightweight)
- **Face Resize:** 299×299 pixels (Xception input requirement)
- **Normalization:** `[0.5, 0.5, 0.5]` mean and std (FF++ standard)

### 4. Trust Score Logic
- **Raw model output:** 1.0 = Fake, 0.0 = Real
- **Trust Score = 1 - raw_score** (inverted for UX: 1.0 = Trusted/Real, 0.0 = Fake)
- **Threshold:** 0.5 (trust_score > 0.5 → REAL, else → FAKE)
- **Display:** Green box + "TRUST: REAL (score)" or Red box + "TRUST: FAKE (score)"

### 5. AMD Compatibility Requirement
- All code must remain **ONNX-exportable** for future AMD Ryzen AI NPU deployment
- Do **NOT** use CUDA-only ops that break ONNX compatibility
- Do **NOT** introduce cloud-based APIs — **ALL processing must be LOCAL-ONLY** for privacy compliance

---

## 🧱 AGENT BEHAVIOR RULES

1. **Never rewrite `shield_xception.py` without explicit approval.** Suggest changes as diffs/plans.
2. **Never suggest cloud-based APIs.** All processing must remain local for privacy.
3. **Always preserve the `ShieldXception` class structure.** Modifications should extend, not replace.
4. **Always preserve the Trust Score logic** (inversion from raw model output).
5. **When suggesting optimizations**, present as an implementation plan first — do not auto-execute.
6. **All new scripts** should be in separate files, not injected into `shield_xception.py`.
7. **Respect the execution order:**
   - Level 1: Core engine (shield_xception.py) ✅ DONE
   - Level 1.5: FPS optimization of face-cropping loop
   - Level 2: ONNX export script for AMD NPU compatibility
   - Level 3+: UI overlay, multi-face, temporal analysis (future)

---

## 📁 PROJECT STRUCTURE
```
Shield Ryzen V1 inayat hussain/
├── shield_xception.py      # Core engine — DO NOT AUTO-MODIFY
├── ffpp_c23.pth             # Xception weights (FaceForensics++ c23)
├── GEMINI.md                # This file — Agent workspace rules
└── docs/
    └── architecture.md      # Architecture reference document
```

## 🔧 TECH STACK
| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.13                         |
| DNN Framework    | PyTorch + timm                      |
| Face Detection   | MediaPipe                           |
| Vision I/O       | OpenCV                              |
| GPU Runtime      | CUDA (NVIDIA RTX 3050)              |
| Future Runtime   | ONNX Runtime (AMD Ryzen AI NPU)     |
| Inference Mode   | Real-time webcam (30 FPS target)    |
