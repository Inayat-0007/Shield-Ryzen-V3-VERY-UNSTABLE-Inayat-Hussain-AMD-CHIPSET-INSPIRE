# Shield-Ryzen V2 — Complete Prototype Documentation
## 100% Knowledge Reference | AMD Slingshot 2026
### Developer: Inayat Hussain

---

# TABLE OF CONTENTS

1. [Project Identity](#1-project-identity)
2. [What This Prototype Does](#2-what-this-prototype-does)
3. [Architecture Overview](#3-architecture-overview)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
5. [The Complete Processing Pipeline](#5-the-complete-processing-pipeline)
6. [Neural Network: XceptionNet](#6-neural-network-xceptionnet)
7. [Face Detection & Preprocessing](#7-face-detection--preprocessing)
8. [The 3-Tier Security Decision System](#8-the-3-tier-security-decision-system)
9. [Plugin System](#9-plugin-system)
10. [Utility Algorithms & Math](#10-utility-algorithms--math)
11. [HUD Display System](#11-hud-display-system)
12. [Security & Privacy Features](#12-security--privacy-features)
13. [Configuration System](#13-configuration-system)
14. [How to Run](#14-how-to-run)
15. [Limitations](#15-limitations)
16. [Alternatives & Future Work](#16-alternatives--future-work)
17. [Dependencies](#17-dependencies)
18. [Glossary](#18-glossary)
19. [Complete Deep-Dive Folder Structure](#19-complete-deep-dive-folder-structure)

---

# 1. PROJECT IDENTITY

| Field | Value |
|---|---|
| **Name** | Shield-Ryzen V2 (Shield-Xception) |
| **Competition** | AMD Slingshot 2026 |
| **Developer** | Inayat Hussain |
| **Purpose** | Real-time deepfake detection via webcam |
| **Core Model** | XceptionNet (FaceForensics++ c23 weights) |
| **Dev GPU** | NVIDIA RTX 3050 Laptop GPU (CUDA) |
| **Target Hardware** | AMD Ryzen AI NPU (via ONNX Runtime) |
| **Inference** | 100% LOCAL — no cloud, no internet |
| **Language** | Python 3.13 |
| **Framework** | PyTorch + timm + ONNX Runtime |

---

# 2. WHAT THIS PROTOTYPE DOES

Shield-Ryzen V2 is a **real-time deepfake detection system** that runs entirely on your local machine. It captures video from your webcam, detects faces, and determines whether each face is **REAL** (a live human) or **FAKE** (a deepfake, screen replay, or mask).

**In plain English:** You sit in front of your webcam. The system draws a box around your face and shows a trust verdict:
- **Green "VERIFIED"** = You are a real, live human
- **Red "FAKE"** = The system detected a deepfake or presentation attack
- **Orange "SUSPICIOUS"** = Something is off but not confirmed fake
- **Yellow "WAIT_BLINK"** = Waiting for you to blink to prove liveness

It does this by combining:
1. A **neural network** (XceptionNet) that classifies face images as real/fake
2. **Liveness checks** (blink detection, eye tracking)
3. **Forensic analysis** (texture sharpness, frequency spectrum, screen light detection)
4. **Plugin-based analysis** (heartbeat detection, adversarial patch detection, etc.)

---

# 3. ARCHITECTURE OVERVIEW

## 3.1 Triple-Buffer Async Pipeline

The system runs on **3 parallel threads** for maximum performance:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ CAMERA THREAD│────>│   AI THREAD  │────>│  MAIN THREAD  │
│  (Producer)  │     │ (Processor)  │     │  (HUD/Display)│
│              │     │              │     │               │
│ Captures     │     │ Face detect  │     │ Renders HUD   │
│ frames from  │     │ Neural infer │     │ Shows window  │
│ webcam       │     │ Plugin votes │     │ Handles keys  │
│              │     │ State machine│     │               │
│ Validates    │     │ Audit log    │     │ cv2.imshow()  │
│ frame quality│     │              │     │               │
└──────────────┘     └──────────────┘     └──────────────┘
     │                     │                     │
     └── camera_queue ─────┘── result_queue ─────┘
         (maxsize=1)           (maxsize=1)
```

**Why maxsize=1?** Single-frame buffers ensure the system always processes the **latest** frame, not stale ones. This minimizes latency at the cost of dropping frames when the AI thread is busy.

## 3.2 Data Flow (One Frame)

```
Webcam → ShieldCamera.read_validated_frame()
       → camera_queue
       → ShieldEngine._ai_thread_loop()
         ├── ShieldFacePipeline.detect_faces(frame)
         │   ├── MediaPipe FaceLandmarker → 478 landmarks
         │   ├── Convert to 68-point landmarks
         │   ├── Head pose estimation (yaw/pitch/roll)
         │   ├── Occlusion scoring
         │   └── align_and_crop() → 299×299 normalized tensor
         ├── Neural inference (ONNX or PyTorch)
         │   └── XceptionNet → [fake_prob, real_prob]
         ├── ConfidenceCalibrator.calibrate() (temperature scaling)
         ├── Tier 1: Neural verdict (REAL/FAKE)
         ├── Tier 2: Liveness check
         │   ├── compute_ear() with cosine correction
         │   ├── BlinkTracker.update() (blendshape or EAR-DBS)
         │   └── Blink pattern analysis
         ├── Tier 3: Forensic check
         │   ├── compute_texture_score() (5-layer screen detection)
         │   └── Frequency analysis, Moiré, screen light
         ├── Plugin Analysis (heartbeat, skin, codec, etc.)
         ├── DecisionStateMachine.update(T1, T2, T3, plugins)
         │   └── Truth table + agile hysteresis
         └── FaceResult → result_queue
       → ShieldHUD.render(frame, engine_result)
       → cv2.imshow() → Screen
```

---

# 4. FILE-BY-FILE BREAKDOWN

## Core Engine Files

| File | Lines | Role |
|---|---|---|
| `shield_engine.py` | 897 | Central orchestrator — threads, pipeline, state machine |
| `shield_xception.py` | 512 | XceptionNet model class + SHA-256 integrity verification |
| `shield_face_pipeline.py` | 900 | Face detection (MediaPipe/DNN), landmarks, normalization |
| `shield_utils_core.py` | 1269 | EAR, texture analysis, calibration, state machine, blink tracker |
| `shield_camera.py` | 326 | Camera capture with validation and health monitoring |
| `shield_hud.py` | 536 | Glassmorphism HUD overlay rendering |
| `shield_crypto.py` | 114 | AES-256-GCM biometric encryption |
| `shield_logger.py` | 108 | JSONL structured audit logging |
| `shield_plugin.py` | 69 | Abstract base class for plugins |
| `shield_types.py` | 32 | FaceResult and EngineResult dataclasses |
| `start_shield.py` | 260 | Main launcher with CLI arguments |
| `v3_xdna_engine.py` | 84 | AMD XDNA NPU engine subclass |
| `export_onnx.py` | 76 | PyTorch → ONNX conversion script |
| `config.yaml` | 45 | Runtime configuration |

## Plugin Files (in `plugins/`)

| File | Lines | Purpose |
|---|---|---|
| `rppg_heartbeat.py` | 172 | Remote photoplethysmography (heartbeat from skin color) |
| `challenge_response.py` | 163 | "Simon Says" liveness (blink, turn head, smile) |
| `stereo_depth.py` | 133 | Dual-camera 3D depth verification |
| `skin_reflectance.py` | 119 | Specular highlight + gradient texture analysis |
| `codec_forensics.py` | 118 | JPEG/H.264 double-compression detection |
| `frequency_analyzer.py` | 117 | FFT-based GAN artifact detection |
| `adversarial_detector.py` | 113 | Physical adversarial patch detection |
| `lip_sync_verifier.py` | 105 | Lip-reading phoneme verification |
| `arcface_reid.py` | 149 | ArcFace identity re-identification |

---

# 5. THE COMPLETE PROCESSING PIPELINE

## 5.1 Startup Sequence

1. `start_shield.py` parses CLI args (`--cpu`, `--source`, `--windowed`, etc.)
2. Creates `ShieldEngine(config)` which:
   - Initializes `ShieldCamera` (DirectShow backend on Windows)
   - Initializes `ShieldFacePipeline` (loads MediaPipe `face_landmarker_v2_with_blendshapes.task`)
   - Loads neural model: prefers `shield_ryzen_int8.onnx` (ONNX), falls back to `ffpp_c23.pth` (PyTorch)
   - SHA-256 hash verifies model integrity
   - Creates `ConfidenceCalibrator(temperature=1.5)`
   - Creates `DecisionStateMachine(frames=5)` per face
   - Creates `BlinkTracker` per face
   - Loads all plugins from `plugins/` directory
   - Runs startup calibration (measures camera baseline)
3. `engine.start()` launches camera thread + AI thread
4. Main loop: `engine.get_latest_result()` → `hud.render()` → `cv2.imshow()`

## 5.2 Per-Frame Processing (AI Thread)

For each frame from the camera queue:

### Stage 1: Face Detection
- MediaPipe FaceLandmarker processes the RGB frame
- Returns up to 2 faces (configurable) with 478-point mesh landmarks
- Converts 478→68 landmark mapping for EAR compatibility
- Estimates head pose: yaw (left/right), pitch (up/down), roll (tilt)
- Computes occlusion score from landmark depth variance
- `align_and_crop()`: expands bbox, crops face, resizes to 299×299, BGR→RGB, normalizes to [-1,1], transposes to NCHW

### Stage 2: Neural Inference
- Input: `[1, 3, 299, 299]` float32 tensor (normalized [-1, 1])
- Model outputs `[fake_prob, real_prob]` via softmax
- Temperature scaling: `logits / 1.5` then re-softmax (prevents overconfidence)
- Trust score = `real_prob` (inverted: 1.0 = trusted real, 0.0 = fake)

### Stage 3: Three-Tier Decision

**Tier 1 — Neural:** `trust_score > 0.5` → REAL, else FAKE

**Tier 2 — Liveness:**
- Compute EAR (Eye Aspect Ratio) with cosine angle compensation
- Track blinks via blendshapes (priority) or EAR-DBS (fallback)
- PASS if blink detected within time window

**Tier 3 — Forensic:**
- 5-layer texture analysis:
  - Layer 1: Laplacian variance (sharpness)
  - Layer 2: Distance-texture physics cross-validation
  - Layer 3: Moiré pattern detection (FFT radial profile)
  - Layer 4: Screen light emission (brightness uniformity, chrominance, blue bias)
  - Layer 5: Weak-signal fusion (2+ marginal signals = confirm)
- PASS if not suspicious

### Stage 4: Truth Table Fusion

```
Neural | Liveness | Forensic | → State
-------|----------|----------|--------
REAL   | PASS     | PASS     | REAL → promoted to VERIFIED by engine
REAL   | PASS     | FAIL     | SUSPICIOUS
REAL   | FAIL     | PASS     | WAIT_BLINK
REAL   | FAIL     | FAIL     | HIGH_RISK
FAKE   | PASS     | PASS     | FAKE
FAKE   | PASS     | FAIL     | FAKE
FAKE   | FAIL     | PASS     | FAKE
FAKE   | FAIL     | FAIL     | CRITICAL
```

### Stage 5: Hysteresis
- **Escalation** (→ FAKE/CRITICAL/HIGH_RISK): Immediate (1 frame)
- **De-escalation** (→ REAL/VERIFIED): Requires 5 consecutive frames
- Prevents flickering between states

---

# 6. NEURAL NETWORK: XCEPTIONNET

## 6.1 Architecture
- **Base:** XceptionNet from `timm.create_model('legacy_xception')`
- **Modification:** Final classifier replaced with `Linear(2048, 2)` + Softmax
- **Output:** `[fake_probability, real_probability]`
- **Training Data:** FaceForensics++ dataset, quality c23 (compressed)
- **Weights File:** `ffpp_c23.pth` (83.5 MB, PyTorch) or `shield_ryzen_int8.onnx` (21.4 MB, quantized)

## 6.2 Why XceptionNet?
- Depthwise separable convolutions are efficient for mobile/edge
- Proven state-of-the-art on FaceForensics++ benchmark
- ONNX-exportable for AMD NPU deployment
- Good balance of accuracy vs. inference speed

## 6.3 Normalization (CRITICAL)
```
Input: BGR face crop from webcam
1. BGR → RGB conversion
2. Resize to 299×299
3. Scale to [0, 1]: pixel / 255.0
4. Normalize: (pixel - 0.5) / 0.5
5. Result: [-1.0, +1.0] range (FaceForensics++ standard)
6. Transpose: HWC → CHW → NCHW [1, 3, 299, 299]
```
**If normalization is wrong, the model produces garbage.** This is the #1 bug source.

## 6.4 Model Security
- SHA-256 hash verification on load (prevents tampering)
- Key count verification (expected number of weight tensors)
- Shape verification for ONNX models
- `ModelTamperingError` raised if any check fails

## 6.5 Temperature Scaling
Raw softmax outputs are overconfident (e.g., 0.99 when should be 0.7).
```
logits = log(softmax_probs)
scaled_logits = logits / 1.5   # temperature > 1 = softer
calibrated = softmax(scaled_logits)
```
This makes the 89% confidence threshold actually meaningful.

---

# 7. FACE DETECTION & PREPROCESSING

## 7.1 MediaPipe FaceLandmarker (Primary)
- Model: `face_landmarker_v2_with_blendshapes.task` (3.75 MB)
- 478-point face mesh with 52 blendshape coefficients
- Provides: bounding box, landmarks, head pose, blendshapes, transformation matrix
- Runs in VIDEO mode for temporal smoothing

## 7.2 DNN SSD (Fallback)
- OpenCV's DNN module with SSD face detector
- No landmarks, no blendshapes, no head pose
- Used only when MediaPipe is unavailable

## 7.3 478→68 Landmark Mapping
MediaPipe provides 478 mesh points. Standard face analysis uses 68.
The mapping selects specific mesh indices to approximate the 68-point Dlib standard:
- Jawline: 17 points (indices 10, 338, 297, ...)
- Eyebrows: 10 points
- Nose: 9 points
- Eyes: 12 points (6 per eye)
- Mouth: 20 points

## 7.4 Head Pose Estimation
Computed from nose tip, chin, eye corners, and mouth corners landmarks using `cv2.solvePnP()`:
- **Yaw:** Left/right rotation (-45° to +45°)
- **Pitch:** Up/down rotation (-30° to +30°)
- **Roll:** Head tilt

## 7.5 Occlusion Scoring
Measures how much of the face is occluded (hand, hair, mask edge):
- Examines z-depth variance of forehead, chin, cheek landmarks
- High variance = likely occluded
- Used to reduce confidence in EAR and neural results

---

# 8. THE 3-TIER SECURITY DECISION SYSTEM

## Tier 1: Neural Verdict
- XceptionNet classifies face as real/fake
- Temperature-calibrated confidence
- Threshold: trust_score > 0.5 → REAL

## Tier 2: Liveness (Blink Detection)

### EAR Formula
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```
Where p1-p6 are the 6 eye landmark points (outer, upper1, upper2, inner, lower2, lower1).

### Cosine Angle Compensation
At non-frontal angles, the horizontal eye distance foreshortens:
```
if |yaw| > 20°: corrected_EAR = raw_EAR / cos(yaw_radians)
if |yaw| > 30°: reliability = "LOW" (EAR unreliable)
```

### Dynamic Baseline Scaling (DBS)
Instead of a fixed threshold, the system learns YOUR eye openness:
```
open_state_ear = 0.9 × old + 0.1 × new  (if new > old, fast adapt up)
open_state_ear = 0.998 × old + 0.002 × new  (slow decay down)
close_threshold = open_state_ear × 0.65
reopen_threshold = open_state_ear × 0.90
```

### Blink Validation
- Duration: 60ms–500ms (reject noise < 60ms, reject long close > 500ms)
- Depth: peak EAR must be < 70% of open baseline
- Stuck detection: auto-reset if "in blink" > 1 second

### Blendshape Priority
If MediaPipe blendshapes are available (indices 9=EyeBlinkLeft, 10=EyeBlinkRight):
- Average > 0.5 = eyes closed
- Higher quality than EAR (works at all angles)

### Pattern Analysis
- Coefficient of variation (CV) of inter-blink intervals
- CV > 0.3 = natural (varying intervals), score 1.0
- CV < 0.3 = robotic (too regular), score proportional
- No blinks for 10+ seconds = suspicious

## Tier 3: Forensic (Texture + Frequency)

### Layer 1: Laplacian Variance
```
lap_var = cv2.Laplacian(forehead_gray, CV_64F).var()
threshold = device_baseline × 0.4  (or default 15.0)
```
Low variance = blurry/smooth = suspicious (printed photo, low-res deepfake)

### Layer 2: Distance-Texture Physics
```
max_expected = 700 × (50 / distance_cm)²
max_allowed = max_expected × 8.0  (safety margin)
```
If texture exceeds physics-based maximum at given distance → signal (not confirmation alone)

### Layer 3: Moiré Detection
- 2D FFT of forehead ROI
- Build radial frequency profile
- Autocorrelation to detect periodic peaks
- Screens have regular pixel grids → periodic frequency spikes
- Score > 0.70 = definite screen

### Layer 4: Screen Light Detection
Three sub-signals:
- **Brightness uniformity:** Real faces have shadows (CV 0.25-0.55), screens are uniform (CV 0.08-0.18)
- **Chrominance range:** Real skin has rich color variation, screens have narrow gamut
- **Blue channel bias:** LED backlights emit excess blue (B/G ratio > 0.92)

Combined: `total = brightness×0.45 + chroma×0.35 + blue×0.20`

### Layer 5: Weak-Signal Fusion
- Requires **physical evidence** (Moiré or screen light) — physics alone is NEVER enough
- 2+ signals above 0.35 with at least one physical evidence signal
- Combined average > 0.45 → confirm screen replay

---

# 9. PLUGIN SYSTEM

## 9.1 Architecture
All plugins extend `ShieldPlugin` (abstract base class):
```python
class ShieldPlugin(ABC):
    name: str        # e.g., "heartbeat_rppg"
    tier: str        # "biometric", "forensic", "neural", "temporal"
    analyze(face, frame) → dict  # Returns verdict, confidence, explanation
    release()        # Cleanup
```

## 9.2 Plugin Voting
- Each plugin returns: `{"verdict": "REAL"|"FAKE"|"UNCERTAIN", "confidence": float}`
- **Majority rule:** If >50% of plugins vote FAKE → plugin consensus = FAKE
- **Strong consensus (75%+):** Can override neural verdict
- UNCERTAIN votes are excluded from counts

## 9.3 Active Plugins

| Plugin | Type | What It Detects |
|---|---|---|
| **Heartbeat (rPPG)** | Biometric | Blood flow pulse via green channel FFT (42-180 BPM) |
| **Challenge-Response** | Biometric | Replay attacks via "Simon Says" actions |
| **Stereo Depth** | Biometric | 2D flat screens via dual-camera parallax |
| **Skin Reflectance** | Biometric | Masks via specular/gradient analysis |
| **Frequency Analyzer** | Forensic | GAN artifacts via 2D FFT high-frequency ratio |
| **Codec Forensics** | Forensic | Re-encoded streams via 8×8 blocking artifacts |
| **Adversarial Patch** | Forensic | Physical patches via Sobel gradient clustering |
| **Lip Sync** | Forensic | Pre-recorded video via phoneme verification |
| **ArcFace Re-ID** | Identity | Unknown users via 512-d embedding matching |

---

# 10. UTILITY ALGORITHMS & MATH

## 10.1 Distance Estimation
Two methods:
1. **Transformation matrix** (preferred): `z_cm = matrix[2, 3]` from MediaPipe
2. **Pinhole camera model** (fallback): `D = (F × 14cm) / bbox_width_px`
   - Assumes average face width = 14cm
   - Focal length ≈ frame width (approximation)

## 10.2 Device Calibration
At startup, captures ~100 frames to measure:
- Laplacian variance baseline (camera sharpness)
- User's natural EAR range
- Lighting condition (GOOD/MODERATE/LOW_LIGHT)
- Camera resolution
Saves to `shield_calibration.json` for runtime use.

## 10.3 Signal Smoother
Exponential moving average per-face:
```
smoothed = α × new + (1-α) × old
α = 0.3 for neural confidence
α = 0.15 for texture scores
```

## 10.4 Face Tracking
IoU (Intersection over Union) based tracker assigns persistent IDs to faces across frames. Stale trackers are purged when faces disappear.

---

# 11. HUD DISPLAY SYSTEM

## 11.1 Design
- **Glassmorphism** style with translucent panels
- Color-coded state badges above each face
- Right-side dashboard with detailed metrics
- Bottom bar with uptime and camera health

## 11.2 Color Palette
| State | Color | Meaning |
|---|---|---|
| VERIFIED | Green (#00FF88) | Confirmed real + live |
| REAL | Light Green | Neural says real, awaiting full verification |
| WAIT_BLINK | Yellow (#FFD700) | Needs blink for liveness |
| SUSPICIOUS | Orange (#FF8C00) | Conflicting signals |
| HIGH_RISK | Dark Orange | Only neural passes |
| FAKE | Red (#FF0000) | Confirmed deepfake |
| CRITICAL | Dark Red (#8B0000) | All checks failed |

## 11.3 Dashboard Metrics
- Neural confidence (progress bar)
- EAR value + reliability grade
- Blink count + pattern score
- Distance (cm)
- Head pose (yaw/pitch/roll)
- Tier verdicts (T1/T2/T3)
- Plugin status (dots)

---

# 12. SECURITY & PRIVACY FEATURES

## 12.1 Biometric Encryption (`shield_crypto.py`)
- **AES-256-GCM** authenticated encryption
- Ephemeral keys (generated per session, never stored to disk)
- All face crops and landmarks encrypted in RAM
- Decrypted only during active processing
- Keys securely wiped on exit

## 12.2 Model Integrity
- SHA-256 hash verification on model load
- `ModelTamperingError` if hash mismatch
- Key count and shape verification

## 12.3 Audit Logging (`shield_logger.py`)
- JSONL format (one JSON object per line)
- Every frame's decision logged with:
  - Timestamp, face count, per-face results
  - FPS, timing breakdown, memory usage
- Thread-safe (mutex-locked file writes)

## 12.4 Privacy Promise
- **100% local processing** — no network calls, no cloud
- No face images stored to disk (encrypted in-memory only)
- No telemetry, no analytics
- Designed for GDPR/CCPA compliance

---

# 13. CONFIGURATION SYSTEM

## `config.yaml` Key Settings
```yaml
security:
  confidence_threshold: 0.89   # 89% rule for verification
  blink_threshold: 0.24        # EAR closed threshold
  blink_time_window: 10        # Seconds to detect a blink
  laplacian_threshold: 15      # Minimum texture sharpness

preprocessing:
  input_size: 299              # XceptionNet input resolution
  mean: [0.5, 0.5, 0.5]       # FF++ normalization
  std:  [0.5, 0.5, 0.5]

landmarks:
  left_eye:  [33, 160, 158, 133, 153, 144]
  right_eye: [362, 385, 387, 263, 373, 380]

mediapipe:
  num_faces: 2
  landmarker_model: "face_landmarker_v2_with_blendshapes.task"
```

---

# 14. HOW TO RUN

```bash
# Install dependencies
pip install -r requirements.txt

# Run with CPU (standard engine)
python start_shield.py --cpu

# Run with specific camera
python start_shield.py --cpu --source 1

# Run windowed
python start_shield.py --cpu --windowed --size 1280x720

# Run with AMD NPU optimization
python start_shield.py
```

**Keyboard Controls:**
- `Q` or `ESC`: Exit
- `F`: Toggle fullscreen
- `+`/`-`: Resize window
- `1`-`4`: Preset sizes (960×540 to 1920×1080)

---

# 15. LIMITATIONS

## 15.1 Neural Network
- Trained only on FaceForensics++ (4 manipulation methods)
- May not generalize to novel deepfake techniques (diffusion models, etc.)
- Single-dataset training = domain bias
- No fine-tuning on real-world webcam data

## 15.2 Face Detection
- MediaPipe struggles with extreme angles (>45° yaw)
- Small faces at distance (>1.5m) may not be detected
- Only processes up to 2 faces simultaneously
- No face recognition (just detection + analysis)

## 15.3 Liveness Detection
- EAR unreliable at >30° yaw (marked LOW reliability)
- Blink detection requires ~4 seconds of observation to establish baseline
- People who rarely blink may trigger WAIT_BLINK
- Glasses can affect specular reflection analysis

## 15.4 Forensic Analysis
- Screen replay detection calibrated for LCD/OLED — may miss projectors
- Moiré detection depends on camera-screen distance and angle
- Indoor LED lighting can trigger false positives on blue channel analysis
- Physics-based distance model is approximate (inverse-square assumption)

## 15.5 Performance
- Real-time at ~15-25 FPS on RTX 3050 (GPU) or ~8-15 FPS (CPU)
- Memory grows over long sessions (GC triggered at 500MB growth)
- ONNX INT8 model is ~4x smaller but may lose accuracy on edge cases

## 15.6 Security
- Python's garbage collector doesn't guarantee memory clearing
- XOR fallback if `cryptography` package missing (weak obfuscation)
- Model weights could theoretically be reverse-engineered
- Adversarial attacks specifically targeting XceptionNet may bypass detection

## 15.7 Hardware
- Currently optimized for NVIDIA CUDA (dev environment)
- AMD NPU deployment via Vitis AI EP is untested in production
- Single-camera setup limits depth-based anti-spoofing

---

# 16. ALTERNATIVES & FUTURE WORK

## 16.1 Alternative Models
| Alternative | Pros | Cons |
|---|---|---|
| EfficientNet-B4 | Smaller, faster | Less proven on FF++ |
| Vision Transformer (ViT) | Better generalization | Heavy, not ONNX-friendly |
| Capsule Networks | Rotation invariant | Slow inference |
| Multi-task CNN | Joint detection+classification | Complex training |

## 16.2 Alternative Face Detectors
| Alternative | Pros | Cons |
|---|---|---|
| RetinaFace | More accurate landmarks | Heavier model |
| SCRFD | Very fast | Fewer landmarks |
| YOLOv8-Face | Real-time | No mesh landmarks |
| Dlib HOG+68 | CPU friendly | Poor on angles |

## 16.3 Future Enhancements
- **Temporal analysis:** Analyze face consistency across multiple seconds
- **Multi-camera fusion:** Production stereo depth verification
- **Audio-visual sync:** Correlate lip movement with speech audio
- **Continual learning:** Online adaptation to new deepfake methods
- **AMD NPU deployment:** Full Vitis AI quantization and optimization
- **Multi-face tracking:** Handle meeting/conference scenarios
- **Grad-CAM visualization:** Show which face regions trigger detection

---

# 17. DEPENDENCIES

```
onnxruntime==1.18.0        # ONNX model inference
opencv-python==4.9.0.80    # Video capture, image processing
numpy==1.26.4              # Array math
mediapipe==0.10.14         # Face detection + landmarks
torch==2.2.2               # PyTorch (model loading, CUDA)
torchvision==0.17.2        # Image transforms
timm==1.0.3                # XceptionNet architecture
psutil==5.9.8              # System monitoring
py-cpuinfo==9.0.0          # Hardware detection
PyYAML==6.0.1              # Config loading
Pillow==10.3.0             # Image handling
cryptography               # AES-256-GCM encryption (optional)
scipy                      # Signal processing for rPPG (optional)
```

---

# 18. GLOSSARY

| Term | Definition |
|---|---|
| **EAR** | Eye Aspect Ratio — ratio of eye height to width |
| **rPPG** | Remote Photoplethysmography — detecting pulse from skin color |
| **FFT** | Fast Fourier Transform — frequency domain analysis |
| **Moiré** | Interference pattern from screen pixel grids |
| **Laplacian** | Second derivative operator measuring image sharpness |
| **DBS** | Dynamic Baseline Scaling — adaptive threshold learning |
| **HFER** | High-Frequency Energy Ratio — forensic frequency metric |
| **BAR** | Blocking Artifact Ratio — compression detection metric |
| **SNR** | Signal-to-Noise Ratio — quality of detected signal |
| **GCM** | Galois/Counter Mode — authenticated encryption mode |
| **NCHW** | Batch×Channels×Height×Width tensor format |
| **ONNX** | Open Neural Network Exchange — portable model format |
| **XDNA** | AMD's AI engine architecture for Ryzen AI NPUs |
| **Vitis AI** | AMD's AI inference optimization toolkit |
| **Hysteresis** | Delayed state transitions to prevent flickering |

---

# 19. COMPLETE DEEP-DIVE FOLDER STRUCTURE

Every file and folder in the project, with its exact purpose:

```
SHIELD RYZEN V2 UPDATE 1 INAYAT/
│
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL CORE ENGINE FILES (the brain of the prototype)
│   ══════════════════════════════════════════════════════════
│
├── shield_engine.py              [897 lines] CENTRAL ORCHESTRATOR
│   │   The main engine class (ShieldEngine). Contains:
│   │   - Triple-buffer async pipeline (camera→AI→HUD)
│   │   - Thread management (camera_thread, ai_thread)
│   │   - Per-face state machine (PluginAwareStateMachine)
│   │   - Plugin loading and voting aggregation
│   │   - Memory monitoring (GC at >500MB growth)
│   │   - JSONL audit logging per frame
│   │   - Face tracker with IoU-based ID assignment
│   │   - Fake lockout logic (prevents instant de-escalation)
│   │   - FaceResult/EngineResult construction
│   │   THIS IS THE FILE THAT TIES EVERYTHING TOGETHER.
│   │
├── shield_xception.py            [512 lines] NEURAL NETWORK MODEL
│   │   ShieldXception class wrapping timm's XceptionNet.
│   │   - Linear(2048, 2) + Softmax final layer
│   │   - SHA-256 hash verification on model load
│   │   - Key count verification (prevents wrong weights)
│   │   - ModelTamperingError exception class
│   │   - Legacy EAR/texture functions (backward compat)
│   │   - load_model_with_verification() secure loader
│   │
├── shield_face_pipeline.py       [900 lines] FACE DETECTION + PREPROCESSING
│   │   All face detection, alignment, landmark extraction.
│   │   - MediaPipe FaceLandmarker backend (primary)
│   │   - OpenCV DNN SSD backend (fallback)
│   │   - 478→68 landmark index mapping
│   │   - Head pose estimation via cv2.solvePnP()
│   │   - Occlusion scoring from z-depth variance
│   │   - align_and_crop(): resize→BGR2RGB→normalize[-1,1]→NCHW
│   │   - FaceDetection dataclass with all face metadata
│   │   - Landmark confidence estimation
│   │
├── shield_utils_core.py          [1269 lines] ALGORITHMS + DECISION LOGIC
│   │   The mathematical brain. Contains 5 major components:
│   │   A) compute_ear() — EAR with cosine angle compensation
│   │   B) compute_texture_score() — 5-layer screen detection
│   │      ├── Laplacian variance
│   │      ├── Distance-texture physics check
│   │      ├── Moiré pattern detection (FFT radial profile)
│   │      ├── Screen light emission detection
│   │      └── Weak-signal fusion (2+ signals required)
│   │   C) calibrate_device_baseline() — domain adaptation
│   │   D) ConfidenceCalibrator — temperature scaling
│   │   E) DecisionStateMachine — truth table + hysteresis
│   │   Also: BlinkTracker (DBS), SignalSmoother (EMA),
│   │         estimate_distance(), classify_face()
│   │
├── shield_camera.py              [326 lines] CAMERA INPUT + VALIDATION
│   │   ShieldCamera wrapping cv2.VideoCapture.
│   │   - DirectShow backend on Windows
│   │   - Single-frame buffer (latency minimization)
│   │   - Frame validation: shape, dtype, channels, brightness
│   │   - Frame freshness detection (stale frame rejection)
│   │   - Health monitoring: FPS, drop rate, connection status
│   │   - Thread-safe read with proper cleanup
│   │
├── shield_hud.py                 [536 lines] HEADS-UP DISPLAY
│   │   Modern glassmorphism-style fullscreen HUD.
│   │   - Translucent rounded panels
│   │   - Color-coded state badges above faces
│   │   - Right-side dashboard (confidence, EAR, blinks, etc.)
│   │   - Alert banners for FAKE/CRITICAL states
│   │   - Animated pulse ring for VERIFIED state
│   │   - Top bar (logo, LIVE indicator, FPS)
│   │   - Bottom bar (uptime, camera health)
│   │   - Helper: draw_rounded_rect, draw_text, draw_progress_bar
│   │
├── shield_crypto.py              [114 lines] BIOMETRIC ENCRYPTION
│   │   AES-256-GCM in-memory encryption for face data.
│   │   - Ephemeral keys (per session, never stored)
│   │   - Pickle serialization → AES encrypt
│   │   - XOR fallback if cryptography package missing
│   │   - secure_wipe() to dereference keys
│   │   - Singleton pattern for global access
│   │
├── shield_logger.py              [108 lines] AUDIT LOGGING
│   │   ShieldLogger writing JSONL to logs/shield_audit.jsonl.
│   │   - Thread-safe (mutex lock on file writes)
│   │   - ShieldJSONEncoder handles numpy types
│   │   - Levels: AUDIT, WARN, ERROR, SYSTEM
│   │   - log_frame() for per-frame decisions
│   │   - Auto-records system startup/shutdown
│   │
├── shield_plugin.py              [69 lines] PLUGIN INTERFACE
│   │   Abstract base class (ABC) for all plugins.
│   │   - name (str) — unique identifier
│   │   - tier (str) — biometric/forensic/neural/temporal
│   │   - analyze(face, frame) → dict — returns vote
│   │   - release() — optional cleanup
│   │
├── shield_types.py               [32 lines] DATA TYPES
│   │   @dataclass definitions for structured results.
│   │   - FaceResult: bbox, state, confidence, EAR, texture, tiers
│   │   - EngineResult: frame, state, face_results, fps, timing
│   │
├── start_shield.py               [260 lines] MAIN LAUNCHER
│   │   Entry point: python start_shield.py
│   │   - CLI args: --cpu, --source, --windowed, --size, --model
│   │   - Window management (fullscreen/windowed toggle)
│   │   - Keyboard shortcuts (Q/ESC/F/+/-/1-4)
│   │   - Engine selection (ShieldEngine vs RyzenXDNAEngine)
│   │   - Clean shutdown sequence (stop engine → destroy windows)
│   │   - Crash log writing to crash_log.txt
│   │
├── v3_xdna_engine.py             [84 lines] AMD NPU ENGINE
│   │   RyzenXDNAEngine subclass of ShieldEngine.
│   │   - Enforces VitisAI Execution Provider
│   │   - Process priority boost (ABOVE_NORMAL on Windows)
│   │   - NPU status reporting placeholder
│   │   - Backward compat alias: ShieldXDNAEngine
│   │
├── v3_int8_engine.py             [~26K] LEGACY V3 ENGINE
│   │   Earlier iteration of the engine (kept for reference).
│   │   Contains older processing pipeline logic.
│   │
├── v2_onnx.py                    [~8K] LEGACY V2 ONNX ENGINE
│   │   Previous ONNX-only engine version.
│   │
├── export_onnx.py                [76 lines] ONNX EXPORTER
│   │   Converts PyTorch ShieldXception → ONNX format.
│   │   - Dummy input [1,3,299,299], opset 13
│   │   - Dynamic batch axis for flexibility
│   │   - Prerequisite for INT8 quantization
│   │
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL CONFIGURATION & DATA FILES
│   ══════════════════════════════════════════════════════════
│
├── config.yaml                   [45 lines] RUNTIME CONFIGURATION
│   │   Tunable parameters: confidence_threshold (0.89),
│   │   blink_threshold (0.24), input_size (299),
│   │   normalization mean/std, eye landmark indices,
│   │   MediaPipe settings, performance throttle.
│   │
├── shield_calibration.json       [~400B] DEVICE CALIBRATION DATA
│   │   Output of calibrate_device_baseline().
│   │   Contains laplacian_mean, ear_baseline, lighting condition.
│   │
├── requirements.txt              [40 lines] PINNED DEPENDENCIES
│   │   All versions locked with == for reproducible builds.
│   │
├── requirements_locked.txt       [~2.4K] FULL LOCKED DEPS
│   │   Extended dependency list with all transitive deps.
│   │
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL MODEL FILES (neural network weights)
│   ══════════════════════════════════════════════════════════
│
├── ffpp_c23.pth                  [83.5 MB] PYTORCH WEIGHTS
│   │   FaceForensics++ c23 trained XceptionNet weights.
│   │   Used when ONNX model unavailable. SHA-256 verified.
│   │
├── shield_ryzen_int8.onnx        [21.4 MB] INT8 QUANTIZED ONNX
│   │   Primary inference model. 4x smaller than FP32.
│   │   Optimized for AMD Ryzen AI NPU deployment.
│   │
├── shield_ryzen_v2.onnx          [83.2 MB] FP32 ONNX MODEL
│   │   Full-precision ONNX export (pre-quantization).
│   │
├── shield_ryzen_v2.onnx.data     [84.3 MB] ONNX EXTERNAL DATA
│   │   External weight storage for large ONNX model.
│   │
├── face_landmarker.task          [3.75 MB] MEDIAPIPE MODEL (V1)
│   │   MediaPipe face landmark detection model.
│   │
├── face_landmarker_v2_with_blendshapes.task  [3.75 MB] MEDIAPIPE MODEL (V2)
│   │   Enhanced model with 52 blendshape coefficients.
│   │   Used by shield_face_pipeline.py for blink detection.
│   │
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL DOCUMENTATION
│   ══════════════════════════════════════════════════════════
│
├── README.md                     Project overview and quick start
├── GEMINI.md                     AI agent workspace rules (this file)
├── CHANGELOG.md                  Version history and changes
├── LICENSE                       MIT License
├── SECURITY.md                   Security policy
├── CONTRIBUTING.md               Contribution guidelines
├── TRANSFER_GUIDE.md             Guide for porting to new hardware
├── MODEL_CARD.md                 ML model card (dataset, metrics, bias)
├── CLAIMS_VS_EVIDENCE.md         What we claim vs. what we proved
├── DEVELOPMENT_STATE.txt         Current development status
├── VERSION_STAMP.txt             Build version metadata
├── FINAL_AUDIT_RESULT.md         Final project audit score
├── FULL_PROJECT_AUDIT_V2.md      Detailed V2 audit report
├── PROTOTYPE_EXPLAINED.md        ← THIS FILE (complete documentation)
│
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL UTILITY/DIAGNOSTIC SCRIPTS
│   ══════════════════════════════════════════════════════════
│
├── shield.py                     [~4K] Minimal standalone shield runner
├── live_webcam_demo.py           [~3.3K] Simple webcam demo without full engine
├── shield_audio.py               [~2.2K] Audio input module (microphone)
├── shield_gradcam.py             [~3.2K] Grad-CAM attention visualization
├── shield_hardware_monitor.py    [~2.6K] CPU/GPU/RAM monitoring
├── validate_system.py            [~2.2K] System compatibility checker
├── verify_model.py               [~17K] Comprehensive model verification
├── realtime_audit.py             [~5.4K] Live audit trail viewer
├── debug_audit.py                [~1.2K] Quick audit log parser
├── quantize_int8.py              [~10K] INT8 static quantization script
├── quantize_int4.py              [~3.6K] INT4 quantization (experimental)
├── quantize_ryzen.py             [~5.5K] AMD Ryzen-specific quantization
├── compile_xmodel.sh             [~1.7K] Vitis AI xmodel compilation
├── start.bat                     [~800B] Windows batch launcher
├── setup_github.ps1              [~1K] PowerShell GitHub repo setup
│
├── _analyze_audit.py             [~2.8K] Post-session audit analysis
├── _audit_to_file.py             [~4.2K] Audit log → readable report
├── _deep_snap.py                 [~4.5K] Deep analysis single-frame snapshot
├── _diag_faces.py                [~1.5K] Face detection diagnostics
├── _diag_texture.py              [~1.8K] Texture analysis diagnostics
├── _ear_test.py                  [~2.6K] EAR calculation test harness
├── _far_debug.py                 [~1.2K] Far-distance face debug
├── _live_audit.py                [~5.8K] Live auditing with console output
├── _snap.py                      [~1.9K] Quick frame capture utility
│
│   ══════════════════════════════════════════════════════════
│   ROOT-LEVEL LOG/OUTPUT FILES (generated at runtime)
│   ══════════════════════════════════════════════════════════
│
├── crash_log.txt                 Last crash traceback
├── audit_log.txt                 Human-readable audit output
├── last_run_audit.txt            Full audit from most recent session
├── final_test.txt                Test suite results
├── final_test_results.txt        Duplicate of test results
├── run_audit_success.txt         Successful audit run log
├── _audit_output.txt             Audit analysis output
├── _audit_result.txt             Parsed audit results
├── network_during_inference.txt  Netstat proof (no network calls)
├── final_evidence_score.json     Competition evidence score
├── model_verification_report.json  Model integrity report
├── quantization_report.json      INT8 quantization metrics
│
│
│   ══════════════════════════════════════════════════════════
│                     SUBDIRECTORIES
│   ══════════════════════════════════════════════════════════
│
├── plugins/                          ← PLUGIN MODULES (9 detection plugins)
│   ├── __init__.py                   Plugin package init (registers all plugins)
│   ├── rppg_heartbeat.py             [172 lines] rPPG HEARTBEAT DETECTION
│   │   Detects blood flow pulse via green channel FFT.
│   │   - Extracts forehead ROI green channel mean per frame
│   │   - Buffers 5 seconds of data (75 samples at 15fps)
│   │   - FFT bandpass 0.7-3.0 Hz (42-180 BPM)
│   │   - SNR > 2.0 with valid BPM range → REAL
│   │   - Uses actual timestamps for accurate sampling rate
│   │
│   ├── challenge_response.py         [163 lines] SIMON SAYS LIVENESS
│   │   Random action challenges to defeat replay attacks.
│   │   - 4 challenges: blink_twice, look_left, smile, raise_eyebrows
│   │   - 5-second timeout per challenge
│   │   - Uses blendshapes for verification
│   │   - Timeout → FAKE verdict (replay can't respond)
│   │
│   ├── stereo_depth.py               [133 lines] DUAL-CAMERA DEPTH
│   │   Detects flat 2D screens vs real 3D faces.
│   │   - Initializes secondary camera (if available)
│   │   - Compares face detection across two views
│   │   - Real nose protrudes → different disparity than ears
│   │   - Flat screen → uniform disparity
│   │
│   ├── skin_reflectance.py           [119 lines] SKIN TEXTURE ANALYSIS
│   │   Distinguishes organic skin from masks/screens.
│   │   - Cheek ROI analysis (avoids T-zone glare)
│   │   - Specular highlight ratio (>25% = unnatural)
│   │   - Mean Sobel gradient (<1.5 = smooth mask)
│   │   - High gradient (>80) = screen Moiré
│   │
│   ├── codec_forensics.py            [118 lines] COMPRESSION DETECTION
│   │   Detects double-compressed video streams.
│   │   - Analyzes 256×256 center crop aligned to 8×8 grid
│   │   - Computes gradient at block boundaries vs internal
│   │   - Blocking Artifact Ratio (BAR) > 1.8 → FAKE
│   │   - Virtual cameras re-encode → higher BAR
│   │
│   ├── frequency_analyzer.py         [117 lines] FFT GAN DETECTION
│   │   Detects GAN/Diffusion generation artifacts.
│   │   - 2D FFT of face crop grayscale
│   │   - Log-magnitude high-freq/low-freq energy ratio
│   │   - HFER < 0.45 = suppressed HF (GAN smoothness)
│   │   - HFER > 0.90 = screen Moiré HF spike
│   │
│   ├── adversarial_detector.py       [113 lines] PATCH DETECTION
│   │   Detects physical adversarial patches/stickers.
│   │   - Sobel gradient magnitude map of face
│   │   - Threshold high-gradient pixels (>250)
│   │   - Dilate + find contours of dense clusters
│   │   - 2+ patches >5% face area with >50% density → FAKE
│   │
│   ├── lip_sync_verifier.py          [105 lines] LIP READING
│   │   Verifies lip shape matches spoken phonemes.
│   │   - 3 phonemes: "O" (pucker), "Ee" (smile), "Ahh" (open)
│   │   - Uses blendshape indices (36=pucker, 25=jaw, 44/45=smile)
│   │   - 5-second timeout, challenge-response style
│   │
│   └── arcface_reid.py              [149 lines] FACE RE-IDENTIFICATION
│       Enterprise identity matching via ArcFace embeddings.
│       - 512-d embeddings from 112×112 aligned crops
│       - Cosine similarity matching against encrypted DB
│       - AES-256 encrypted employee database (local only)
│       - Mock mode if ArcFace ONNX model not present
│
├── models/                           ← MODEL FILES + ML TOOLS
│   ├── shield_ryzen_int8.onnx        [21.4 MB] INT8 quantized model (primary)
│   ├── shield_xception_int4.onnx     [21.4 MB] INT4 quantized (experimental)
│   ├── blazeface.onnx                [426 KB] BlazeFace detector (unused)
│   ├── face_detector.tflite          [230 KB] TFLite face detector (unused)
│   ├── model_signature.sha256        [64B] SHA-256 hash of model weights
│   ├── reference_output.json         [259B] Expected output for test input
│   ├── temperature_scaling_params.npy [136B] Calibrated temperature value
│   ├── attribution_classifier.py     [~2.5K] Deepfake method attribution
│   └── knowledge_distillation.py     [~4.4K] Teacher→student model compression
│
├── config/                           ← RUNTIME CONFIGURATION
│   └── decision_thresholds.yaml      [680B] Optimized decision thresholds
│       │   Generated by evaluation/threshold_optimization.py
│       │   Contains per-tier threshold overrides
│
├── plugins/                          (described above)
│
├── tests/                            ← TEST SUITE (16 test files)
│   ├── __init__.py                   Test package init
│   ├── test_model.py                 [15K] XceptionNet loading, inference, hash
│   ├── test_utils.py                 [19K] EAR, texture, calibration, state machine
│   ├── test_face_pipeline.py         [17K] Detection, landmarks, normalization
│   ├── test_engine.py                [~8K] Engine init, threading, processing
│   ├── test_camera.py                [~9K] Camera capture, validation, health
│   ├── test_biometric_plugins.py     [~7K] rPPG, challenge-response, depth
│   ├── test_forensic_plugins.py      [~8K] Frequency, codec, adversarial, lip
│   ├── test_hud.py                   [~4K] HUD rendering, color palette
│   ├── test_advanced_detection.py    [~5K] Edge cases, angle compensation
│   ├── test_amd_hardware.py          [~5K] AMD NPU compatibility checks
│   ├── test_amd_npu.py               [~4K] XDNA engine, Vitis AI EP
│   ├── test_enterprise.py            [~5K] ArcFace, encryption, audit
│   ├── test_integration_final.py     [~4K] End-to-end pipeline test
│   ├── test_quantization.py          [~4K] INT8/INT4 accuracy verification
│   ├── skipped_integration_full.py   [~9K] Skipped heavy integration tests
│   ├── fixtures/                     Test fixture data (synthetic faces)
│   └── temp_final/                   Temporary test outputs
│
├── scripts/                          ← BUILD + MAINTENANCE SCRIPTS
│   ├── emergency_diagnostic.py       [~40K] Comprehensive system diagnostics
│   ├── export_verified_onnx.py       [~2.7K] Verified ONNX export pipeline
│   ├── generate_calibration_v2.py    [~9K] Multi-frame calibration generator
│   ├── generate_evidence_package.ps1 [~1.9K] Evidence package PowerShell script
│   ├── generate_test_fixtures.py     [~5.8K] Synthetic test data generator
│   ├── inspect_blazeface.py          [~420B] BlazeFace model inspector
│   ├── setup_blazeface.py            [~2.2K] BlazeFace setup utility
│   ├── verify_integrity.py           [~1.6K] File integrity checker
│   └── verify_normalization.py       [~10K] Normalization correctness validator
│
├── benchmarks/                       ← PERFORMANCE BENCHMARKS
│   ├── baseline.py                   [~16K] Comprehensive baseline benchmark
│   ├── benchmark_accuracy.py         [~5K] Accuracy on test datasets
│   ├── benchmark_fps.py              [~6K] FPS measurement across configs
│   ├── benchmark_plugins.py          [~4K] Per-plugin latency measurement
│   ├── benchmark_power.py            [~2.2K] Power consumption estimation
│   ├── benchmark_threshold.py        [~3K] Threshold sensitivity analysis
│   ├── benchmark_blazeface.py        [~1.4K] BlazeFace speed test
│   ├── edge_case_test_suite.py       [~6K] Edge case stress testing
│   ├── final_performance_report.py   [~3.5K] Aggregated report generator
│   ├── run_netstat_proof.py          [~1.6K] Network isolation proof
│   ├── accuracy_report.json          Accuracy benchmark results
│   ├── fps_report.json               FPS benchmark results
│   ├── power_report.json             Power benchmark results
│   ├── threshold_report.json         Threshold analysis results
│   └── roc_curves/                   ROC curve data and plots
│
├── security/                         ← SECURITY TESTING
│   ├── adversarial_test_suite.py     [~14K] Adversarial attack simulations
│   ├── audit_trail.py                [~4.6K] Audit trail integrity checker
│   ├── ftpm_wrapper.py               [~2.1K] TPM key storage wrapper
│   ├── verify_integrity.py           [~3.6K] Full integrity verification
│   ├── adversarial_robustness.json   Robustness test results (detailed)
│   └── adversarial_robustness_report.json  Summary robustness report
│
├── evaluation/                       ← ML EVALUATION
│   ├── auc_validation.py             [~12K] AUC/ROC curve validation
│   └── threshold_optimization.py     [~12K] Optimal threshold search
│
├── evidence_package/                 ← COMPETITION EVIDENCE (AMD Slingshot)
│   ├── FINAL_AUDIT_RESULT.md         Final audit score document
│   ├── accuracy_report.json          Accuracy evidence
│   ├── adversarial_robustness_report.json  Security evidence
│   ├── final_evidence_score.json     Combined evidence score
│   ├── final_test_results.txt        Test pass/fail evidence
│   ├── fps_report.json               Performance evidence
│   ├── model_signature.sha256        Model integrity evidence
│   ├── model_verification_report.json  Model verification evidence
│   ├── network_during_inference.txt  Privacy evidence (no network)
│   ├── power_report.json             Power efficiency evidence
│   ├── quantization_report.json      Quantization evidence
│   ├── requirements_locked.txt       Dependency reproducibility evidence
│   ├── test_results.txt              Unit test evidence
│   ├── threshold_report.json         Threshold calibration evidence
│   └── roc_curves/                   ROC curve evidence plots
│
├── docs/                             ← DOCUMENTATION
│   ├── architecture.md               [~4K] System architecture reference
│   ├── DEVELOPMENT_LOG.md            [~23K] Full development timeline
│   ├── PROTOTYPE_EXPLAINED.md        [~25K] Deep-dive documentation
│   ├── PROTOTYPE_AUDIT_REPORT.md     [~2K] Audit report
│   ├── COMPLIANCE.md                 [~3K] GDPR/CCPA compliance guide
│   ├── THREAT_MODEL.md               [~3K] Security threat analysis
│   └── VALIDATION_REPORT.md          [~2K] System validation results
│
├── shield_utils/                     ← UTILITY SUBPACKAGE
│   ├── __init__.py                   [~2K] Package init (re-exports utilities)
│   ├── blazeface_detector.py         [~6.5K] BlazeFace ONNX detector
│   ├── calibrated_decision.py        [~1.6K] Calibrated decision helper
│   └── occlusion_detector.py         [~6K] Face occlusion analysis
│
├── performance/                      ← PERFORMANCE OPTIMIZATION
│   ├── preprocessing_worker.py       [~3.9K] Background preprocessing thread
│   └── zero_copy_buffer.py           [~1.3K] Zero-copy frame buffer
│
├── shield_audio_module/              ← AUDIO PROCESSING
│   └── lip_sync_verifier.py          [~1.8K] Audio-visual lip sync
│
├── shield_liveness/                  ← LIVENESS DETECTION
│   └── challenge_response.py         [~3K] Earlier challenge-response version
│
├── shield_temporal/                  ← TEMPORAL ANALYSIS
│   └── temporal_consistency.py       [~4.4K] Frame-to-frame consistency
│
├── shield_frequency/                 ← FREQUENCY ANALYSIS
│   └── frequency_analyzer.py         [~3.8K] Earlier frequency analyzer version
│
├── camera/                           ← CAMERA UTILITIES
│   └── amd_direct_capture.py         [~2.1K] AMD DirectCapture API wrapper
│
├── logs/                             ← RUNTIME LOGS (generated)
│   ├── shield_audit.jsonl            JSONL audit trail (per-frame)
│   └── texture_debug.log             Texture analysis debug output
│
├── PRESENTATION DATA/                ← COMPETITION PRESENTATION ASSETS
│   ├── IMAGES/                       Presentation screenshots/diagrams
│   ├── MINDMAP/                      Architecture mind maps
│   ├── PODCAST/                      Audio presentation files
│   ├── PPT/                          PowerPoint slides
│   └── VIDEOS/                       Demo video recordings
│
├── data/                             ← TRAINING/TEST DATA (empty/gitignored)
├── diagnostics/                      ← DIAGNOSTIC OUTPUT (empty/generated)
│
├── .git/                             Git repository
├── .gitignore                        Git ignore rules
├── .venv/                            Python virtual environment
└── __pycache__/                      Python bytecode cache
```

## 19.1 File Count Summary

| Category | Files | Total Size |
|---|---|---|
| **Core Engine** (root .py) | 14 files | ~190 KB |
| **Plugins** | 9 files + __init__ | ~48 KB |
| **Models** (weights) | 6 model files | ~213 MB |
| **Tests** | 16 test files | ~120 KB |
| **Scripts** | 9 utility scripts | ~74 KB |
| **Benchmarks** | 10 scripts + 5 reports | ~52 KB |
| **Security** | 4 scripts + 2 reports | ~25 KB |
| **Evaluation** | 2 scripts | ~24 KB |
| **Documentation** | 15 markdown files | ~80 KB |
| **Evidence Package** | 15 evidence files | ~70 KB |
| **Subpackages** | 8 files across 6 packages | ~25 KB |
| **Config** | 3 YAML/JSON files | ~3.4 KB |
| **TOTAL** | ~100+ source files | ~213 MB (models dominate) |

## 19.2 Which Files Actually Run During Inference?

When you execute `python start_shield.py --cpu`, these files are loaded:

```
STARTUP CHAIN:
  start_shield.py
    → v3_xdna_engine.py (or shield_engine.py directly with --cpu)
      → shield_engine.py ← MAIN ORCHESTRATOR
        → shield_camera.py
        → shield_face_pipeline.py
        → shield_xception.py (model loading)
        → shield_utils_core.py (all algorithms)
        → shield_crypto.py
        → shield_logger.py
        → shield_types.py (dataclasses)
        → config.yaml (configuration)
        → plugins/__init__.py
          → plugins/rppg_heartbeat.py
          → plugins/challenge_response.py
          → plugins/stereo_depth.py
          → plugins/skin_reflectance.py
          → plugins/codec_forensics.py
          → plugins/frequency_analyzer.py
          → plugins/adversarial_detector.py
          → plugins/lip_sync_verifier.py
          → plugins/arcface_reid.py
      → shield_hud.py ← HUD RENDERING

MODEL FILES LOADED:
  → face_landmarker_v2_with_blendshapes.task (MediaPipe)
  → shield_ryzen_int8.onnx (or ffpp_c23.pth fallback)

RUNTIME FILES CREATED/UPDATED:
  → logs/shield_audit.jsonl (audit trail)
  → logs/texture_debug.log (debug output)
  → shield_calibration.json (if first run)
```

**Files NOT loaded during inference:** All test files, benchmarks, scripts, evaluation, security tests, evidence package, documentation, legacy engines, and presentation data. These are development/competition support files only.

---

*Documentation generated for AMD Slingshot 2026 competition.*
*Developer: Inayat Hussain*
*Shield-Ryzen V2 — UPDATE 1*
*Last updated: 2026-02-22*
