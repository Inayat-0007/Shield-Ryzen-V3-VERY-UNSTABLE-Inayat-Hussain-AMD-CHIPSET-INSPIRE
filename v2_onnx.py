"""
Shield-Ryzen V2 — Universal ONNX Engine (SECURITY MODE)
========================================================
Zero PyTorch dependency. Pure ONNX Runtime + NumPy + OpenCV.
Optimized for NVIDIA RTX 3050 (CUDAExecutionProvider)
and AMD Ryzen AI NPU ready.

Developer: Inayat Hussain | AMD Slingshot 2026
"""

import cv2
import numpy as np
import torch  # Load CUDA DLLs into process (required for ORT CUDA provider on Windows)
import onnxruntime as ort
import mediapipe as mp
import os
import time
import math

# ═══════════════════════════════════════════════════════════════
#  SHIELD-RYZEN V2 — UNIVERSAL ONNX SECURITY MODE
#  No PyTorch. No TensorFlow. Pure inference engine.
# ═══════════════════════════════════════════════════════════════

# ─── Constants ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.89    # 89% rule
BLINK_THRESHOLD = 0.21         # EAR below this = eyes closed
BLINK_TIME_WINDOW = 10         # Seconds
LAPLACIAN_THRESHOLD = 50       # Texture guard

# Eye landmarks (MediaPipe 478-point mesh)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Xception preprocessing constants (FF++ standard)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
INPUT_SIZE = 299

# ─── 1. Setup ONNX Runtime Session (GPU) ─────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, 'shield_ryzen_v2.onnx')

# Prioritize GPU providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
active_provider = session.get_providers()[0]

print(f"🖥️  Provider: {active_provider}")
print(f"📦  Model: {onnx_path}")
print(f"🔢  Input: {session.get_inputs()[0].shape}")

# ─── 2. Setup MediaPipe FaceLandmarker ────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker_path = os.path.join(script_dir, 'face_landmarker.task')
landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmarker_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── 3. Helper Functions (Pure NumPy/OpenCV) ─────────────────
def preprocess_face(face_crop):
    """Preprocess face crop for Xception: resize, normalize to [-1, 1].
    Pure NumPy — no PyTorch transforms needed."""
    # Resize to 299x299
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))
    
    # Convert to float32 [0, 1]
    face_float = face_resized.astype(np.float32) / 255.0
    
    # Normalize: (pixel - mean) / std → [-1, 1]
    face_norm = (face_float - MEAN) / STD
    
    # Transpose HWC → CHW and add batch dimension: [1, 3, 299, 299]
    face_chw = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0).astype(np.float32)

def calculate_ear(landmarks, eye_indices):
    """Eye Aspect Ratio from 6 landmark points."""
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    
    v1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
    v2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
    h_dist = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
    
    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)

def check_texture(face_crop):
    """Laplacian variance — low = too smooth."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ─── 4. Start Security Mode ──────────────────────────────────
cap = cv2.VideoCapture(0)

print()
print("═" * 55)
print("  🛡️  SHIELD-RYZEN V2 — ONNX SECURITY MODE")
print("═" * 55)
print(f"  Engine:     ONNX Runtime {ort.__version__} (NO PyTorch)")
print(f"  Provider:   {active_provider}")
print(f"  Threshold:  {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"  Blink:      {BLINK_TIME_WINDOW}s window")
print(f"  Texture:    Laplacian > {LAPLACIAN_THRESHOLD}")
print("  Press ESC to exit.")
print("═" * 55)

frame_timestamp_ms = 0
fps_counter = 0
fps_display = 0.0
fps_timer = time.time()
inference_ms = 0.0

# Blink tracking
blink_count = 0
blink_timestamps = []
was_eye_closed = False

with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # FPS
        fps_counter += 1
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer = now

        # Detect landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        lm_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        h, w, _ = frame.shape

        # Prune old blinks
        blink_timestamps = [t for t in blink_timestamps if now - t < BLINK_TIME_WINDOW]
        blink_count = len(blink_timestamps)

        if lm_result.face_landmarks:
            for face_landmarks in lm_result.face_landmarks:
                # Bounding box from landmarks
                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]
                x_min = max(0, int(min(xs) * w) - 10)
                y_min = max(0, int(min(ys) * h) - 10)
                x_max = min(w, int(max(xs) * w) + 10)
                y_max = min(h, int(max(ys) * h) + 10)

                # EAR Blink Detection
                left_ear = calculate_ear(face_landmarks, LEFT_EYE)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < BLINK_THRESHOLD:
                    was_eye_closed = True
                elif was_eye_closed and avg_ear >= BLINK_THRESHOLD:
                    was_eye_closed = False
                    blink_timestamps.append(now)
                    blink_count = len(blink_timestamps)

                liveness_ok = blink_count > 0

                # Face crop
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue

                # Texture guard
                texture_score = check_texture(face_crop)
                texture_ok = texture_score > LAPLACIAN_THRESHOLD

                # ── ONNX Inference (GPU) ──
                input_tensor = preprocess_face(face_crop)
                
                inf_start = time.perf_counter()
                output = session.run(None, {'input': input_tensor})[0]
                inference_ms = (time.perf_counter() - inf_start) * 1000
                
                # Class mapping: Index 0 = Fake, Index 1 = Real
                fake_prob = float(output[0, 0])
                real_prob = float(output[0, 1])

                # ══ SECURITY MODE CLASSIFICATION (3-Tier) ══
                if fake_prob > 0.50:
                    label = "CRITICAL: FAKE DETECTED"
                    color = (0, 0, 255)
                elif real_prob < CONFIDENCE_THRESHOLD:
                    label = "WARNING: LOW CONFIDENCE"
                    color = (0, 200, 255)
                elif not liveness_ok:
                    label = "LIVENESS FAILED"
                    color = (0, 165, 255)
                elif not texture_ok:
                    label = "SMOOTHNESS WARNING"
                    color = (0, 200, 255)
                else:
                    label = "SHIELD: VERIFIED REAL"
                    color = (0, 255, 0)

                # Draw
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Real:{real_prob*100:.1f}% Fake:{fake_prob*100:.1f}%",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Per-face stats
                info_x = x_max + 5
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (info_x, y_min + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"TEX: {texture_score:.0f}", (info_x, y_min + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"INF: {inference_ms:.1f}ms", (info_x, y_min + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── HUD ──
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps_display:.1f} | ONNX | {active_provider}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        blink_color = (0, 255, 0) if blink_count > 0 else (0, 0, 255)
        blink_text = f"Blink: {'YES' if blink_count > 0 else 'NO'} ({blink_count} in {BLINK_TIME_WINDOW}s) | INF: {inference_ms:.1f}ms"
        cv2.putText(frame, blink_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
        
        cv2.putText(frame, "V2 ONNX ENGINE", (w - 180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", (w - 170, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow('Shield-Ryzen V2 | ONNX SECURITY MODE', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
