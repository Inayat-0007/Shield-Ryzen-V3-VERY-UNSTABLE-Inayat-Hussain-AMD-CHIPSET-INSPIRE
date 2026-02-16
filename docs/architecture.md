# 🛡️ Shield-Xception Architecture Reference

## Overview
Shield-Xception is a **real-time deepfake detection system** built for the **AMD Slingshot 2026** competition. It uses an XceptionNet backbone trained on FaceForensics++ (c23 compression) to classify faces as Real or Fake via a live webcam feed.

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam    │────▶│  MediaPipe   │────▶│  XceptionNet │────▶│  Trust UI   │
│  (OpenCV)   │     │  Face Detect │     │  (ffpp_c23)  │     │  Overlay    │
│  30 FPS     │     │  299x299 crop│     │  Sigmoid Out │     │  Real/Fake  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
       │                                        │
       │                  CUDA                  │
       └──────────── RTX 3050 GPU ──────────────┘
```

## Data Flow (Per Frame)

1. **Capture:** OpenCV reads BGR frame from webcam (`cv2.VideoCapture(0)`)
2. **Detection:** Frame converted to RGB → MediaPipe extracts face bounding boxes
3. **Crop:** Each face cropped from original BGR frame using bbox coordinates
4. **Transform:** Crop → PIL Image → Resize 299×299 → Tensor → Normalize [0.5, 0.5, 0.5]
5. **Inference:** Tensor → CUDA → `ShieldXception.forward()` → Sigmoid → raw_score
6. **Trust Score:** `trust_score = 1 - raw_score` (0.0=Fake, 1.0=Real)
7. **Display:** Bounding box + label overlay on original frame → `cv2.imshow()`

## ShieldXception Model Architecture

```python
ShieldXception(nn.Module)
├── self.model = timm.create_model('xception', pretrained=False, num_classes=1)
│   ├── Entry Flow (3 conv blocks with separable convolutions)
│   ├── Middle Flow (8 repeated blocks)
│   ├── Exit Flow (2 blocks + global average pooling)
│   └── FC Head → 1 output neuron
└── self.sigmoid = nn.Sigmoid()  # Squash to [0, 1] range
```

- **Input:** `[B, 3, 299, 299]` — Batch of RGB face crops
- **Output:** `[B, 1]` — Probability (1.0 = Fake, 0.0 = Real)

## Weight Loading Strategy

The `ffpp_c23.pth` weights may come in different formats:
- **Wrapped in dict:** `state_dict['model']` is extracted
- **DataParallel prefix:** `module.` prefix is stripped from all keys
- **Loaded with `strict=False`:** Allows partial loading if architecture differs slightly

## Key Constants

| Parameter               | Value                         |
|-------------------------|-------------------------------|
| Input Resolution        | 299 × 299 px                  |
| Normalization Mean      | [0.5, 0.5, 0.5]              |
| Normalization Std       | [0.5, 0.5, 0.5]              |
| Face Detection Model    | MediaPipe model_selection=0   |
| Detection Confidence    | 0.5                           |
| Trust Threshold         | 0.5                           |
| Escape Key              | ESC (keycode 27)              |

## Development Roadmap

| Level | Task                              | Status       |
|-------|-----------------------------------|--------------|
| 1.0   | Core XceptionNet + webcam loop    | ✅ Complete   |
| 1.5   | FPS optimization (face loop)      | 🔜 Next      |
| 2.0   | ONNX export for AMD Ryzen AI NPU  | 📋 Planned   |
| 2.5   | Transparent overlay UI            | 📋 Planned   |
| 3.0   | Multi-face + temporal analysis    | 📋 Planned   |
| 3.5   | AMD Ryzen AI NPU deployment       | 📋 Planned   |

## ONNX Compatibility Notes (for Level 2)

When exporting to ONNX, ensure:
- Use `torch.onnx.export()` with `dynamic_axes` for variable batch size
- Verify all ops in XceptionNet are ONNX-compatible
- Target opset version 17+ for best AMD compatibility
- Test with `onnxruntime` before `onnxruntime-directml` (AMD)

## File Reference

| File                | Purpose                                      |
|---------------------|----------------------------------------------|
| `shield_xception.py`| Core engine — real-time deepfake detection   |
| `ffpp_c23.pth`      | Pre-trained Xception weights (FF++ c23)      |
| `GEMINI.md`         | Agent workspace rules & guardrails           |
| `docs/architecture.md` | This file — architecture reference        |
