"""
Diagnostic: Compute detailed texture breakdown for last audit frames.
Shows exact screen_light, moire, physics scores so we can tune thresholds.
"""
import json
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shield_utils_core import _detect_screen_light, _detect_moire_pattern, _compute_hf_energy_ratio

# Read audit log to get current state
audit_path = os.path.join(os.path.dirname(__file__), "logs", "shield_audit.jsonl")
if not os.path.exists(audit_path):
    print("No audit log found")
    sys.exit(1)

with open(audit_path, 'r') as f:
    lines = f.readlines()

print(f"Total audit entries: {len(lines)}")
print("=" * 100)

# Show last 5 entries with full detail
for line in lines[-5:]:
    try:
        d = json.loads(line.strip())
        faces = d.get("faces", [])
        for fi, face in enumerate(faces):
            adv = face.get("advanced_info", {})
            state = face.get("state", "?")
            
            # Get all metrics
            conf = adv.get("confidence", face.get("confidence", 0))
            ear = adv.get("ear", 0)
            blinks = adv.get("blinks", 0)
            dist = adv.get("distance_cm", 0)
            tex = adv.get("texture_score", face.get("texture_score", 0))
            tex_explain = adv.get("texture_explain", "")
            sr = adv.get("screen_replay", False)
            
            print(f"STATE: {state:12s} | Conf: {conf:.3f} | Tex: {tex:6.1f} | Dist: {dist:.0f}cm | EAR: {ear:.3f} | Blinks: {blinks} | SR: {sr}")
            if tex_explain:
                print(f"  Texture: {tex_explain[:120]}")
            print()
    except Exception as e:
        pass

print("\nDiagnostic complete.")
