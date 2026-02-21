"""Quick EAR diagnostic — tests compute_ear with realistic eye landmarks."""
import math
from shield_utils_core import compute_ear, LEFT_EYE, RIGHT_EYE
import numpy as np

print("=" * 60)
print("EAR DIAGNOSTIC — Testing compute_ear()")
print("=" * 60)
print(f"LEFT_EYE indices:  {LEFT_EYE}")
print(f"RIGHT_EYE indices: {RIGHT_EYE}")

# Create a fake landmarks array with 478 points (all zeros)
landmarks = np.zeros((478, 2), dtype=np.float32)

# Simulate a REAL left eye at reasonable pixel coords
# idx: 33=outer, 160=upper1, 158=upper2, 133=inner, 153=lower2, 144=lower1
# Normal open eye — vertical ~10px, horizontal ~30px
landmarks[33]  = [200.0, 200.0]  # outer corner
landmarks[160] = [210.0, 192.0]  # upper 1
landmarks[158] = [220.0, 191.0]  # upper 2
landmarks[133] = [230.0, 200.0]  # inner corner  
landmarks[153] = [220.0, 208.0]  # lower 2
landmarks[144] = [210.0, 209.0]  # lower 1

ear_l, rel_l = compute_ear(landmarks, LEFT_EYE, (0.0, 0.0, 0.0), True)
print(f"\nTest 1 - Synthetic open eye: EAR = {ear_l:.4f}, rel = {rel_l}")
print(f"  Expected: ~0.25-0.35 (open eye)")

# Simulate closed eye
landmarks[160] = [210.0, 199.0]
landmarks[158] = [220.0, 199.0]
landmarks[153] = [220.0, 201.0]
landmarks[144] = [210.0, 201.0]

ear_l_closed, rel_closed = compute_ear(landmarks, LEFT_EYE, (0.0, 0.0, 0.0), True)
print(f"\nTest 2 - Synthetic closed eye: EAR = {ear_l_closed:.4f}, rel = {rel_closed}")
print(f"  Expected: ~0.05-0.15 (closed eye)")

# Now test with REAL head_pose = (0,0,0) — which is what the audit shows
print(f"\nTest 3 - Zero head pose: (0.0, 0.0, 0.0)")
print(f"  This is what the audit shows for ALL frames.")
print(f"  If head_pose is always (0,0,0), the solvePnP in face_pipeline may be failing silently.")

# Check if compute_ear returns 0.000 for any reason
landmarks[33]  = [200.0, 200.0]
landmarks[160] = [200.0, 200.0]  # ALL same point
landmarks[158] = [200.0, 200.0]
landmarks[133] = [200.0, 200.0]
landmarks[153] = [200.0, 200.0]
landmarks[144] = [200.0, 200.0]

ear_zero, rel_zero = compute_ear(landmarks, LEFT_EYE, (0.0, 0.0, 0.0), True)
print(f"\nTest 4 - All points collapsed (h_dist=0): EAR = {ear_zero:.4f}, rel = {rel_zero}")
print(f"  Expected: 0.05 (fallback for h_dist < 1e-6)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("EAR=0.000 in the audit means the actual EAR computed is")
print("between 0.0000 and 0.0004 (rounds to 0.000).")
print("Since blink_source=BLENDSHAPES, blendshape blinks bypass EAR.")
print("The audit shows EAR but the BLENDSHAPE path returns baseline=0.0")
print("=" * 60)
