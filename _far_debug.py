"""Find far-distance false positives with texture explanations."""
import json

lines = open("logs/shield_audit.jsonl", encoding='utf-8').readlines()
count = 0
results = []
for l in lines:
    try:
        d = json.loads(l.strip())
        if d.get("event") != "frame_processed":
            continue
        faces = d["data"].get("face_results", [])
        if not faces:
            continue
        f = faces[0]
        adv = f.get("advanced_info", {})
        dist = adv.get("distance_cm", 0)
        state = f.get("state", "?")
        tex = f.get("texture_score", 0)
        explain = f.get("texture_explanation", "")
        sr = adv.get("screen_replay", False)
        
        if dist > 80 and state in ("FAKE", "SUSPICIOUS"):
            count += 1
            results.append(f"#{count:3d} D={dist:.0f}cm | {state:12s} | tex={tex:.0f} | SR={sr}")
            results.append(f"     {explain[:200]}")
            results.append("")
            if count >= 20:
                break
    except:
        pass

results.append(f"\nTotal far-distance false positives shown: {count}")

with open("logs/far_debug_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("Done")
