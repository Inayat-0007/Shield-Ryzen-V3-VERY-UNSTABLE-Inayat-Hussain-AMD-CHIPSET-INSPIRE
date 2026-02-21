import json, os

audit = os.path.join(os.path.dirname(__file__), "logs", "shield_audit.jsonl")
out_path = os.path.join(os.path.dirname(__file__), "logs", "live_snapshot.txt")

with open(audit, 'r', encoding='utf-8', errors='replace') as af:
    lines = af.readlines()

count = 0
with open(out_path, 'w', encoding='utf-8') as out:
    for l in lines[-80:]:
        try:
            d = json.loads(l.strip())
            data = d.get('data', {}) or {}
            faces = data.get('face_results', []) or []
            for f in faces:
                count += 1
                adv = f.get('advanced_info', {}) or {}
                state = f.get('state', '?')
                conf = f.get('confidence', 0) or 0
                tex = f.get('texture_score', 0) or 0
                sr = f.get('screen_replay', False) or False
                # FIX: Key is 'ear_value' not 'ear'
                ear = adv.get('ear_value', 0) or 0
                blinks = adv.get('blinks', 0) or 0
                dist = adv.get('distance_cm', 0) or 0
                alert = f.get('face_alert', '') or ''
                tex_exp = adv.get('texture_explain', '') or ''
                fid = f.get('face_id', f.get('tracker_id', '?'))
                age = adv.get('face_age_s', f.get('face_age_s', 0)) or 0
                ear_rel = adv.get('ear_reliability', '?')
                hp = adv.get('head_pose', {}) or {}
                yaw = hp.get('yaw', 0) if isinstance(hp, dict) else 0
                pitch = hp.get('pitch', 0) if isinstance(hp, dict) else 0
                out.write(f"{count:4d} | {state:12s} | Conf:{conf:.1%} | EAR:{ear:.3f} ({ear_rel}) | Bk:{blinks:<3} | D:{dist:>3.0f}cm | Tex:{tex:>7.1f} | Y:{yaw:+.1f} P:{pitch:+.1f} | SR:{sr} | {alert}\n")
                if tex_exp:
                    out.write(f"     TEX: {tex_exp}\n")
        except:
            pass

print(f"Done - {count} entries written")
