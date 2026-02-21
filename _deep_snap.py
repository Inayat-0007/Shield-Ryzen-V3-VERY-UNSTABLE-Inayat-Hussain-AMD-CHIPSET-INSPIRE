"""Deep audit snapshot — extracts ALL fields for debugging. FIXED key names."""
import json, os

audit = os.path.join(os.path.dirname(__file__), "logs", "shield_audit.jsonl")
out_path = os.path.join(os.path.dirname(__file__), "logs", "deep_snapshot.txt")

with open(audit, 'r', encoding='utf-8', errors='replace') as af:
    lines = af.readlines()

count = 0
stats = {"states": {}, "ear_values": [], "blink_counts": [], "textures": [], "fps_list": []}

with open(out_path, 'w', encoding='utf-8') as out:
    out.write("=" * 120 + "\n")
    out.write("SHIELD-RYZEN DEEP AUDIT SNAPSHOT\n")
    out.write("=" * 120 + "\n\n")
    
    for l in lines:
        try:
            d = json.loads(l.strip())
            data = d.get('data', {}) or {}
            fps = data.get('fps', 0) or 0
            timing = data.get('timing', {}) or {}
            mem = data.get('memory_mb', 0) or 0
            faces = data.get('face_results', []) or []
            
            if fps:
                stats["fps_list"].append(fps)
            
            for f in faces:
                count += 1
                adv = f.get('advanced_info', {}) or {}
                
                state = f.get('state', '?')
                conf = f.get('confidence', 0) or 0
                tex = f.get('texture_score', 0) or 0
                sr = f.get('screen_replay', False) or False
                # FIXED: key is 'ear_value' not 'ear'
                ear = adv.get('ear_value', 0) or 0
                blinks = adv.get('blinks', 0) or 0
                dist = adv.get('distance_cm', 0) or 0
                alert = f.get('face_alert', '') or ''
                tex_exp = adv.get('texture_explain', '') or ''
                fid = f.get('face_id', f.get('tracker_id', '?'))
                ear_rel = adv.get('ear_reliability', '?')
                tiers = adv.get('tier_verdicts', {}) or {}
                plugins = adv.get('plugins', []) or []
                hp = adv.get('head_pose', {}) or {}
                yaw = hp.get('yaw', 0) if isinstance(hp, dict) else 0
                pitch = hp.get('pitch', 0) if isinstance(hp, dict) else 0
                age = adv.get('face_age_s', f.get('face_age_s', 0)) or 0
                blink_src = adv.get('blink_source', '?')
                
                # Track stats
                stats["states"][state] = stats["states"].get(state, 0) + 1
                stats["ear_values"].append(ear)
                stats["blink_counts"].append(blinks)
                stats["textures"].append(tex)
                
                # Write detailed line
                out.write(f"#{count:5d} | FID:{fid:<3} | {state:12s} | Conf:{conf:.1%} | "
                          f"EAR:{ear:.3f} ({ear_rel}) | Bk:{blinks:<3} src:{blink_src} | "
                          f"D:{dist:>3.0f}cm | Tex:{tex:>7.1f} | SR:{sr}\n")
                out.write(f"        Tiers: {tiers} | Y:{yaw:+.1f} P:{pitch:+.1f} | Age:{age:.0f}s\n")
                if plugins:
                    plugin_str = " | ".join([f"{p.get('name','?')}:{p.get('verdict','?')}" for p in plugins[:5]])
                    out.write(f"        Plugins: {plugin_str}\n")
                if tex_exp:
                    out.write(f"        TexExplain: {tex_exp}\n")
                if alert:
                    out.write(f"        ALERT: {alert}\n")
                out.write("\n")
        except Exception as e:
            pass

    # Summary
    out.write("\n" + "=" * 120 + "\n")
    out.write("SUMMARY STATISTICS\n")
    out.write("=" * 120 + "\n")
    out.write(f"Total frames analyzed: {count}\n")
    out.write(f"State distribution: {stats['states']}\n")
    if stats["fps_list"]:
        out.write(f"FPS: min={min(stats['fps_list']):.1f} max={max(stats['fps_list']):.1f} avg={sum(stats['fps_list'])/len(stats['fps_list']):.1f}\n")
    if stats["ear_values"]:
        out.write(f"EAR: min={min(stats['ear_values']):.4f} max={max(stats['ear_values']):.4f} avg={sum(stats['ear_values'])/len(stats['ear_values']):.4f}\n")
    if stats["blink_counts"]:
        out.write(f"Blinks: min={min(stats['blink_counts'])} max={max(stats['blink_counts'])}\n")
    if stats["textures"]:
        out.write(f"Texture: min={min(stats['textures']):.1f} max={max(stats['textures']):.1f} avg={sum(stats['textures'])/len(stats['textures']):.1f}\n")

print(f"Deep snapshot done — {count} entries")
