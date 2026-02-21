import json

lines = open('logs/shield_audit.jsonl').readlines()
face_stats = {}
for l in lines:
    try:
        d = json.loads(l.strip())
        data = d.get('data', {})
        fr = data.get('face_results', [])
        for f in fr:
            fid = f.get('face_id', -1)
            s = f.get('state', '?')
            c = f.get('confidence', 0)
            if fid >= 0:
                if fid not in face_stats:
                    face_stats[fid] = {'total': 0, 'fake': 0, 'real': 0, 'verified': 0, 'suspicious': 0, 'confs': []}
                face_stats[fid]['total'] += 1
                face_stats[fid]['confs'].append(c)
                if s == 'FAKE': face_stats[fid]['fake'] += 1
                elif s == 'REAL': face_stats[fid]['real'] += 1
                elif s == 'VERIFIED': face_stats[fid]['verified'] += 1
                elif s == 'SUSPICIOUS': face_stats[fid]['suspicious'] += 1
    except:
        pass

for fid, stats in sorted(face_stats.items()):
    confs = stats['confs']
    avg_c = sum(confs)/len(confs) if confs else 0
    min_c = min(confs) if confs else 0
    max_c = max(confs) if confs else 0
    fake_pct = (stats['fake']/stats['total']*100) if stats['total'] > 0 else 0
    print(f"Face {fid}: total={stats['total']:5d} FAKE={stats['fake']:4d}({fake_pct:.0f}%) REAL={stats['real']:4d} VERIFIED={stats['verified']:4d} SUSP={stats['suspicious']:4d} | conf avg={avg_c:.3f} [{min_c:.3f}-{max_c:.3f}]")
