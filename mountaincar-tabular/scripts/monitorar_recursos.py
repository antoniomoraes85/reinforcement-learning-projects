"""Sample only this project's Python campaign tree; no persistent automation."""
from pathlib import Path
import json,time
import psutil
ROOT=Path(__file__).resolve().parents[1]
rows=[]
while True:
    roots=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if 'python' in p.info['name'].lower() and any('scripts/executar_final.py' in s or 'scripts\\executar_final.py' in s for s in p.info['cmdline'] or []):
                roots.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if not roots:break
    processes={p.pid:p for root in roots for p in [root,*root.children(recursive=True)]}
    rss=cpu=0
    for p in processes.values():
        try:
            rss+=p.memory_info().rss; t=p.cpu_times(); cpu+=t.user+t.system
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    rows.append({'utc_epoch':time.time(),'rss_sum_bytes':rss,'cpu_seconds':cpu,'processes':len(processes)})
    time.sleep(5)
target=ROOT/'relatorios/recursos_campanha.json'
target.write_text(json.dumps({'sampling_seconds':5,'rss_note':'sum RSS; may double-count shared pages; monitoring begins after launch',
                             'peak_rss_sum_bytes':max((r['rss_sum_bytes'] for r in rows),default=0),'samples':rows},indent=2),encoding='utf-8')
print(f'{len(rows)} resource samples saved')
