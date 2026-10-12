"""Prepare new protocol and short smoke only; never runs final campaign."""
from pathlib import Path
import sys
import copy
import time
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json,expand_final
from mountaincar_rl.runner import run_batch,setup_logging
from mountaincar_rl.storage import atomic_write_json
from mountaincar_rl.experiment import library_versions


def main():
    protocol=read_json(ROOT/'configs/protocolo_final.json')
    protocol.update(experiment_id='cmp629-final-v2-20261011', experiment_version='mountaincar-final-v2.0', bootstrap_time_limit=True)
    protocol['evaluation']={'episodes':100,'reset_seeds':list(range(800000,800100))}
    protocol['demo_reset_seed']=900000  # fixed before watching any rollout; distinct from final evaluation
    dest=ROOT/'configs/protocolo_final_v2.json'
    if dest.exists():
        assert read_json(dest)==protocol
    else: atomic_write_json(dest,protocol)
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')
    runs=expand_final(protocol,selection)
    smoke=[]
    for c in runs:
        if c['seed']==protocol['seeds'][0]:
            c=copy.deepcopy(c)
            c.update(phase='audit',experiment_id='cmp629-audit-smoke-v2',experiment_version='audit-smoke-v2')
            c['seed']=700001
            c['training']['episodes']=5
            c['evaluation']={'episodes':2,'reset_seeds':[700101,700102]}
            smoke.append(c)
    output=ROOT/'resultados/auditoria_v2'
    setup_logging(output,'audit-smoke')
    started=time.perf_counter()
    receipts=run_batch(smoke,output,2,'audit-smoke')
    elapsed=time.perf_counter()-started
    results=[read_json(item['path']) for item in receipts]
    durations=[r['training_metrics']['training_time_seconds'] for r in results]
    sizes=sum(p.stat().st_size for p in output.glob('*') if p.is_file())
    atomic_write_json(ROOT/'relatorios/smoke_fase_a.json',{
        'purpose':'5 episodes per cell, engineering smoke; NOT final or representative learning',
        'runs':len(results),'workers':2,'wall_seconds':elapsed,'versions':library_versions(),
        'training_seconds_by_run':durations,'artifact_bytes':sizes,
        'rough_final_serial_training_seconds':sum(durations)*(1000/5)*10,
        'rough_final_two_worker_seconds':sum(durations)*(1000/5)*10/2,
        'conservative_storage_bytes_linear_extrapolation':sizes*(1000/5)*10,
        'estimate_caveat':'Linear extrapolation from 5 exploratory episodes; convergence, contention, evaluation and videos change cost. Not a guarantee.',
        'receipts':receipts})
    print(f'Smoke: {len(results)} runs, 2 workers, wall={elapsed:.3f}s; no final run')


if __name__=='__main__': main()
