"""Recover incomplete bundles without overwriting them; skip completed valid runs."""
from pathlib import Path
from datetime import datetime,timezone
import sys,subprocess,shutil
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json,expand_final,run_filename
from mountaincar_rl.storage import inspect_result_file
from mountaincar_rl.checkpoints import load_checkpoint,file_hash


def recover(directory,configs):
    directory=directory.resolve()
    assert directory==ROOT/'resultados/cmp629-final-v2-20261011'
    for c in configs:
        target=directory/run_filename(c)
        complete=False
        if target.exists() and inspect_result_file(target,c)[0]:
            try:
                r=read_json(target); cp=r['checkpoint']
                load_checkpoint(directory/cp['file'],cp['sha256'],c)
                complete=all(file_hash(directory/t['file'])==t['sha256'] for t in r['tables'])
            except (KeyError,ValueError,OSError):pass
        if complete:continue
        parts=[target.with_suffix(suffix) for suffix in ['.json','.npz','.training.csv','.evaluation.csv','.lock','.error.json']]
        parts=[p for p in parts if p.exists()]
        if parts:
            dest=directory/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')/target.stem
            dest.mkdir(parents=True,exist_ok=False)
            for p in parts:
                assert p.parent==directory
                shutil.move(str(p),str(dest/p.name))
            print('Partial/invalid bundle preserved:',dest.relative_to(ROOT))


def main():
    if '--autorizado-apos-go' not in sys.argv:
        raise SystemExit('Requires --autorizado-apos-go')
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            command=' '.join(p.info['cmdline'] or [])
            if p.pid!=__import__('os').getpid() and 'python' in p.name().lower() and ('scripts/executar_final.py' in command or 'scripts\\executar_final.py' in command):
                raise SystemExit('An active campaign exists; do not recover while workers are running')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert read_json(ROOT/'relatorios/gate_fase_a.json')['decision']=='GO'
    p=read_json(ROOT/'configs/protocolo_final_v2.json')
    configs=expand_final(p,read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json'))
    recover(ROOT/'resultados/cmp629-final-v2-20261011',configs)
    subprocess.run([sys.executable,str(ROOT/'scripts/executar_final.py'),'--autorizado-apos-go','--workers','2'],check=True,cwd=ROOT)


if __name__=='__main__':main()
