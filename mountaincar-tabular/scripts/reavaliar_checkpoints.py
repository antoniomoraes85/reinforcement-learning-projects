"""Re-evaluate every saved final Q table; does not train or tune any parameter."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json
from mountaincar_rl.agents import evaluate_agent
from mountaincar_rl.checkpoints import load_checkpoint


def main():
    start=time.perf_counter();directory=ROOT/'resultados/cmp629-final-v2-20261011'
    paths=sorted(directory.glob('*__*__s*__*.json')); assert len(paths)==60
    hashes=[]
    for path in paths:
        r=read_json(path);cp=r['checkpoint']
        q,c=load_checkpoint(directory/cp['file'],cp['sha256'],r['config'])
        actual=evaluate_agent(q,c);expected=r['evaluation_metrics']
        for key in ['steps','returns','successes','terminated','truncated','reset_seeds','success_rate','mean_steps_success','mean_steps_all']:
            assert actual[key]==expected[key],f'{path.name}: {key}'
        hashes.append(r['config_hash'])
    report={'validated_policies':60,'greedy_episodes_reproduced':6000,'new_training_runs':0,
            'wall_seconds':time.perf_counter()-start,'config_hashes':hashes,'status':'passed'}
    (ROOT/'relatorios/final_v2/reavaliacao_checkpoints.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('60 saved policies / 6000 greedy episodes reproduced exactly; no training')


if __name__=='__main__':main()
