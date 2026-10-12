"""Read-only audit of stored final evidence. No training or policy evaluation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json,config_hash,expand_final,run_filename
from mountaincar_rl.storage import inspect_result_file
from mountaincar_rl.checkpoints import load_checkpoint,file_hash
from mountaincar_rl.agents import Discretizer


def main():
    protocol=read_json(ROOT/'configs/protocolo_final_v2.json')
    assert config_hash(protocol)=='fb1deb3ea35f1323438c10878f24225dfdb6fdebcd4c12cf50d0e714d2cf4094'
    directory=ROOT/'resultados/cmp629-final-v2-20261011'
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')
    configs=expand_final(protocol,selection); assert len(configs)==60
    rows=pd.read_csv(ROOT/'relatorios/final_v2/metricas_por_seed.csv').set_index('config_hash')
    charts=pd.read_csv(ROOT/'figuras/final_v2/selected_seed_metrics.csv').set_index('config_hash')
    assert len(rows)==len(charts)==60
    hashes={}; runs={}
    for c in configs:
        path=directory/run_filename(c); ok,errors=inspect_result_file(path,c); assert ok,errors
        r=read_json(path);cp=r['checkpoint']; load_checkpoint(directory/cp['file'],cp['sha256'],c)
        assert r['evaluation_metrics']['reset_seeds']==protocol['evaluation']['reset_seeds']
        assert np.isclose(rows.loc[r['config_hash'],'evaluation_success_rate'],r['evaluation_metrics']['success_rate'])
        assert np.isclose(charts.loc[r['config_hash'],'evaluation_success_rate'],r['evaluation_metrics']['success_rate'])
        for p in [path,directory/cp['file'],*[directory/t['file'] for t in r['tables']]]:
            hashes[p.relative_to(ROOT).as_posix()]=file_hash(p)
        for t in r['tables']:assert file_hash(directory/t['file'])==t['sha256']
        runs[(c['algorithm'],c['discretization']['name'],c['seed'])]=r
    media=ROOT/'figuras/animacoes_final_v2'; manifest=read_json(media/'manifesto_animacoes.json')
    video_reports=[]
    for item in manifest['rollouts']:
        r=runs[(item['algorithm'],item['discretization'],item['train_seed'])]
        assert item['train_seed']==min(protocol['seeds']) and item['reset_seed']==protocol['demo_reset_seed']
        assert item['source_checkpoint_sha256']==r['checkpoint']['sha256']
        assert file_hash(media/item['video'])==item['video_sha256']
        q,c=load_checkpoint(directory/r['checkpoint']['file'],item['source_checkpoint_sha256'],r['config'])
        d=Discretizer.from_config(c)
        trajectory=pd.read_csv(media/(Path(item['video']).stem+'.trajectory.csv'))
        for row in trajectory.itertuples():
            values=q[d.transform((row.position,row.velocity))]
            assert values[row.action]==values.max(), 'Saved trajectory action is not greedy for verified checkpoint'
        assert len(trajectory)==item['steps'] and trajectory.reward.sum()==item['return']
        assert bool(trajectory.terminated.iloc[-1])==item['success']
        np.testing.assert_allclose(trajectory[['position','velocity']].iloc[0],item['initial_state'])
    for disc in ['4x2','12x4']:
        items=[i for i in manifest['rollouts'] if i['discretization']==disc]
        assert len(items)==3 and len({tuple(i['initial_state']) for i in items})==1
    qa=ROOT/'relatorios/qa_entrega';qa.mkdir(exist_ok=True)
    for path in media.glob('*.mp4'):
        reader=imageio.get_reader(path);count=0
        for frame in reader:count+=1
        reader.close(); assert count>0
        if path.name.startswith('comparacao'):imageio.imwrite(qa/(path.stem+'_ultimo_frame.png'),frame)
        video_reports.append({'file':path.name,'frames':count,'sha256':file_hash(path)})
    pngs=list((ROOT/'figuras/final_v2').glob('*.png'))+list(media.glob('*.png'))
    for p in pngs:
        with Image.open(p) as im:assert min(im.info.get('dpi',(0,0)))>=200
    target=ROOT/'relatorios/manifesto_final_preservado.json'
    if target.exists():assert read_json(target)==hashes,'Final raw artifacts changed'
    else:target.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    result={'final_runs':60,'checkpoints':60,'raw_csvs':120,'training_runs_executed':0,'raw_artifacts_hashed':len(hashes),
            'chart_seed_rows_reconciled':60,'verified_greedy_trajectories':6,'decoded_videos':video_reports,
            'pngs_checked':len(pngs),'status':'passed','protocol_sha256':config_hash(protocol)}
    (ROOT/'relatorios/auditoria_entrega.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
