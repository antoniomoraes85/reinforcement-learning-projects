"""Validate all 60 saved bundles before producing final inference and report."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json,expand_final,run_filename,config_hash
from mountaincar_rl.storage import inspect_result_file,atomic_write_json
from mountaincar_rl.checkpoints import load_checkpoint,file_hash
from mountaincar_rl.analysis import final_seed_rows,inferential_tests

DIRECTORY=ROOT/'resultados/cmp629-final-v2-20261011'


def main():
    p=read_json(ROOT/'configs/protocolo_final_v2.json')
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')
    expected=expand_final(p,selection)
    results=[]
    for c in expected:
        path=DIRECTORY/run_filename(c)
        ok,errors=inspect_result_file(path,c)
        if not ok: raise ValueError(f'{path.name}: {errors}')
        r=read_json(path); cp=r['checkpoint']; load_checkpoint(DIRECTORY/cp['file'],cp['sha256'],c)
        for table in r['tables']: assert file_hash(DIRECTORY/table['file'])==table['sha256']
        t=r['training_metrics']; e=r['evaluation_metrics']
        assert e['reset_seeds']==p['evaluation']['reset_seeds']
        assert set(t['episode_reset_seeds']).isdisjoint(e['reset_seeds'])
        assert all(a or b for a,b in zip(e['terminated'],e['truncated']))
        assert e['successes']==e['terminated']
        successful=[n for n,s in zip(e['steps'],e['successes']) if s]
        assert (e['mean_steps_success'] is None) if not successful else np.isclose(e['mean_steps_success'],np.mean(successful))
        assert e['inference_calls']==sum(e['steps'])
        results.append(r)
    assert len(results)==60 and len(list(DIRECTORY.glob('*__*__s*__*.json')))==60
    out=ROOT/'relatorios/final_v2'; out.mkdir(exist_ok=True)
    rows=pd.DataFrame(final_seed_rows(results,p)).rename(columns={'stable_episode':'first_threshold_episode'})
    by_hash={r['config_hash']:r for r in results}
    for field in ['real_td_updates','planning_updates','trace_components','inference_mean_ns']:
        rows[field]=[by_hash[h]['evaluation_metrics']['inference_mean_ns'] if field=='inference_mean_ns'
                     else by_hash[h]['training_metrics']['cumulative_'+field][-1] for h in rows.config_hash]
    rows['evaluation_return']=-rows.mean_steps_all
    rows.to_csv(out/'metricas_por_seed.csv',index=False)
    summaries=[]
    for (a,d),g in rows.groupby(['algorithm','discretization']):
        for field in ['evaluation_success_rate','evaluation_return','mean_steps_all','mean_steps_success',
                      'auc_normalized','final_window_success','total_env_steps','training_time_seconds',
                      'real_td_updates','planning_updates','trace_components','inference_mean_ns']:
            values=g[field].dropna().to_numpy(); n=len(values)
            mean=float(values.mean()) if n else None
            sd=float(values.std(ddof=1)) if n>1 else None
            margin=stats.t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else None
            summaries.append(dict(algorithm=a,discretization=d,metric=field,n_seeds=n,mean=mean,
                                  median=float(np.median(values)) if n else None,std=sd,
                                  ci95_low=mean-margin if margin is not None else None,
                                  ci95_high=mean+margin if margin is not None else None))
    pd.DataFrame(summaries).to_csv(out/'estatisticas_entre_seeds.csv',index=False)
    inference=inferential_tests(final_seed_rows(results,p),p)
    atomic_write_json(out/'inferencia_auc.json',inference)
    atomic_write_json(out/'validacao_final.json',{'valid':True,'expected':60,'valid_runs':60,
        'checkpoints':60,'raw_csvs':120,'protocol_sha256':config_hash(p),'evaluation_seeds_common':True})
    compact=rows.groupby(['algorithm','discretization']).agg(
        seeds=('seed','nunique'),success=('evaluation_success_rate','mean'),std=('evaluation_success_rate','std'),
        steps_all=('mean_steps_all','mean'),steps_success=('mean_steps_success','mean'),
        train_final=('final_window_success','mean'),auc=('auc_normalized','mean')).reset_index()
    lines=['| Algoritmo | Grade | Seeds | Sucesso avaliação | DP entre seeds | Passos com falhas | Passos sucessos* | Treino final | AUC |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in compact.itertuples():
        conditional='não definido' if pd.isna(r.steps_success) else f'{r.steps_success:.2f}'
        lines.append(f'| {r.algorithm} | {r.discretization} | {r.seeds} | {r.success:.2%} | {r.std:.2%} | {r.steps_all:.2f} | {conditional} | {r.train_final:.2%} | {r.auc:.5f} |')
    table='\n'.join(lines)
    (out/'RESULTADOS.md').write_text('# Campanha final v2 — resultados reais\n\n'+table+'\n\n* Média dos valores condicionais por política com ao menos um sucesso. Conferir `n_seeds` na tabela longa; políticas sem sucesso não têm duração condicional. A duração incluindo falhas inclui todas as políticas.\n',encoding='utf-8')
    print(table)
    print('60/60 validated; checkpoint and CSV SHA-256; common held-out reset seeds')


if __name__=='__main__': main()
