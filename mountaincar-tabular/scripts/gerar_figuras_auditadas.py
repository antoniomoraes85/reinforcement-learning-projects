"""Figures from real per-run histories; never retrains or edits source data."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.analysis import load_valid_results,triage_rows
from mountaincar_rl.config import read_json,ALGORITHMS
from mountaincar_rl.metrics import moving_average,student_t_curve_ci

LABELS={'q_learning':'Q-learning','sarsa_lambda':'SARSA(λ)','dyna_q':'Dyna-Q'}
COLORS={'q_learning':'#0072B2','sarsa_lambda':'#D55E00','dyna_q':'#009E73'}


def save(fig,path):
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dados',type=Path,default=ROOT/'resultados/triagem')
    p.add_argument('--saida',type=Path,default=ROOT/'figuras/auditoria_v2')
    args=p.parse_args()
    if 'triagem' in args.saida.resolve().parts: raise ValueError('Do not overwrite legacy figures')
    args.saida.mkdir(parents=True,exist_ok=True)
    all_results,problems=load_valid_results(args.dados)
    if problems or not all_results: raise ValueError(problems or 'No real data')
    protocol=read_json(ROOT/'configs/protocolo_triagem.json')
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')['selected_hyperparameters']
    selected=[r for r in all_results if r['config']['hyperparameters']==selection[r['config']['algorithm']]]
    rows=pd.DataFrame(triage_rows(selected,protocol))
    rows=rows.rename(columns={'stable_episode':'first_threshold_episode','interactions_to_stability':'interactions_to_first_threshold'})
    # Evaluation return is exactly -steps for unmodified MountainCar; explicit derivation, not inferred learning.
    rows['evaluation_mean_return']=-rows['mean_steps_all']
    rows.to_csv(args.saida/'selected_seed_metrics.csv',index=False)
    full=pd.DataFrame(triage_rows(all_results,protocol))
    full.to_csv(args.saida/'all_seed_metrics.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    campaign=all_results[0]['config'].get('experiment_id',all_results[0]['config']['experiment_version'])
    # No pooling episodes as independent units. Error bars are t intervals across trained policies.
    metrics=[('evaluation_success_rate','Sucesso na avaliação',(0,1)),
             ('evaluation_mean_return','Retorno médio na avaliação',(-200,0)),
             ('mean_steps_all','Passos com falhas',(0,200)),
             ('mean_steps_success','Passos entre sucessos',(0,200)),
             ('auc_normalized','AUC normalizada (treino)',(0,1)),
             ('final_window_success','Sucesso últimos 100 episódios (treino)',(0,1))]
    for disc in ['4x2','12x4']:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,(field,label,limits) in zip(axes.flat,metrics):
            for i,alg in enumerate(ALGORITHMS):
                values=rows.loc[(rows.discretization==disc)&(rows.algorithm==alg),field].dropna().to_numpy()
                ax.scatter(i+np.linspace(-.1,.1,len(values)),values,color=COLORS[alg],s=25)
                if len(values):
                    ax.scatter(i,values.mean(),color='black',marker='_',s=150)
                    if len(values)>1:
                        margin=stats.t.ppf(.975,len(values)-1)*stats.sem(values)
                        ax.errorbar(i,values.mean(),yerr=margin,color=COLORS[alg],capsize=5)
                else: ax.text(i,.5*sum(limits),'sem sucesso',rotation=90,ha='center',fontsize=8)
            ax.set(xticks=range(3),xticklabels=[LABELS[a] for a in ALGORITHMS],ylabel=label,ylim=limits)
            ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{campaign} | {disc} | pontos = políticas; média e IC t 95%\nIC pode ultrapassar limites físicos; eixos limitados à faixa da métrica',fontsize=11)
        save(fig,args.saida/f'painel_{disc}')
        group=[r for r in selected if r['config']['discretization']['name']==disc]
        for field,label,limits,rolling in [('episode_returns','Retorno móvel (20 episódios)',(-200,0),20),
                                           ('episode_successes','Sucesso móvel (20 episódios)',(0,1),20)]:
            fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
            for alg in ALGORITHMS:
                runs=sorted([r for r in group if r['config']['algorithm']==alg],key=lambda r:r['config']['seed'])
                if not runs: continue
                ys=np.array([moving_average(r['training_metrics'][field],rolling) for r in runs])
                x=np.arange(1,ys.shape[1]+1)
                mean,_,lo,hi=student_t_curve_ci(ys)
                axes[0].plot(x,mean,label=LABELS[alg],color=COLORS[alg]); axes[0].fill_between(x,lo,hi,color=COLORS[alg],alpha=.16)
                # Restrict interpolation to common observed interaction support; no extrapolation.
                xs=[np.asarray(r['training_metrics']['cumulative_env_steps'])[rolling-1:] for r in runs]
                grid=np.linspace(max(a[0] for a in xs),min(a[-1] for a in xs),300)
                interpolated=np.array([np.interp(grid,xs[i],ys[i,rolling-1:]) for i in range(len(runs))])
                mean,_,lo,hi=student_t_curve_ci(interpolated)
                axes[1].plot(grid,mean,label=LABELS[alg],color=COLORS[alg]); axes[1].fill_between(grid,lo,hi,color=COLORS[alg],alpha=.16)
            axes[0].set(xlabel='Episódios',ylabel=label,ylim=limits); axes[1].set(xlabel='Interações reais (suporte comum)',ylim=limits)
            axes[0].legend(); fig.suptitle(f'{campaign} | {disc} | IC t 95% entre seeds; curvas reais')
            save(fig,args.saida/f'{field}_{disc}')
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,alg in zip(axes,ALGORITHMS):
            subset=full[(full.algorithm==alg)&(full.discretization==disc)].copy()
            subset['configuration']=subset.apply(lambda r:f"α={r.alpha:g}; λ={r['lambda']:g}; d={r.epsilon_decay:g}; N={int(r.N)}",axis=1)
            names=sorted(subset.configuration.unique())
            for i,name in enumerate(names):
                values=subset.loc[subset.configuration==name,'auc_normalized']
                ax.scatter(values,np.full(len(values),i),color=COLORS[alg],alpha=.6)
                ax.scatter(values.mean(),i,color='black',marker='|',s=100)
            ax.set(xlim=(0,1),yticks=range(len(names)),yticklabels=names,xlabel='AUC de treino',title=LABELS[alg])
            ax.tick_params(axis='y',labelsize=7)
        fig.suptitle(f'Sensibilidade e distribuição da AUC | {disc} | todas configurações da campanha')
        save(fig,args.saida/f'sensibilidade_{disc}')
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for alg in ALGORITHMS:
            sub=rows[(rows.algorithm==alg)&(rows.discretization==disc)]
            axes[0].scatter(sub.total_env_steps,sub.evaluation_success_rate,label=LABELS[alg],color=COLORS[alg])
            axes[1].scatter(sub.training_time_seconds,sub.evaluation_success_rate,color=COLORS[alg])
            reached=sub.dropna(subset=['first_threshold_episode'])
            axes[2].scatter(reached.first_threshold_episode,reached.final_window_success,label=f'{LABELS[alg]}: {len(reached)}/{len(sub)} atingiram',color=COLORS[alg])
        axes[0].set(xlabel='Interações reais',ylabel='Sucesso na avaliação',ylim=(0,1)); axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Tempo de treino (s)',ylim=(0,1)); axes[2].set(xlabel='Primeiro atingimento do limiar',ylabel='Sucesso janela final (treino)',ylim=(0,1),xlim=(0,1000)); axes[2].legend(fontsize=7)
        fig.suptitle(f'{disc} | custo × qualidade; atingimento não é estabilidade sustentada')
        save(fig,args.saida/f'custos_limiar_{disc}')
    print(f'{len(selected)} selected policies / {len(all_results)} real runs; figures in {args.saida}')


if __name__=='__main__': main()
