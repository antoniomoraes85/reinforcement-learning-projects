from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gerar_figuras_auditadas import save,COLORS,LABELS,ALGORITHMS


def main():
    data=pd.read_csv(ROOT/'relatorios/final_v2/metricas_por_seed.csv')
    out=ROOT/'figuras/final_v2'; out.mkdir(exist_ok=True)
    fields=[('total_env_steps','Interações reais'),('training_time_seconds','Tempo real de treino (s)'),
            ('real_td_updates','Atualizações TD reais'),('planning_updates','Backups simulados Dyna'),
            ('trace_components','Componentes dos traços SARSA'),('inference_mean_ns','Latência de política (ns)')]
    for disc in ['4x2','12x4']:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,(field,label) in zip(axes.flat,fields):
            for alg in ALGORITHMS:
                g=data[(data.discretization==disc)&(data.algorithm==alg)]
                ax.scatter(g[field],g.evaluation_success_rate,label=LABELS[alg],color=COLORS[alg],alpha=.8)
            ax.set(xlabel=label,ylabel='Sucesso avaliação',ylim=(-.02,1.02),xlim=(-.02*max(1,data[field].max()),1.05*max(1,data[field].max())))
            ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Campanha final v2 | {disc} | 10 políticas/método; custos em unidades distintas')
        save(fig,out/f'custos_separados_{disc}')
    print('Separate cost/quality panels generated from real counters')


if __name__=='__main__':main()
