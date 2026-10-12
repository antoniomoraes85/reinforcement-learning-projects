"""Real rgb_array rollouts from verified checkpoints; no training."""
from pathlib import Path
import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from PIL import Image,ImageDraw
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.checkpoints import load_checkpoint,file_hash
from mountaincar_rl.config import read_json,ALGORITHMS
from mountaincar_rl.agents import Discretizer,epsilon_greedy
from mountaincar_rl.storage import atomic_write_csv,atomic_write_json,inspect_result_file

DEMO_SEED=900000  # fixed beforehand, disjoint from final evaluation


def save_video(frames,path):
    try:
        imageio.mimsave(path.with_suffix('.mp4'),frames,fps=30,macro_block_size=1)
        return path.with_suffix('.mp4')
    except (OSError,RuntimeError,ValueError):
        imageio.mimsave(path.with_suffix('.gif'),frames,duration=1000/30,loop=0)
        return path.with_suffix('.gif')


def rollout(result,source,output):
    cp=result['checkpoint']
    q,c=load_checkpoint(source/cp['file'],cp['sha256'],result['config'])
    d=Discretizer.from_config(c)
    env=gym.make('MountainCar-v0',render_mode='rgb_array')
    rng=np.random.default_rng(np.random.SeedSequence([DEMO_SEED,3]))
    frames=[]; rows=[]; total=0
    try:
        observation,_=env.reset(seed=DEMO_SEED)
        for step in range(1,201):
            previous=observation.copy()
            action=epsilon_greedy(q,d.transform(observation),0,rng)
            observation,reward,terminated,truncated,_=env.step(action)
            total+=float(reward)
            canvas=Image.fromarray(env.render())
            draw=ImageDraw.Draw(canvas)
            draw.rectangle((0,0,600,78),fill='white')
            draw.text((8,5),f"{c['experiment_id']} | {c['algorithm']} {c['discretization']['name']} | train seed {c['seed']}\n"
                      f"eval seed {DEMO_SEED} | step {step} | x={observation[0]:.3f} v={observation[1]:.3f}\n"
                      f"action={action} return={total:g} | " + ('SUCCESS' if terminated else 'FAILURE: time limit' if truncated else 'running'),fill='black')
            frames.append(np.array(canvas))
            rows.append(dict(step=step,position=float(previous[0]),velocity=float(previous[1]),
                             next_position=float(observation[0]),next_velocity=float(observation[1]),action=action,
                             reward=float(reward),cumulative_return=total,distance_to_goal=max(0.,.5-float(observation[0])),
                             terminated=bool(terminated),truncated=bool(truncated)))
            if terminated or truncated: break
    finally: env.close()
    name=f"{c['algorithm']}__{c['discretization']['name']}__s{c['seed']}"
    atomic_write_csv(output/(name+'.trajectory.csv'),rows)
    movie=save_video(frames,output/name)
    # Q cells plotted at physical bin boundaries; ties explicitly shown as -1.
    x=[-1.2,*d.position_edges,.6]; y=[-.07,*d.velocity_edges,.07]
    best=q.argmax(axis=2); best=np.where((q==q.max(axis=2,keepdims=True)).sum(axis=2)>1,-1,best)
    fig,axes=plt.subplots(1,4,figsize=(13,3))
    for a in range(3):
        m=axes[a].pcolormesh(x,y,q[:,:,a].T,vmin=q.min(),vmax=q.max(),shading='flat')
        axes[a].set(title=f'Q(s,{a})',xlabel='posição',ylabel='velocidade'); fig.colorbar(m,ax=axes[a])
    m=axes[3].pcolormesh(x,y,best.T,vmin=-1,vmax=2,shading='flat',cmap='viridis')
    axes[3].set(title='Ação greedy (-1: empate)',xlabel='posição'); fig.colorbar(m,ax=axes[3],ticks=[-1,0,1,2])
    fig.suptitle(f"{c['experiment_id']} | {name}"); fig.tight_layout(); fig.savefig(output/(name+'.policy.png'),dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    axes[0].plot([r['position'] for r in rows],[r['velocity'] for r in rows]); axes[0].axvline(.5,color='gray',linestyle='--'); axes[0].set(xlabel='posição',ylabel='velocidade')
    axes[1].plot([r['step'] for r in rows],[r['distance_to_goal'] for r in rows]); axes[1].set(xlabel='passo',ylabel='distância ao objetivo')
    axes[2].step([r['step'] for r in rows],[r['action'] for r in rows]); axes[2].set(xlabel='passo',ylabel='ação',yticks=[0,1,2])
    fig.suptitle(f"{name}: {'sucesso' if rows[-1]['terminated'] else 'falha'} | demo, não estimativa"); fig.tight_layout(); fig.savefig(output/(name+'.trajectory.png'),dpi=300); plt.close(fig)
    return frames,{'algorithm':c['algorithm'],'discretization':c['discretization']['name'],'train_seed':c['seed'],
                   'reset_seed':DEMO_SEED,'source_checkpoint_sha256':cp['sha256'],'video':movie.name,
                   'video_sha256':file_hash(movie),'steps':len(rows),'return':total,'success':rows[-1]['terminated'],
                   'experiment_id':c['experiment_id'],'initial_state':[rows[0]['position'],rows[0]['velocity']]}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dados',type=Path,default=ROOT/'resultados/auditoria_v2')
    p.add_argument('--saida',type=Path,default=ROOT/'figuras/animacoes_auditoria_v2')
    args=p.parse_args(); args.saida.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    records=[]
    results=[]
    for path in args.dados.glob('*__*__s*__*.json'):
        if not inspect_result_file(path)[0]: raise ValueError(f'Invalid result {path}')
        results.append(read_json(path))
    for disc in ['4x2','12x4']:
        groups=[]
        for algorithm in ALGORITHMS:
            candidates=[r for r in results if r['config']['algorithm']==algorithm and r['config']['discretization']['name']==disc]
            if not candidates: raise ValueError(f'Missing checkpoint: {algorithm}/{disc}')
            # Predeclared selection: smallest training seed; never select by performance.
            result=min(candidates,key=lambda r:r['config']['seed'])
            frames,record=rollout(result,args.dados,args.saida); groups.append(frames); records.append(record)
        assert all(r['initial_state']==records[-1]['initial_state'] for r in records[-3:])
        combined=[np.concatenate([frames[min(i,len(frames)-1)] for frames in groups],axis=1)
                  for i in range(max(map(len,groups)))]
        save_video(combined,args.saida/f'comparacao_{disc}')
    atomic_write_json(args.saida/'manifesto_animacoes.json',{'selection_rule':'smallest training seed, fixed demo seed 900000; no outcome selection','rollouts':records})
    print(f'{len(records)} real rollouts; 2 side-by-side comparisons; source={args.dados}')


if __name__=='__main__': main()
