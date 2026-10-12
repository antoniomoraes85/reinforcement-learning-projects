from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.analysis import inferential_tests
from mountaincar_rl.metrics import student_t_curve_ci
from mountaincar_rl.agents import train_agent
from mountaincar_rl.config import read_json,expand_final
from test_phase_a import cfg


def test_all_tied_inference_is_explicit_not_nan():
    p=read_json(ROOT/'configs/protocolo_final_v2.json')
    rows=[{'algorithm':a,'discretization':d,'seed':s,'auc_normalized':0.}
          for a in ['q_learning','sarsa_lambda','dyna_q'] for d in ['4x2','12x4'] for s in [1,2,3]]
    r=inferential_tests(rows,p)
    json.dumps(r,allow_nan=False)
    for item in r['by_discretization'].values():
        assert item['friedman']['status']=='undefined_all_tied'
        assert all(pair['p_holm']==1 for pair in item['post_hoc'])


def test_missing_rolling_prefix_is_not_zero():
    mean,std,lo,hi=student_t_curve_ci(np.array([[np.nan,1,2],[np.nan,1,4]]))
    assert np.isnan(mean[0]) and np.isnan(lo[0])
    assert mean[2]==3 and std[2]==pytest.approx(np.sqrt(2))


@pytest.mark.parametrize('algorithm',['q_learning','sarsa_lambda','dyna_q'])
def test_fine_discretization_reproduction(algorithm):
    c=cfg(algorithm,'12x4')
    a,m=train_agent(c); b,n=train_agent(c)
    np.testing.assert_array_equal(a,b)
    assert m['episode_returns']==n['episode_returns']


@pytest.mark.parametrize('disc',['4x2','12x4'])
def test_n_zero_identical_across_two_seeds(disc):
    for seed in [7,91]:
        c=cfg('q_learning',disc); c['seed']=seed
        other=deepcopy(c); other['algorithm']='dyna_q'; other['hyperparameters']['N']=0
        a,m=train_agent(c); b,n=train_agent(other)
        np.testing.assert_array_equal(a,b)
        assert m['episode_returns']==n['episode_returns']
        assert m['episode_reset_seeds']==n['episode_reset_seeds']


def test_final_configuration_fixed():
    p=read_json(ROOT/'configs/protocolo_final_v2.json')
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')
    runs=expand_final(p,selection)
    assert len(runs)==60
    assert len({r['seed'] for r in runs})==10
    assert all(r['evaluation']['reset_seeds']==list(range(800000,800100)) for r in runs)
    assert p['demo_reset_seed'] not in p['evaluation']['reset_seeds']
    assert selection['selected_hyperparameters']['q_learning']['alpha']==.1
    assert selection['selected_hyperparameters']['sarsa_lambda']['alpha']==.02
    assert selection['selected_hyperparameters']['dyna_q']['N']==5
