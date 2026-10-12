from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mountaincar_rl import agents
from mountaincar_rl.agents import *
from mountaincar_rl.config import config_hash, expand_triage, read_json, validate_protocol
from mountaincar_rl.checkpoints import save_checkpoint, load_checkpoint
from mountaincar_rl.experiment import execute_and_store
from mountaincar_rl.storage import load_json, inspect_result_file


def cfg(algorithm='q_learning', disc='4x2'):
    result = next(deepcopy(c) for c in expand_triage(read_json(ROOT/'configs/protocolo_triagem.json'))
                  if c['algorithm'] == algorithm and c['discretization']['name'] == disc)
    result.update(phase='audit', experiment_id='phase-a-tests', experiment_version='audit-v2')
    result['training']['episodes'] = 2
    result['evaluation'] = {'episodes': 2, 'reset_seeds': [800001, 800002]}
    return result


@pytest.mark.parametrize('disc', ['4x2', '12x4'])
def test_environment_and_boundaries(disc):
    c = cfg(disc=disc)
    d = Discretizer.from_config(c)
    env = make_env(c)
    assert env.spec.max_episode_steps == 200 and env.action_space.n == 3
    np.testing.assert_allclose(env.observation_space.low, [-1.2, -.07])
    np.testing.assert_allclose(env.observation_space.high, [.6, .07])
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_array_equal(a, b)
    assert -.6 <= a[0] <= -.4 and a[1] == 0
    assert d.transform((-1.2, -.07)) == (0, 0)
    assert d.transform((.6, .07)) == tuple(x-1 for x in d.shape)
    for axis, edges in enumerate((d.position_edges, d.velocity_edges)):
        for i, edge in enumerate(edges):
            observation = [-.5, 0.]
            observation[axis] = edge
            assert d.transform(observation)[axis] == i+1
            observation[axis] = np.nextafter(edge, -np.inf)
            assert d.transform(observation)[axis] == i
    for _ in range(200):
        _, reward, terminal, truncated, _ = env.step(1)
    assert truncated and not terminal and reward == -1
    env.reset(seed=123)
    env.unwrapped.state = np.array([.51, .01])
    assert env.step(2)[2]
    env.close()


def test_manual_td_all_algorithms_and_trace_decay():
    q = np.zeros((4, 2, 3)); q[0,0,1] = 1.; q[1,1,2] = 2.
    assert scalar_q_update(q, (0,0), 1, -1+.99*2, .1)[1] == pytest.approx(.998)
    q[0,0,1] = 1.
    e = np.zeros_like(q); e[0,1,0] = .5
    delta, active = sarsa_trace_update(q, e, (0,0), 1, -1, (1,1), 2, False, .1, .99, .9)
    assert delta == pytest.approx(-.02) and active == 2
    assert q[0,0,1] == pytest.approx(.998) and q[0,1,0] == pytest.approx(-.001)
    assert e[0,0,1] == pytest.approx(.891) and e[0,1,0] == pytest.approx(.4455)
    model = EmpiricalTransitionModel(); model.observe((0,0), 1, -1, (1,1), False)
    q[0,0,1] = 1.
    assert planning_update(q, model, .1, .99, np.random.default_rng(0))['new_value'] == pytest.approx(.998)
    for terminal in (True, False):
        table = np.ones((4,2,3))*2
        trace = np.zeros_like(table)
        sarsa_trace_update(table, trace, (0,0), 0, -1, (1,1), None if terminal else 0, terminal, .1, .99, .9)
        assert table[0,0,0] == pytest.approx(1.7 if terminal else 1.898)


def test_empirical_70_30_fixed_statistical_test():
    model = EmpiricalTransitionModel()
    for i in range(100):
        model.observe((0,0), 2, -1 if i < 70 else -2, (0,1) if i < 70 else (1,1), i >= 70)
    rng = np.random.default_rng(20261011)
    fraction = sum(model.sample(rng)[1][2] for _ in range(10000))/10000
    assert abs(fraction-.3) < 5*np.sqrt(.3*.7/10000)
    assert len(model.outcomes[((0,0),2)]) == 2


@pytest.mark.parametrize('algorithm', ['q_learning', 'sarsa_lambda', 'dyna_q'])
def test_reproduction_counters_eval_immutable(algorithm):
    c = cfg(algorithm)
    q1, m1 = train_agent(c); q2, m2 = train_agent(c)
    np.testing.assert_array_equal(q1,q2)
    assert {k:v for k,v in m1.items() if k != 'training_time_seconds'} == {k:v for k,v in m2.items() if k != 'training_time_seconds'}
    steps = sum(m1['episode_steps'])
    assert m1['cumulative_real_td_updates'][-1] == steps
    assert m1['cumulative_planning_updates'][-1] == (steps*c['hyperparameters']['N'] if algorithm=='dyna_q' else 0)
    assert (m1['cumulative_trace_components'][-1] >= steps) if algorithm=='sarsa_lambda' else m1['cumulative_trace_components'][-1] == 0
    before=q1.copy(); v1=evaluate_agent(q1,c); v2=evaluate_agent(q1,c)
    np.testing.assert_array_equal(before,q1)
    for key in ['steps','returns','successes','terminated','truncated','reset_seeds']:
        assert v1[key]==v2[key]
    assert v1['reset_seeds']==[800001,800002]
    assert set(v1['reset_seeds']).isdisjoint(m1['episode_reset_seeds'])
    assert v1['returns']==[-float(x) for x in v1['steps']]
    assert v1['inference_calls']==sum(v1['steps'])


@pytest.mark.parametrize('terminal', [True,False])
@pytest.mark.parametrize('algorithm', ['q_learning','sarsa_lambda','dyna_q'])
def test_single_step_episodes_reset_traces_and_time_limit(monkeypatch,terminal,algorithm):
    class Space:
        def seed(self, seed): pass
    class Scripted:
        action_space=Space()
        def reset(self,seed): return np.array([-.5,0.]),{}
        def step(self,a): return np.array([-.5,0.]),-1.,terminal,not terminal,{}
        def close(self): pass
    monkeypatch.setattr(agents,'make_env',lambda c: Scripted())
    monkeypatch.setattr(agents,'epsilon_greedy',lambda *args: 0)
    c=cfg(algorithm); c['hyperparameters'].update(alpha=.1,N=0)
    q,m=train_agent(c)
    # Each episode one transition; second SARSA trace must start at zero.
    expected=-.19 if terminal else (-.199 if algorithm=='sarsa_lambda' else -.19)
    # SARSA bootstraps the same chosen action; Q uses still-zero other actions.
    if not terminal and algorithm=='sarsa_lambda': expected=-.1999
    assert q[1,1,0]==pytest.approx(expected)
    assert m['cumulative_trace_components'][-1] == (2 if algorithm=='sarsa_lambda' else 0)


def test_ties_and_serialization(tmp_path):
    c=cfg(); q=np.zeros((4,2,3)); path=tmp_path/'q.npz'
    digest=save_checkpoint(path,q,c); loaded,cc=load_checkpoint(path,digest,c)
    np.testing.assert_array_equal(q,loaded)
    rng=np.random.default_rng(7)
    assert set(epsilon_greedy(loaded,(0,0),0,rng) for _ in range(100))=={0,1,2}
    loaded[0,0,2]=1
    assert epsilon_greedy(loaded,(0,0),0,rng)==2
    with pytest.raises(ValueError): load_checkpoint(path,'bad',c)
    with pytest.raises(FileExistsError): save_checkpoint(path,q,c)


def test_cache_full_configuration_and_bundle(tmp_path):
    c=cfg()
    paths=[('algorithm',),('experiment_version',),('experiment_id',),('seed',),
           ('discretization','position_edges'),('discretization','velocity_edges'),
           ('hyperparameters','alpha'),('hyperparameters','lambda'),('hyperparameters','N'),
           ('hyperparameters','epsilon_decay'),('training','gamma'),('training','epsilon_start'),
           ('training','epsilon_min'),('training','episodes'),('evaluation','episodes'),('evaluation','reset_seeds')]
    for path in paths:
        changed=deepcopy(c); node=changed
        for k in path[:-1]: node=node[k]
        node[path[-1]]='different-value'
        assert config_hash(changed)!=config_hash(c)
    first=execute_and_store(c,tmp_path)
    assert first['status']=='executed'
    assert execute_and_store(c,tmp_path)['status']=='cached'
    r=load_json(first['path']); assert inspect_result_file(first['path'],c)[0]
    q,cc=load_checkpoint(tmp_path/r['checkpoint']['file'],r['checkpoint']['sha256'],c)
    assert len(r['tables'])==2
    with pytest.raises(PermissionError): execute_and_store(dict(c,phase='triagem'),tmp_path)


def test_malformed_json_returns_validation_failure(tmp_path):
    p=tmp_path/'broken.json'; p.write_text('null')
    assert inspect_result_file(p)[0] is False


def test_invalid_discretization_rejected():
    p=read_json(ROOT/'configs/protocolo_triagem.json')
    p['discretizations']['4x2']['position_edges']=[-.2,-.6,.2]
    with pytest.raises(ValueError): validate_protocol(p)
