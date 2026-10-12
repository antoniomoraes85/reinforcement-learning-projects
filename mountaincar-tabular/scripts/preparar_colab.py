"""Generate the presentation notebook with nbformat, no training by default."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook()
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    nb.cells=[
        md('''# CMP629 — MountainCar tabular: evidência, métodos e limites

## Resumo
Modo apresentação: **nenhum treinamento é executado por padrão**. Os gráficos de desempenho são da triagem real (108 execuções; 3 seeds por configuração). Os vídeos são de um **smoke técnico novo de 5 episódios**, não das políticas históricas nem do experimento final. Os 60 treinamentos finais continuam pendentes de autorização.

Este notebook substitui o snapshot exploratório por um leitor dos artefatos auditados. Abrir a branch do PR, não presumir que estes arquivos já existem na main.'''),
        code('''RUN_TRAINING = False
RUN_DEMO = False
import io, json, gzip, hashlib, urllib.request, pathlib, zipfile
import numpy as np
import pandas as pd
from IPython.display import display, Image, Video, Markdown
BRANCH = "codex/cmp629-fase-a-20261011"
BASE = f"https://raw.githubusercontent.com/antoniomoraes85/reinforcement-learning-projects/{BRANCH}/mountaincar-tabular"
OUT = pathlib.Path("cmp629_apresentacao")
OUT.mkdir(exist_ok=True)
def fetch(relative):
    with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
        return response.read()
manifest = json.loads(fetch("dados/auditoria_v2/manifesto_publicacao.json"))
def verified(relative):
    data = fetch(relative)
    assert hashlib.sha256(data).hexdigest() == manifest[relative], relative
    return data
print("Modo apresentação; fonte pública:", BRANCH)'''),
        md(r'''## Contexto e métodos
O carro precisa chegar à posição 0,5 usando embalo. Observação contínua: posição e velocidade. Ações 0/1/2: esquerda/neutro/direita; recompensa −1 por passo; limite padrão de 200 passos. Discretizações não uniformes 4×2 e 12×4, conforme protocolo.

| Método | Política de atualização | Modelo |
|---|---|---|
| Q-learning | off-policy (alvo greedy) | model-free |
| SARSA(λ) | on-policy (próxima ação ε-greedy) | model-free com traços acumulativos |
| Dyna-Q | off-policy | aprendizado direto + planejamento com modelo empírico |

Q-learning: $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_b Q(s',b)-Q(s,a)]$.
SARSA: $\delta=r+\gamma Q(s',a')-Q(s,a)$; $e(s,a)\leftarrow e(s,a)+1$; $Q\leftarrow Q+\alpha\delta e$; $e\leftarrow\gamma\lambda e$. Traços zerados a cada episódio.
Dyna-Q faz um backup real e N backups de desfechos amostrados do modelo. Cada par discreto armazena contagens de múltiplos desfechos, pois a discretização agrega estados contínuos.

Em término real, bootstrap zero. Em truncamento por tempo, mantemos bootstrap: aproximação do processo subjacente sem término, embora a avaliação continue limitada a 200 passos. Isso não é uma solução tabular exata do MDP finito de 200 passos (tempo não está no estado).

### Hipóteses e limites
Desempates greedy aleatórios reprodutíveis. γ=0,99. Seleção global pela AUC média de treinamento nas duas discretizações; avaliação não usada para selecionar. Seeds de treino são a unidade independente; episódios avaliados não são replicações de treinamento. Avaliação histórica usava seeds derivadas de cada seed de treino; v2 usa o mesmo conjunto fixo para todas as políticas. Isso impede tratar as duas campanhas como réplicas idênticas.'''),
        md('## Dados e validação\nAs 108 execuções foram localizadas e validadas individualmente. O arquivo abaixo é uma cópia comprimida derivada dos JSONs reais, não uma reconstrução de curvas a partir de médias.'),
        code('''histories = json.loads(gzip.decompress(verified("dados/auditoria_v2/triagem_historicos.json.gz")))
assert len(histories) == 108
assert len({r["config_hash"] for r in histories}) == 108
assert all(len(r["training_metrics"]["episode_returns"]) == 1000 for r in histories)
seed_table = pd.read_csv(io.BytesIO(verified("dados/auditoria_v2/selected_seed_metrics.csv")))
assert len(seed_table) == 18 and seed_table.config_hash.nunique() == 18
summary = seed_table.groupby(["algorithm","discretization"]).agg(
    n_seeds=("seed","nunique"), sucesso_avaliacao=("evaluation_success_rate","mean"),
    dp_entre_seeds=("evaluation_success_rate","std"), sucesso_final_treino=("final_window_success","mean"),
    passos_com_falhas=("mean_steps_all","mean"))
display(summary.round(4))
summary.to_csv(OUT/"resumo_triagem.csv")'''),
        md('''## Resultados exploratórios
SARSA(λ) 12×4 atingiu 100% nos episódios avaliados das três políticas selecionadas. Isso não prova superioridade geral. Dyna-Q 4×2 teve 0% nas políticas selecionadas. *State aliasing* é hipótese, não causalidade estabelecida. Valores de janela final de treino incluem exploração e não são taxas de avaliação.

Mesmas escalas por métrica, discretizações separadas. Pontos: políticas independentes; barras: IC t 95% entre seeds (n=3, grande incerteza).'''),
        code('''for disc in ("4x2", "12x4"):
    content = verified(f"figuras/auditoria_v2/painel_{disc}.png")
    (OUT/f"painel_{disc}.png").write_bytes(content)
    display(Image(content, width=1050))'''),
        md('### Aprendizagem observada\nCurvas reais, média móvel de 20 episódios e IC entre seeds. No eixo de interações, interpolação apenas no suporte observado comum; sem extrapolação. Primeiro atingimento do limiar não significa desempenho sustentado.'),
        code('''for disc in ("4x2", "12x4"):
    display(Image(verified(f"figuras/auditoria_v2/episode_returns_{disc}.png"), width=1000))'''),
        md('''### Custos e limitações de medição
`total_q_updates` histórico não é unidade homogênea: não comparar diretamente SARSA com os outros. A v2 registra separadamente interações, TD reais, planejamento, componentes dos traços, wall-clock e latência de seleção da ação (sem renderização). Hardware e carga afetam tempos.

### Animações reais do smoke técnico
Políticas treinadas por **apenas 5 episódios** para testar o sistema. Vídeos não representam a qualidade final. Seed de exibição fixada previamente em 900000, menor seed de treino disponível; falhas não são ocultadas. As políticas históricas não tinham checkpoints e não foram retreinadas.'''),
        code('''for disc in ("4x2", "12x4"):
    path = OUT/f"smoke_comparacao_{disc}.mp4"
    path.write_bytes(verified(f"figuras/animacoes_auditoria_v2/comparacao_{disc}.mp4"))
    display(Video(str(path), embed=True, width=1050))'''),
        md('''## Demonstração nova opcional
Somente se `RUN_TRAINING=True`: instala Gymnasium e baixa o mesmo módulo auditado, treinando uma nova política Q-learning por 5 episódios, sem executar a campanha final. `RUN_DEMO=True` mostra um quadro de um checkpoint técnico já público; não treina nem modifica os históricos.'''),
        code('''if RUN_TRAINING or RUN_DEMO:
    import sys, subprocess, importlib.util
    subprocess.check_call([sys.executable,"-m","pip","install","gymnasium[classic-control]==1.2.1"])
    module_path = OUT/"agents.py"
    module_path.write_bytes(verified("src/mountaincar_rl/agents.py"))
    spec = importlib.util.spec_from_file_location("cmp629_agents",module_path)
    agents = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=agents
    spec.loader.exec_module(agents)
if RUN_TRAINING:
    import copy
    demo_config=copy.deepcopy(histories[0]["config"])
    demo_config.update(algorithm="q_learning",phase="demo",experiment_id="new-colab-demo",seed=990001)
    demo_config["training"]["episodes"]=5
    demo_config["hyperparameters"]["N"]=0
    q_demo, metrics_demo=agents.train_agent(demo_config)
    print("NOVA DEMONSTRAÇÃO, NÃO TRIAGEM:",metrics_demo["episode_returns"])
if RUN_DEMO:
    import gymnasium as gym
    checkpoint = io.BytesIO(verified("dados/auditoria_v2/demo_checkpoint.npz"))
    with np.load(checkpoint,allow_pickle=False) as bundle:
        q_demo=bundle["q"]; demo_config=json.loads(str(bundle["config"]))
    env=gym.make("MountainCar-v0",render_mode="rgb_array")
    try:
        observation,_=env.reset(seed=900000)
        action=agents.epsilon_greedy(q_demo,agents.Discretizer.from_config(demo_config).transform(observation),0,np.random.default_rng(900000))
        env.step(action)
        from PIL import Image as PILImage
        display(PILImage.fromarray(env.render()))
    finally: env.close()'''),
        md('''## Próximos passos e exportação
60 treinamentos: 3 algoritmos × 2 discretizações × 10 novas seeds, ainda não autorizados nesta entrega. Congelar hiperparâmetros e não reajustar pelo teste. Interpretar resultados futuros por seed, com incerteza e limitações. Colab hospedado não foi executado por esta auditoria; validação automatizada usa kernel local limpo.

A célula final exporta tabelas, figuras e parecer. Não inicia treinamento.'''),
        code('''report = verified("relatorios/parecer_liberacao_final.md")
(OUT/"parecer_liberacao_final.md").write_bytes(report)
with zipfile.ZipFile("CMP629_apresentacao.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():
        if path.is_file(): archive.write(path,path.name)
print("Exportação: CMP629_apresentacao.zip; treinamento não executado por padrão")
# No Colab, descomente para baixar:
# from google.colab import files
# files.download("CMP629_apresentacao.zip")''')]
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    if (ROOT/'relatorios/final_v2/validacao_final.json').exists():
        nb.cells[0]=md('''# CMP629 — MountainCar: entrega final

**60 treinamentos finais validados**, além das 108 execuções históricas preservadas. Modo apresentação, sem retreino por padrão. A campanha final usa 10 seeds de treinamento por algoritmo/grade, 100 episódios de avaliação por política e hiperparâmetros congelados pela triagem. Vídeos finais carregam checkpoints reais; os vídeos de smoke permanecem identificados separadamente.

O notebook apresenta primeiro a proveniência da triagem e depois os resultados finais, sem misturar amostras.''')
        new=[md('''## Campanha final: 60 políticas independentes

Dez seeds por célula; IC calculado entre políticas. As 100 avaliações de uma política não são 100 treinamentos. Retorno e duração são redundantes neste benchmark (recompensa −1 por passo), mas registrados separadamente para auditoria. Parâmetros globais não foram reajustados pelo teste final.'''),
             code('''final_histories=json.loads(gzip.decompress(verified("dados/auditoria_v2/final_historicos.json.gz")))
assert len(final_histories)==60 and len({r["config_hash"] for r in final_histories})==60
final_metrics=pd.read_csv(io.BytesIO(verified("relatorios/final_v2/metricas_por_seed.csv")))
final_summary=pd.read_csv(io.BytesIO(verified("relatorios/final_v2/estatisticas_entre_seeds.csv")))
display(final_summary[final_summary.metric.isin(["evaluation_success_rate","mean_steps_all","mean_steps_success"])].round(4))
display(Markdown(verified("relatorios/final_v2/RESULTADOS.md").decode("utf-8")))
final_metrics.to_csv(OUT/"metricas_finais_por_seed.csv",index=False)'''),
             code('''for disc in ("4x2","12x4"):
    for figure in ("painel", "episode_returns", "episode_successes", "custos_limiar", "custos_separados"):
        data=verified(f"figuras/final_v2/{figure}_{disc}.png")
        (OUT/f"final_{figure}_{disc}.png").write_bytes(data)
        display(Image(data,width=1050))'''),
             md('''### Políticas, trajetórias e animações finais

Menor seed de treino (3100), reset 900000 previamente fixado, mesmas condições iniciais entre métodos. Não é seleção da melhor política. Mapas Q usam as fronteiras físicas das células; −1 na política significa empate. Uma trajetória não estima a taxa média de sucesso.'''),
             code('''for disc in ("4x2","12x4"):
    for alg in ("q_learning","sarsa_lambda","dyna_q"):
        name=f"{alg}__{disc}__s3100"
        display(Image(verified(f"figuras/animacoes_final_v2/{name}.policy.png"),width=1050))
        display(Image(verified(f"figuras/animacoes_final_v2/{name}.trajectory.png"),width=1000))
    path=OUT/f"final_comparacao_{disc}.mp4"
    path.write_bytes(verified(f"figuras/animacoes_final_v2/comparacao_{disc}.mp4"))
    display(Video(str(path),embed=True,width=1050))'''),
             md('''### Interpretação e limites

As diferenças valem para este protocolo, estas grades e os hiperparâmetros globais escolhidos. O efeito isolado de aliasing não foi identificado. Sensibilidade a N vem da triagem, não de novos ajustes finais. IC t com dez seeds é aproximado; conferir pontos individuais, mediana e DP. Comparações inferenciais AUC usam Friedman e Wilcoxon pareado com Holm por grade, conforme protocolo; sucesso avaliado é relatado descritivamente. Duração condicional omite políticas sem sucesso e traz seu denominador próprio. Latência e wall-clock dependem da máquina/carga.

Resultado/Discussão completos e condições de reprodução estão no relatório técnico público.''')]
        provenance=md('''## Apêndice: proveniência da triagem

As 108 execuções históricas abaixo foram preservadas. Esta tabela é **exploratória** e não integra as 60 execuções finais. Todos os gráficos e vídeos exibidos acima são da campanha final; figuras de sensibilidade a N da triagem permanecem disponíveis no repositório, sem atribuição ao experimento final.''')
        nb.cells=nb.cells[:3]+new+[provenance,nb.cells[4],md('''## Exportação e reprodução

Campanha final concluída e validada; nenhum retreino é necessário para apresentar. Scripts e bundles completos permitem reprodução. O Colab hospedado não foi executado por esta máquina; a validação usa kernel local limpo e downloads públicos.'''),nb.cells[-1]]
        nb.cells[-1].source=nb.cells[-1].source.replace('report = verified("relatorios/parecer_liberacao_final.md")',
            'academic = verified("relatorios/final_v2/RELATORIO_TECNICO.md")\n(OUT/"RELATORIO_TECNICO.md").write_bytes(academic)\nreport = verified("relatorios/parecer_liberacao_final.md")')
        nb.cells[0].source=nb.cells[0].source.replace('O notebook apresenta primeiro a proveniência da triagem e depois os resultados finais, sem misturar amostras.',
            'Todos os gráficos e vídeos exibidos são da campanha final. A proveniência da triagem aparece em apêndice, sem misturar amostras.')
        new[-1].source += '\n\nIC degenerado (DP zero, todas as políticas com 0% ou 100%) descreve esta amostra, não prova probabilidade populacional exatamente zero ou um. IC t não truncados estão na tabela; limites fora de [0,1] evidenciam a aproximação.'
    for cell in nb.cells:
        if cell.cell_type=='markdown':cell.source=cell.source.replace('\\\\','\\')
    import sys
    if '--main' in sys.argv:
        for cell in nb.cells:
            if cell.cell_type=='code':cell.source=cell.source.replace('BRANCH = "codex/cmp629-fase-a-20261011"','BRANCH = "main"')
    target=ROOT/'notebooks/MountainCar_CMP629_Colab.ipynb'
    target.parent.mkdir(exist_ok=True)
    nbf.validate(nb); nbf.write(nb,target)
    print(target)


if __name__=='__main__':main()
