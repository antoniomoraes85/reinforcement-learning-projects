"""Write academic report only from validated final evidence."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'relatorios/final_v2'
    validation=json.loads((out/'validacao_final.json').read_text(encoding='utf-8'))
    assert validation['valid_runs']==60
    rows=pd.read_csv(out/'metricas_por_seed.csv')
    table=(out/'RESULTADOS.md').read_text(encoding='utf-8')
    inference=json.loads((out/'inferencia_auc.json').read_text(encoding='utf-8'))
    tests=[]
    for disc,item in inference['by_discretization'].items():
        f=item['friedman']
        tests.append(f"### Grade {disc}\n\nFriedman AUC: estatística={f['statistic']}, p={f['p_value']} ({f['status']}).\n")
        tests.append('| Comparação AUC | p bruto | p Holm (por grade) | Rank-biserial | Diferença média | IC bootstrap 95% |\n|---|---:|---:|---:|---:|---|')
        for pair in item['post_hoc']:
            tests.append(f"| {pair['left']} − {pair['right']} | {pair['p_raw']:.5g} | {pair['p_holm']:.5g} | {pair['rank_biserial_left_minus_right']:.3f} | {pair['mean_difference_left_minus_right']:.5f} | [{pair['bootstrap_ci95_low']:.5f}, {pair['bootstrap_ci95_high']:.5f}] |")
    training_seconds=rows.training_time_seconds.sum()
    observations=[]
    for disc,g in rows.groupby('discretization'):
        means=g.groupby('algorithm').evaluation_success_rate.mean().sort_values(ascending=False)
        observations.append(f"Na grade {disc}, a maior média observada de sucesso na avaliação foi {means.iloc[0]:.2%} ({means.index[0]}). Isso descreve estas dez políticas por método, não estabelece superioridade universal.")
    report=f'''# Comparação tabular no MountainCar-v0 — CMP629

## Resumo
Foram preservados 108 treinamentos exploratórios e executados 60 treinamentos finais independentes (três algoritmos, duas discretizações, dez seeds). Q-learning, SARSA(λ) e Dyna-Q foram comparados com hiperparâmetros globais selecionados exclusivamente pela AUC de treino da triagem. A avaliação final empregou 100 resets comuns por política, sem exploração. Dados por episódio, contadores, tempos e checkpoints foram salvos e validados. As conclusões são restritas ao ambiente e protocolo estudados.

## 1. Problema e questão experimental
Como os três métodos tabulares se comportam nas grades 4×2 e 12×4, sob parâmetros globais previamente selecionados? MountainCar exige ganhar embalo para atingir o objetivo. O benchmark permanece com três ações e limite de 200 passos. A representação grosseira pode perder informação, mas este experimento não identifica causalmente o efeito de aliasing.

## 2. Métodos
Q-learning é off-policy/model-free. SARSA(λ) é on-policy/model-free e propaga TD por traços acumulativos, reiniciados a cada episódio. Dyna-Q combina um backup real com N=5 backups do modelo empírico de múltiplos desfechos. Seeds independentes de treino, avaliação comum e desempates reprodutíveis foram controlados. Bootstrap permanece em truncamento temporal e é zerado no terminal real; trata-se do processo subjacente, não do MDP de horizonte finito com tempo no estado.

Treino: 1000 episódios/política, γ=0,99, ε inicial 1 e mínimo 0,02. Q: α=0,1, decay=0,99; SARSA: α=0,02, λ=0,9, decay=0,995; Dyna: α=0,05, N=5, decay=0,995. Limiares exatos e seeds constam em `configs/protocolo_final_v2.json`. Nenhum parâmetro foi selecionado novamente a partir dos testes finais. Metodologia detalhada e fórmulas em `docs/METODOLOGIA_FINAL.md` e no notebook.

## 3. Métricas e análise
Sucesso na avaliação; retorno episódico; passos com e sem condicionamento ao sucesso; AUC de treino; sucesso na janela final exploratória; interações reais, TD reais, planejamento, componentes de traços, tempo de treino e latência de seleção da ação. Cada seed de treinamento é uma unidade; as 100 avaliações não são replicações de treinamento. Tabela longa contém média/mediana/DP/IC t entre seeds e denominadores. Curvas interpoladas no eixo de interações não extrapolam observações.

## 4. Resultados observados
{table}

{chr(10).join(observations)}

### Comparações inferenciais da AUC de treinamento
Não confundir estes testes com testes de taxa de sucesso de avaliação. Holm controla três comparações por grade; não uma família global englobando as duas grades. Bootstrap é pareado por seed, 10000 reamostragens com seed fixa.

{chr(10).join(tests)}

## 5. Custo computacional
Soma do tempo de treinamento individual: {training_seconds:.2f} s ({training_seconds/60:.2f} min). Não é wall-clock da campanha paralela. Total de interações reais: {int(rows.total_env_steps.sum()):,}; backups simulados Dyna: {int(rows.planning_updates.sum()):,}; componentes dos traços SARSA: {int(rows.trace_components.sum()):,}. Essas unidades não são intercambiáveis. Dois workers foram usados; tempos incluem efeitos de carga local e não constituem benchmark isolado de CPU. Latência exclui renderização e transição do ambiente.

## 6. Políticas e demonstrações
Vídeos finais usam checkpoints reais da menor seed de treino (3100), avaliados no reset 900000 definido antes da visualização. Comparações lado a lado usam a mesma condição inicial, mantêm identificação do algoritmo/grade/seeds/ação/retorno e exibem fracassos. São demonstrações, não estimativas de desempenho médio. Vídeos de smoke têm diretório e rótulo próprios. Não existem checkpoints das políticas históricas da triagem; elas não foram reconstruídas.

## 7. Limitações e ameaças à validade
Amostra de dez seeds, duas grades, um único benchmark, resets estreitos e hiperparâmetros globais. A discretização pode violar a propriedade de Markov. Não houve ablação capaz de atribuir causalidade ao modelo ou aliasing. Critério de primeiro atingimento não implica estabilidade sustentada; avaliar a janela final separadamente. IC t é aproximado, pode ultrapassar faixas físicas; consultar números não truncados da tabela longa. Duração condicionada é indefinida para políticas sem sucesso; o denominador está explicitado. Esquemas de seeds da triagem e final diferem; não os agrupar. Tempos são dependentes do hardware e da carga. Colab hospedado não foi executado nesta auditoria; há validação com kernel limpo local e downloads públicos.

## 8. Conclusão
O experimento oferece uma comparação reproduzível dos métodos sob o protocolo congelado, não um ranking geral de algoritmos. A evidência combina desempenho e variabilidade entre políticas, com custos em unidades explícitas. Discussões de causalidade ou transferência para controle MPC/RL demandam novos experimentos: esta entrega serve de referência metodológica, não de evidência direta sobre sistemas físicos.

## 9. Reprodutibilidade e artefatos
`validacao_final.json`: 60/60 bundles validados, 60 NPZs e 120 CSVs; JSON por execução vincula configuração, versions, commit, datas e hashes. `metricas_por_seed.csv`, `estatisticas_entre_seeds.csv`, `inferencia_auc.json`: resultados inspectáveis. `figuras/final_v2` e `figuras/animacoes_final_v2`: gráficos/rollouts reais. Manifesto SHA-256 preserva os históricos; extração pública não sobrescreve arquivos divergentes. Testes TD/traços/Dyna/seeds/cache e reavaliação dos checkpoints completam as verificações.
'''
    (out/'RELATORIO_TECNICO.md').write_text(report,encoding='utf-8')
    print('Academic report generated from validated 60-run evidence')


if __name__=='__main__':main()
