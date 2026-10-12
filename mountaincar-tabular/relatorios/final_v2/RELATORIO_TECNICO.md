# Comparação tabular no MountainCar-v0 — CMP629

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
### Campanha final v2 — resultados reais

| Algoritmo | Grade | Seeds | Sucesso avaliação | DP entre seeds | Passos com falhas | Passos sucessos* | Treino final | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dyna_q | 12x4 | 10 | 31.80% | 46.70% | 190.01 | 169.64 | 44.10% | 0.03456 |
| dyna_q | 4x2 | 10 | 0.00% | 0.00% | 200.00 | não definido | 0.00% | 0.00000 |
| q_learning | 12x4 | 10 | 73.60% | 39.77% | 167.06 | 156.21 | 44.30% | 0.03089 |
| q_learning | 4x2 | 10 | 38.20% | 49.41% | 184.15 | 158.56 | 31.00% | 0.03901 |
| sarsa_lambda | 12x4 | 10 | 100.00% | 0.00% | 148.23 | 148.23 | 99.60% | 0.09849 |
| sarsa_lambda | 4x2 | 10 | 83.70% | 33.90% | 161.77 | 153.19 | 72.80% | 0.09103 |

* Média dos valores condicionais por política com ao menos um sucesso. Conferir `n_seeds` na tabela longa; políticas sem sucesso não têm duração condicional. A duração incluindo falhas inclui todas as políticas.


Na grade 12x4, a maior média observada de sucesso na avaliação foi 100.00% (sarsa_lambda). Isso descreve estas dez políticas por método, não estabelece superioridade universal.
Na grade 4x2, a maior média observada de sucesso na avaliação foi 83.70% (sarsa_lambda). Isso descreve estas dez políticas por método, não estabelece superioridade universal.

### Incerteza e variabilidade entre seeds

| Método | Grade | Média de sucesso | Mediana | DP (pontos percentuais) | IC t 95% da média, não truncado |
|---|---|---:|---:|---:|---:|
| Q-learning | 4×2 | 38,20% | 0,00% | 49,41 | [2,85%; 73,55%] |
| Q-learning | 12×4 | 73,60% | 88,50% | 39,77 | [45,15%; 102,05%] |
| SARSA(λ) | 4×2 | 83,70% | 100,00% | 33,90 | [59,45%; 107,95%] |
| SARSA(λ) | 12×4 | 100,00% | 100,00% | 0,00 | [100,00%; 100,00%] |
| Dyna-Q | 4×2 | 0,00% | 0,00% | 0,00 | [0,00%; 0,00%] |
| Dyna-Q | 12×4 | 31,80% | 1,50% | 46,70 | [−1,61%; 65,21%] |

Cada linha resume dez políticas. As grandes diferenças entre média e mediana em Q-learning 4×2 e Dyna-Q 12×4 evidenciam forte variabilidade; uma média isolada não representa bem uma política típica. IC t é uma aproximação não restrita ao intervalo [0,1]; valores externos não são probabilidades válidas. IC degenerado nas células com DP zero descreve a amostra observada e **não prova** sucesso ou fracasso certo em novas seeds. A tabela longa publica também IC de retorno, duração, AUC e custos. Para duração condicional, o número de políticas com sucesso é 4/10 (Q 4×2), 8/10 (Q 12×4), 9/10 (SARSA 4×2), 10/10 (SARSA 12×4), 0/10 (Dyna 4×2) e 6/10 (Dyna 12×4); conferir os denominadores antes de comparar.

### Comparações inferenciais da AUC de treinamento
Não confundir estes testes com testes de taxa de sucesso de avaliação. Holm controla três comparações por grade; não uma família global englobando as duas grades. Bootstrap é pareado por seed, 10000 reamostragens com seed fixa.

### Grade 12x4

Friedman AUC: estatística=15.200000000000017, p=0.0005004514334406062 (computed).

| Comparação AUC | p bruto | p Holm (por grade) | Rank-biserial | Diferença média | IC bootstrap 95% |
|---|---:|---:|---:|---:|---|
| q_learning − sarsa_lambda | 0.0019531 | 0.0058594 | -1.000 | -0.06760 | [-0.08457, -0.04887] |
| q_learning − dyna_q | 0.27539 | 0.27539 | -0.418 | -0.00367 | [-0.00997, 0.00186] |
| sarsa_lambda − dyna_q | 0.0019531 | 0.0058594 | 1.000 | 0.06392 | [0.04628, 0.08017] |
### Grade 4x2

Friedman AUC: estatística=16.80000000000001, p=0.00022486732417884692 (computed).

| Comparação AUC | p bruto | p Holm (por grade) | Rank-biserial | Diferença média | IC bootstrap 95% |
|---|---:|---:|---:|---:|---|
| q_learning − sarsa_lambda | 0.0097656 | 0.0097656 | -0.891 | -0.05202 | [-0.08837, -0.02188] |
| q_learning − dyna_q | 0.0019531 | 0.0058594 | 1.000 | 0.03901 | [0.02858, 0.04944] |
| sarsa_lambda − dyna_q | 0.0019531 | 0.0058594 | 1.000 | 0.09103 | [0.06333, 0.12382] |

## 5. Custo computacional
Soma do tempo de treinamento individual: 3492.42 s (58.21 min). Não é wall-clock da campanha paralela. Total de interações reais: 11,411,906; backups simulados Dyna: 19,654,480; componentes dos traços SARSA: 52,575,778. Essas unidades não são intercambiáveis. Dois workers foram usados; tempos incluem efeitos de carga local e não constituem benchmark isolado de CPU. Latência exclui renderização e transição do ambiente.

## 6. Políticas e demonstrações
Vídeos finais usam checkpoints reais da menor seed de treino (3100), avaliados no reset 900000 definido antes da visualização. Comparações lado a lado usam a mesma condição inicial, mantêm identificação do algoritmo/grade/seeds/ação/retorno e exibem fracassos. São demonstrações, não estimativas de desempenho médio. Vídeos de smoke têm diretório e rótulo próprios. Não existem checkpoints das políticas históricas da triagem; elas não foram reconstruídas.

## 7. Limitações e ameaças à validade
Amostra de dez seeds, duas grades, um único benchmark, resets estreitos e hiperparâmetros globais. A discretização pode violar a propriedade de Markov. Não houve ablação capaz de atribuir causalidade ao modelo ou aliasing. Critério de primeiro atingimento não implica estabilidade sustentada; avaliar a janela final separadamente. IC t é aproximado, pode ultrapassar faixas físicas; consultar números não truncados da tabela longa. Duração condicionada é indefinida para políticas sem sucesso; o denominador está explicitado. Esquemas de seeds da triagem e final diferem; não os agrupar. Tempos são dependentes do hardware e da carga. Colab hospedado não foi executado nesta auditoria; há validação com kernel limpo local e downloads públicos.

## 8. Conclusão
O experimento oferece uma comparação reproduzível dos métodos sob o protocolo congelado, não um ranking geral de algoritmos. A evidência combina desempenho e variabilidade entre políticas, com custos em unidades explícitas. Discussões de causalidade ou transferência para controle MPC/RL demandam novos experimentos: esta entrega serve de referência metodológica, não de evidência direta sobre sistemas físicos.

## 9. Reprodutibilidade e artefatos
`validacao_final.json`: 60/60 bundles validados, 60 NPZs e 120 CSVs; JSON por execução vincula configuração, versões, commit, datas e hashes. `metricas_por_seed.csv`, `estatisticas_entre_seeds.csv`, `inferencia_auc.json`: resultados inspectáveis. `figuras/final_v2` e `figuras/animacoes_final_v2`: gráficos/rollouts reais. Manifesto SHA-256 preserva os históricos; extração pública não sobrescreve arquivos divergentes. A entrega verifica hashes, reconciliação das tabelas e ações greedy das seis trajetórias contra seus checkpoints; não afirma nova reavaliação das 60 políticas. Nenhum treinamento foi executado na rodada de entrega de 12/10/2026.
