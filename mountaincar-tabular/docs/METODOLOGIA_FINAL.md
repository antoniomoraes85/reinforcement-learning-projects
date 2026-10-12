# Metodologia e ameaças à validade

## Delineamento
MountainCar-v0 (Gymnasium 1.2.1), 3 ações, TimeLimit 200, recompensa −1 a cada passo, γ=0,99. Estado tabular obtido por `numpy.digitize` das observações posição/velocidade; igualdade ao limiar entra no bin superior. Grades não uniformes: 4×2 e 12×4; limiares no protocolo.

Triagem histórica: 108 treinamentos, 1000 episódios, seeds 100/1100/2100, 50 avaliações por política; seis configurações por método, duas grades. Seleção global pela maior AUC de treino agregada às grades, com hierarquia de desempate do protocolo legado. Q-learning α=0,1, decay=0,99; SARSA α=0,02, λ=0,9, decay=0,995; Dyna α=0,05, N=5, decay=0,995. Não selecionar novamente pelos resultados finais.

Final: 60 treinamentos, 1000 episódios, dez seeds novas predefinidas (3100 a 12100 com incremento 1000). ε começa em 1, decai por episódio até mínimo 0,02. Seeds de reset de treino derivadas por SeedSequence; política e planejamento usam RNG da execução. Avaliação greedy usa 100 resets comuns (800000–800099) e RNG de desempate por episódio. O conjunto comum reduz diferenças de condições iniciais, mas não transforma episódios em replicações de treinamento.

## Algoritmos e modelo
Q-learning usa alvo greedy off-policy; SARSA(λ) usa próxima ação da política exploratória, traços acumulativos e reset de traços entre episódios. Dyna-Q usa Q-learning real e N backups de pares já visitados escolhidos uniformemente; o desfecho é sorteado proporcionalmente às contagens empíricas de `(recompensa, próximo estado, terminal)`. Não armazena só a última transição. Contagens são condicionadas a estados agregados e à distribuição de visitas induzida pela política: não garantem um modelo Markov exato.

Terminal real zera bootstrap; limite de tempo encerra a coleta mas mantém bootstrap. A avaliação continua limitada a 200 passos. Essa escolha estima o processo subjacente, não uma formulação exata de horizonte finito que inclua o tempo restante no estado. Não misturar essa decisão com a métrica de sucesso dentro de 200 passos.

## Métricas e unidades
- Retorno episódico e passos: ambos gravados; neste ambiente, retorno = −passos.
- Sucesso: sinal `terminated` de atingir objetivo; `truncated` é registrado separadamente e pode ocorrer junto de término no passo limite.
- Duração com falhas: todos os episódios; duração condicional: somente sucessos. Política sem sucesso recebe valor ausente, não zero nem 200 nessa métrica condicional.
- AUC normalizada: trapezóides da curva `(retorno+200)/200`, divididos por `n−1`; medida de aprendizagem de treino, não avaliação independente.
- Sucesso móvel: janela cheia de 20 episódios. Primeiro atingimento: cinco janelas consecutivas com sucesso ≥0,8; não certifica manutenção posterior. Janela final: 100 episódios exploratórios.
- Custo: interações reais; erros TD reais; backups de planejamento; componentes de traços atualizados; wall-clock de treino; latência de ação (lookup, argmax e desempate, sem env.step/renderização). Não comparar `total_q_updates` entre algoritmos como operação homogênea.

## Estatística
Unidade: seed de treinamento, dez políticas por célula final. Tabelas trazem média, mediana, DP e IC t 95% entre seeds. Duração condicional informa quantas políticas tiveram sucessos; pode ter n menor. Curvas usam média/IC entre seeds; no eixo de interações, interpolação no suporte real comum, sem extrapolação.

Comparação inferencial da AUC predefinida: Friedman por grade, seguido de Wilcoxon pareado por seed, correção Holm dentro da família de três comparações da grade, efeito rank-biserial e bootstrap pareado da diferença média (10000 reamostragens, seed fixa). Empates totais geram Friedman indefinido explicitamente; não um p-value fictício. Se interpretadas ambas grades conjuntamente, há uma família maior não coberta por essa correção; evitar alegações globais. Taxa de sucesso é descritiva com IC entre políticas, não teste binomial sobre 6000 episódios como se independentes.

## Ameaças e limites
Pequeno número de seeds, distribuição estreita de resets, apenas um ambiente, duas discretizações e hiperparâmetros globais. Seleção usa a triagem, que deve permanecer exploratória. Avaliação v2 usa seeds diferentes do procedimento histórico; não agrupar campanhas. Aliasing é plausível, não causa demonstrada; exige ablação própria. Comparar N na triagem não é isolar custo causal no final. IC t é aproximação e pode ultrapassar limites físicos; gráficos limitados devem declarar isso. Inferência/tempo dependem de hardware, carga e bibliotecas; não equivalem a FLOPs. O pipeline fez outras tarefas leves durante a campanha, portanto wall-clock é uma medição operacional, não benchmark isolado de desempenho computacional.

## Demonstrações
Menor seed de treinamento, reset 900000 fixado antes de ver as trajetórias. Mostrar falhas, sem seleção do melhor caso. Mapas/rollouts vêm de Q-tables verificadas; não há checkpoints históricos, logo não há vídeo autêntico das políticas da triagem. Vídeos não entram na medição de latência.

## Referências técnicas de base
Sutton e Barto, *Reinforcement Learning: An Introduction*, 2ª edição: TD, controle on/off-policy, Dyna e eligibility traces. Implementação do benchmark: código instalado do Gymnasium 1.2.1, `gymnasium/envs/classic_control/mountain_car.py`; wrappers de TimeLimit e protocolo local são as fontes verificadas para os parâmetros do experimento. Este documento não afirma cumprir um enunciado que não tenha sido relido nesta rodada.
