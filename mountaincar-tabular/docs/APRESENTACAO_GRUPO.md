# Roteiro de apresentação — mini-projeto MountainCar

**Objetivo:** mostrar o método, os resultados da triagem e as limitações verificadas. Não confundir com apresentação definitiva do trabalho.

## Sequência (estimativa: 6 a 8 minutos)

1. **Problema:** o carro parte de um vale e deve atingir a bandeira. A recompensa é −1 por passo, com limite usual de 200; ações 0/1/2.
2. **Questão experimental:** como algoritmo e resolução da discretização afetam qualidade de aprendizagem e custo computacional?
3. **Algoritmos:** Q-learning (model-free/off-policy/TD); SARSA(λ) (model-free/on-policy/TD com traços); Dyna-Q (model-based/off-policy/TD com planejamento simulado).
4. **Protocolo:** 2 discretizações, 6 hiperparâmetros por algoritmo, 3 seeds, 1.000 episódios por execução e 50 episódios de avaliação.
5. **Seleção:** maior AUC média *do treinamento* por algoritmo; **não** selecionar pelo teste.
6. **Resultados:** SARSA(λ) 12×4 teve 100% médio nas 3 seeds; Dyna-Q 4×2 teve 0% nas 18 execuções da triagem; Q-learning 4×2 superou 12×4 com a configuração global selecionada.
7. **Cautelas:** 3 seeds; sem dados brutos por episódio; possíveis variações por discretização; contagem Q não normalizada entre algoritmos.
8. **Próximos experimentos:** testes integrados Dyna-Q e protocolo final após validação.

## Perguntas prováveis

**Por que Dyna-Q falhou em 4×2?** Não há causa demonstrada. Testar aliasing de estado, modelo empírico e efeitos de N.

**Podemos afirmar que SARSA(λ) é melhor?** Não de forma geral. Somente há evidência preliminar neste protocolo de triagem.

**Por que a seleção usa AUC do treino e não sucesso de avaliação?** Para evitar usar o conjunto de avaliação na escolha de hiperparâmetros.

**108 execuções significam 108 seeds independentes?** Não. São 18 configurações/discretizações × 3 seeds; algumas condições têm apenas 3 replicações independentes.

**Os números são do experimento final?** Não. Esta é a triagem.

## Links

- [Notebook Colab](../notebooks/MountainCar_CMP629_Colab.ipynb)
- [Auditoria](AUDITORIA.md)
- [Resultados selecionados](../dados/triagem/selecionadas_por_discretizacao.csv)
