# Auditoria independente — MountainCar (11/10/2026)

## O que foi efetivamente verificado

- CSV de triagem: **108 linhas distintas**, hashes únicos e taxa de sucesso no intervalo [0,1].
- Delineamento: 3 algoritmos × 6 configurações por algoritmo × 2 discretizações × 3 seeds = 108.
- A escolha de hiperparâmetros obedece à maior AUC média de treinamento, **agregando as discretizações**, com desempate segundo o protocolo original.
- `validacao_resultados.json` informa 108/108 válidos; os 108 arquivos JSON individuais **não vieram nos pacotes**.
- O código `EmpiricalTransitionModel` recebido usa contador de múltiplos desfechos por par (estado discretizado, ação), em vez de sobrescrever pela última transição.
- Teste isolado 70/30 de distribuição empírica retornou 0,7015 para o primeiro desfecho; alvos de TD terminal e não terminal passaram.
- Não foi executado treinamento integrado do Gymnasium neste ambiente de auditoria.

## Resultado das configurações selecionadas

| Algoritmo | Discretização | AUC de treinamento | Sucesso final de treino | Sucesso na avaliação |
|---|---|---:|---:|---:|
| SARSA(λ) | 12×4 | 0,08972 | 99,67% | **100,00%** |
| SARSA(λ) | 4×2 | 0,09108 | 54,33% | 63,33% |
| Q-learning | 12×4 | 0,03016 | 32,00% | 22,67% |
| Q-learning | 4×2 | 0,06293 | 40,00% | 62,00% |
| Dyna-Q | 12×4 | 0,03510 | 44,00% | 51,33% |
| Dyna-Q | 4×2 | 0,00000 | 0,00% | **0,00%** |

As médias têm apenas **3 seeds de treinamento independentes** por célula, apesar de 50 episódios de avaliação por política. Evitar pseudorreplicação.

**Dyna-Q 4×2:** 0% de sucesso em todas as 18 execuções de triagem referentes à discretização 4×2. *State aliasing* e efeitos do modelo são **hipóteses**, não causas estabelecidas.

## Problemas P0

1. `stable_episode` sinaliza o **primeiro atingimento** de limiar em janelas consecutivas, não desempenho sustentado até o final. Em 10 de 18 execuções selecionadas, o limiar foi atingido mas não persistiu na janela final.
2. `final_window_success` é treino **com exploração**; `evaluation_success_rate` é teste da política.
3. Os gráficos originais de heatmaps tinham escalas inconsistentes. As novas visualizações devem ter limites globais da mesma métrica.
4. `total_q_updates` usa unidades distintas entre métodos: SARSA(λ) contabiliza coordenadas não nulas do traço, Q-learning e Dyna-Q contabilizam atualizações escalares.

## Próximo gate

- Testes de invariantes do Dyna-Q (`N=0` versus Q-learning sob parâmetros/seed idênticos).
- Conferir avaliação sem ruído de exploração, registrar retorno por episódio.
- Manter dados da triagem imutáveis; diferenciar análises exploratórias e confirmatórias.
- Regenerar as figuras com escalas corretas.
- **Não executar o experimento final até satisfazer os gates.**

## Proveniência

Síntese baseada em `PACOTE_REVISAO_TRIAGEM(1).zip`, `PACOTE_AUDITORIA_DYNA(1).zip` e auditoria independente produzida em 11/10/2026. O código disponibilizado era um *snapshot parcial* (módulos `config.py`, `storage.py`, scripts de runner e testes originais não vieram na exportação). Não representar esse snapshot como repositório integral executável.


## Publicação dos dados
O arquivo `resumo_execucoes.csv.gz` disponível no GitHub é uma **cópia derivada sem a coluna `config_hash`**, reduzida para publicação e armazenamento compacto. As 108 linhas e os valores numéricos das demais colunas foram preservados, sem arredondamento. A identidade pode ser verificada pelo conjunto (algoritmo, discretização, seed, α, λ, epsilon_decay, N). O CSV original com hashes está no pacote local de revisão, ainda não incorporado individualmente ao repositório.