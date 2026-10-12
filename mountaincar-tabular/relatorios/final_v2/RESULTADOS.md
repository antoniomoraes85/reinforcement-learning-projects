# Campanha final v2 — resultados reais

| Algoritmo | Grade | Seeds | Sucesso avaliação | DP entre seeds | Passos com falhas | Passos sucessos* | Treino final | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dyna_q | 12x4 | 10 | 31.80% | 46.70% | 190.01 | 169.64 | 44.10% | 0.03456 |
| dyna_q | 4x2 | 10 | 0.00% | 0.00% | 200.00 | não definido | 0.00% | 0.00000 |
| q_learning | 12x4 | 10 | 73.60% | 39.77% | 167.06 | 156.21 | 44.30% | 0.03089 |
| q_learning | 4x2 | 10 | 38.20% | 49.41% | 184.15 | 158.56 | 31.00% | 0.03901 |
| sarsa_lambda | 12x4 | 10 | 100.00% | 0.00% | 148.23 | 148.23 | 99.60% | 0.09849 |
| sarsa_lambda | 4x2 | 10 | 83.70% | 33.90% | 161.77 | 153.19 | 72.80% | 0.09103 |

* Média dos valores condicionais por política com ao menos um sucesso. Conferir `n_seeds` na tabela longa; políticas sem sucesso não têm duração condicional. A duração incluindo falhas inclui todas as políticas.
