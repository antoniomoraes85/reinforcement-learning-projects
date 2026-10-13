# CMP629 — MountainCar tabular

Comparação de Q-learning, SARSA(λ) e Dyna-Q em MountainCar-v0, com grades 4×2 e 12×4.

## Entrega final consolidada — 13/10/2026

- [Notebook obrigatório, com dados e vídeos incorporados](entrega_final/CMP629_MOUNTAINCAR_ENTREGA_FINAL.ipynb)
- [Abrir notebook final no Google Colab](https://colab.research.google.com/github/antoniomoraes85/reinforcement-learning-projects/blob/main/mountaincar-tabular/entrega_final/CMP629_MOUNTAINCAR_ENTREGA_FINAL.ipynb)
- [Relatório acadêmico definitivo — PDF](entrega_final/CMP629_MOUNTAINCAR_RELATORIO_DEFINITIVO.pdf)
- [Baixar pacote completo — ZIP](https://github.com/antoniomoraes85/reinforcement-learning-projects/raw/refs/heads/main/mountaincar-tabular/entrega_final/CMP629_MOUNTAINCAR_PACOTE_ENTREGA.zip)
- [Validação](entrega_final/VALIDACAO_ENTREGA.json) · [Notas de integração e limites](entrega_final/INTEGRACAO_GITHUB.md)

Notebook final: 15 células executadas sem erro em processo Python novo com IPython, sem novos treinamentos. PDF: 36 páginas, 16 figuras, apresentação ABNT, percentis entre seeds e apêndice de código. O teste no Colab hospedado permanece pendente. Os arquivos abaixo preservam o histórico anterior; para submissão, use a entrega consolidada acima.

## Estado verificável

**Campanha final concluída: 60/60 execuções válidas, 60 checkpoints e 120 CSVs brutos.** Dez seeds por algoritmo/grade, 100 avaliações comuns por política. Triagem: **108 JSONs reais preservados**. Gate FASE A: GO (25 testes); suíte adicional anterior: 33 testes aprovados. Os seis smokes técnicos são separados e não integram a campanha final. Nenhum treinamento foi executado na rodada de entrega de 12/10/2026.

- [Estado canônico](STATUS_CMP629.md)
- [Parecer técnico](relatorios/parecer_liberacao_final.md)
- [Relatório acadêmico final](relatorios/final_v2/RELATORIO_TECNICO.md)
- [Resultados quantitativos](relatorios/final_v2/RESULTADOS.md) · [Média, mediana, DP e IC entre seeds](relatorios/final_v2/estatisticas_entre_seeds.csv)
- [Gráficos finais](figuras/final_v2) · [Vídeos finais](figuras/animacoes_final_v2)
- [Comparação 4×2 — MP4](figuras/animacoes_final_v2/comparacao_4x2.mp4) · [Comparação 12×4 — MP4](figuras/animacoes_final_v2/comparacao_12x4.mp4)
- [PowerShell e reprodução](COMO_EXECUTAR_POWERSHELL.md)
- [Roteiro para o grupo](docs/ROTEIRO_APRESENTACAO.md)
- [Google Colab — branch da entrega](https://colab.research.google.com/github/antoniomoraes85/reinforcement-learning-projects/blob/codex/cmp629-fase-a-20261011/mountaincar-tabular/notebooks/MountainCar_CMP629_Colab.ipynb)

## Arquitetura e métricas

`src/mountaincar_rl`: agentes, configuração, persistência, análise. `testes`: testes TD/modelo/ambiente/reprodução/cache. `scripts`: execução, validação, gráficos e vídeos. `configs/protocolo_final_v2.json`: protocolo congelado. NPZ contém Q e configuração, sem pickle; hashes vinculam checkpoints e CSVs ao JSON da execução.

Uma interação real = um `env.step`. Um TD real = um erro TD processado, mesmo quando SARSA atualiza várias coordenadas. Planejamento Dyna e componentes dos traços têm contadores próprios; não somá-los como operações equivalentes entre métodos. Tempo de treino exclui avaliação/renderização. Latência mede lookup/argmax/desempate, sem discretização/renderização; depende do hardware.

`stable_episode` histórico é **primeiro atingimento do limiar**, não estabilidade sustentada. Sucesso final do treino é exploratório, avaliação é greedy. Bootstrap é mantido no TimeLimit e zerado no terminal real. Não é formulação exata de horizonte finito com tempo no estado.

## Dados públicos e checkpoints

`dados/auditoria_v2/triagem_originais.zip` preserva os 108 JSONs e resumos científicos. `triagem_historicos.json.gz` é uma serialização comprimida derivada para o notebook; valores não alterados. Os logs operacionais locais não são publicados. `manifesto_publicacao.json` registra SHA-256 dos arquivos públicos.

[final_v2_bundles.zip](dados/auditoria_v2/final_v2_bundles.zip) contém os 60 JSONs, 60 NPZs e 120 CSVs, preservados byte a byte. Para reprodução local, extrair com `scripts/extrair_bundles.py`; não misturar com `resultados/triagem`. O SHA-256 do protocolo canônico é `fb1deb3ea35f1323438c10878f24225dfdb6fdebcd4c12cf50d0e714d2cf4094`. `manifesto_final_preservado.json` identifica os 240 arquivos originais.

## Resultados principais (descritivos)

| Método | Sucesso 4×2 | Sucesso 12×4 |
|---|---:|---:|
| Q-learning | 38,2% | 73,6% |
| SARSA(λ) | 83,7% | 100,0% |
| Dyna-Q | 0,0% | 31,8% |

Média de dez políticas por célula, não episódios independentes. SARSA apresentou a maior média neste protocolo; isso não é superioridade universal. Há forte variabilidade entre seeds (por exemplo, DP de 46,70 pontos percentuais em Dyna-Q 12×4). Consultar IC, medianas e limitações no relatório. DP zero em 100% de sucesso observado não garante sucesso em novas seeds.

## Visualização

`figuras/final_v2`: resultados das 60 execuções; `figuras/animacoes_final_v2`: seis vídeos individuais e duas comparações, vinculados aos checkpoints verificados. O notebook exibe somente gráficos/vídeos finais; triagem aparece em apêndice de proveniência. `figuras/auditoria_v2` preserva análises exploratórias e `figuras/animacoes_auditoria_v2` preserva o **smoke técnico**. A triagem original não salvava checkpoints; não recriamos vídeos históricos. Política de exibição: menor seed de treino (3100), reset 900000; falhas visíveis. No caso demonstrado, SARSA atinge o objetivo nas duas grades; Q-learning e Dyna-Q falham nessa seed, sem seleção de melhor caso.

Modo padrão do notebook não treina. `RUN_TRAINING` e `RUN_DEMO` ficam falsos. Colab hospedado não foi executado por esta máquina; validação usa kernel local limpo e downloads públicos. Não alegar superioridade geral nem causalidade do state aliasing. Material da disciplina/PDFs, credenciais e projeto Monte Carlo não são alterados/publicados por esta entrega.
