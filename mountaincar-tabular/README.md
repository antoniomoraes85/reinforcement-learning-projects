# CMP629 — MountainCar tabular

Comparação de Q-learning, SARSA(λ) e Dyna-Q em MountainCar-v0, com grades 4×2 e 12×4.

## Estado verificável

Triagem: **108 JSONs reais preservados**, três seeds por configuração. Gate FASE A: **GO**, 25 testes aprovados; seis smokes técnicos de cinco episódios. Campanha final de **60 treinamentos autorizada e em execução**; só tratar como concluída após `relatorios/final_v2/validacao_final.json` confirmar 60/60. Smoke não é campanha final.

- [Estado canônico](STATUS_CMP629.md)
- [Parecer técnico](relatorios/parecer_liberacao_final.md)
- [PowerShell e reprodução](COMO_EXECUTAR_POWERSHELL.md)
- [Roteiro para o grupo](docs/ROTEIRO_APRESENTACAO.md)
- [Google Colab — branch da entrega](https://colab.research.google.com/github/antoniomoraes85/reinforcement-learning-projects/blob/codex/cmp629-fase-a-20261011/mountaincar-tabular/notebooks/MountainCar_CMP629_Colab.ipynb)

## Arquitetura e métricas

`src/mountaincar_rl`: agentes, configuração, persistência, análise. `testes`: testes TD/modelo/ambiente/reprodução/cache. `scripts`: execução, validação, gráficos e vídeos. `configs/protocolo_final_v2.json`: protocolo congelado. NPZ contém Q e configuração, sem pickle; hashes vinculam checkpoints e CSVs ao JSON da execução.

Uma interação real = um `env.step`. Um TD real = um erro TD processado, mesmo quando SARSA atualiza várias coordenadas. Planejamento Dyna e componentes dos traços têm contadores próprios; não somá-los como operações equivalentes entre métodos. Tempo de treino exclui avaliação/renderização. Latência mede lookup/argmax/desempate, sem discretização/renderização; depende do hardware.

`stable_episode` histórico é **primeiro atingimento do limiar**, não estabilidade sustentada. Sucesso final do treino é exploratório, avaliação é greedy. Bootstrap é mantido no TimeLimit e zerado no terminal real. Não é formulação exata de horizonte finito com tempo no estado.

## Dados públicos e checkpoints

`dados/auditoria_v2/triagem_originais.zip` preserva os 108 JSONs e resumos científicos. `triagem_historicos.json.gz` é uma serialização comprimida derivada para o notebook; valores não alterados. Os logs operacionais locais não são publicados. `manifesto_publicacao.json` registra SHA-256 dos arquivos públicos.

Após conclusão, `final_v2_bundles.zip` conterá os 60 JSONs, 60 NPZs e 120 CSVs. Para reprodução local, extrair na pasta indicada em `COMO_EXECUTAR_POWERSHELL.md`; não misturar com `resultados/triagem`.

## Visualização

`figuras/auditoria_v2`: desempenho histórico, escalas comuns, curvas reais, sensibilidade/custos. `figuras/animacoes_auditoria_v2`: **smoke técnico**, não políticas históricas. A triagem original não salvava checkpoints; não recriamos nem atribuimos vídeos a ela. Política de exibição: menor seed de treino, reset 900000; falhas visíveis.

Modo padrão do notebook não treina. `RUN_TRAINING` e `RUN_DEMO` ficam falsos. Colab hospedado não foi executado por esta máquina; validação usa kernel local limpo e downloads públicos. Não alegar superioridade geral nem causalidade do state aliasing. Material da disciplina/PDFs, credenciais e projeto Monte Carlo não são alterados/publicados por esta entrega.
