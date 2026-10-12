# Checklist da entrega — 12/10/2026

- [x] 108 JSONs históricos preservados; manifesto completo local de 121 arquivos conferido.
- [x] 60 JSONs finais, 60 NPZs e 120 CSVs brutos verificados, sem novo treinamento.
- [x] Protocolo SHA-256 `fb1deb3ea35f1323438c10878f24225dfdb6fdebcd4c12cf50d0e714d2cf4094` confirmado.
- [x] Tabelas dos gráficos reconciliadas com as 60 execuções reais.
- [x] 12 gráficos científicos finais em PNG 300 DPI/PDF/SVG; seis mapas Q/política e seis trajetórias em PNG 300 DPI.
- [x] Seis MP4s individuais e duas comparações; checkpoints vinculados por hash, ações greedy conferidas e decodificação completa.
- [x] Relatório: médias, medianas, DP, IC entre seeds, denominadores condicionais, inferência AUC e limitações.
- [x] Notebook final: seis células de código executadas em ambiente Python novo, kernel novo e diretório vazio; zero exceções; treinamento desativado.
- [x] Preview HTML inspecionado: equações, tabelas, gráficos e mapas renderizados; dois vídeos de 1800×400, 6,667 s, prontos sem erro no navegador local.
- [x] Downloads públicos da branch: 203 arquivos conferidos por SHA-256 antes do commit com outputs do notebook; verificação final é registrada em `verificacao_publicacao.json`.
- [x] Monte Carlo sem alteração: árvore Git de referência `d6b36978a408b4fe71fb38e0b303842d2d86986f`.

Pacotes públicos: `dados/auditoria_v2/triagem_originais.zip`, `final_v2_bundles.zip`, históricos comprimidos e `manifesto_publicacao.json`. Os bundles são ZIPs pequenos (final ≈3,3 MB), dispensando LFS. Os CSVs/NPZs dentro do ZIP preservam os bytes originais.

Limitação explícita: Google Colab **hospedado não foi executado**. O que foi validado é o notebook com kernel local limpo, usando exclusivamente URLs públicas e hashes. Suíte anterior: 33 testes aprovados, não repetidos nesta rodada. Treinos novos nesta rodada: zero.
