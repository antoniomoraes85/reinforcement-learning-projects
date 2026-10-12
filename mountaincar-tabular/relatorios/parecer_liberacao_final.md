# Parecer de liberação final — FASE A

Data: 11/10/2026. **GO técnico para a campanha final v2.** Autorização explícita recebida na ordem de finalização integral. Não é aprovação acadêmica nem conclusão sobre superioridade de métodos.

## Evidências executadas
- Suíte pytest: **25 testes aprovados, 0 falhas** (`testes_fase_a.xml`).
- Validador: 108/108 JSONs da triagem; zero ausentes/corrompidos. Manifesto SHA-256: 121 arquivos históricos, preservados byte a byte.
- Smoke: seis configurações (três métodos × duas discretizações), cinco episódios cada, dois workers; **10.906 s** wall-clock. Não são seis execuções finais.
- Seis checkpoints NPZ carregados e hashes conferidos; CSVs/JSONs gravados separadamente. Cache completo testado sem retreino; parcial inválido preservado e rejeitado.
- Ambiente, limites e seeds; TD manual Q/SARSA/Dyna; traços acumulativos e reset; terminal versus truncamento; modelo empírico 70/30 com seed fixa; Dyna N=0 equivalente a Q-learning; reprodução seed-a-seed; política greedy e desempates; avaliação não altera Q.

## Correções
Checkpoints verificáveis, avaliação v2 com 100 seeds comuns fixas, logs de retorno/término/epsilon, separação de TD real/planejamento/componentes dos traços e latência. Dyna já armazenava múltiplos desfechos: nenhum bug de sobrescrita foi inventado. Escalas e nomenclatura corrigidas nas figuras novas, sem alterar CSVs antigos. `total_q_updates` mantido apenas por compatibilidade; não é custo homogêneo entre algoritmos.

## Decisão metodológica
Bootstrap em truncamento temporal; zero somente em terminal real. Trata-se do processo subjacente contínuo com avaliação limitada a 200 passos, não de solução exata de horizonte finito com tempo no estado. Traços zerados a cada episódio. Avaliação histórica e v2 têm esquemas distintos de seeds: não misturar campanhas como replicações idênticas.

## Estimativas medidas, não garantias
Smoke ocupa 45358 bytes (sem vídeos). Extrapolação linear conservadora: 90.7 MB; treino serial 137.2 min ou cerca de 68.6 min com dois workers, mais avaliação/inicialização. Cinco episódios são insuficientes para previsão precisa; convergência/carga variam. RAM será observada durante a campanha; não foi medida no primeiro smoke.

## Ressalvas não bloqueantes
Warnings de depreciação do Pyparsing/Matplotlib, sem falhas. Só três seeds por célula na triagem; não alegar superioridade geral. Não existem checkpoints históricos da triagem. Vídeos do smoke não representam desempenho final. Colab hospedado e publicação ainda em preparação na emissão deste gate.

## Comandos efetivamente executados
```powershell
& '..\.venv\Scripts\python.exe' -B -m pytest -p no:cacheprovider -q --junitxml=relatorios/testes_fase_a.xml
& '..\.venv\Scripts\python.exe' -B scripts/preservar_historico.py
& '..\.venv\Scripts\python.exe' -B scripts/preparar_fase_a.py
& '..\.venv\Scripts\python.exe' -B scripts/validar_resultados.py --fase triagem --relatorio relatorios/validacao_triagem_fase_a.json
& '..\.venv\Scripts\python.exe' -B scripts/gerar_figuras_auditadas.py
```

Ambiente testado: Python 3.12.14, Gymnasium 1.2.1, NumPy 2.3.3. A `.venv` interna (Python 3.14) não foi adotada; usamos a `.venv` do diretório pai, sem apagar nenhuma delas.
