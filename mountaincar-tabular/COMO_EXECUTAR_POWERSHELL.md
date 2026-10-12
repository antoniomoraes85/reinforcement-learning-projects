# Execução e recuperação — PowerShell

Execute na raiz do mini-projeto. Python suportado/testado: 3.12. No computador original foi usado `..\.venv\Scripts\python.exe`; a `.venv` interna anterior é Python 3.14 e não foi modificada. Num clone limpo:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-entrega.txt
& '.\.venv\Scripts\python.exe' -m pytest testes -q
```

## Validação (não treina)

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/preservar_historico.py
& '.\.venv\Scripts\python.exe' -B scripts/analisar_campanha_v2.py
```

No clone público, extraia primeiro os bundles publicados conforme README; eles preservam arquivos e checksums. Não rode a triagem de novo. A verificação do manifesto histórico completo inclui arquivos locais auxiliares; os bundles de publicação incluem os 108 JSONs e os CSVs científicos, não logs pessoais.

## Campanha autorizada / retomada

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/executar_final.py --autorizado-apos-go --workers 2
```

Este comando exige o gate GO. Reutiliza bundles completos válidos com hashes de JSON/CSV/Q-table; não repete treinamentos completos. Os resultados permanecem em `resultados/cmp629-final-v2-20261011`. Um bundle parcial/corrompido é preservado e rejeitado, nunca silenciosamente sobrescrito. Se houver interrupção durante uma gravação, use o recuperador documentado após encerrar todos os workers da campanha. Não remova locks de um processo ativo. Dois lançamentos simultâneos não são permitidos.

## Figuras e animações (sem treino)

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/gerar_figuras_auditadas.py --dados resultados/cmp629-final-v2-20261011 --saida figuras/final_v2
& '.\.venv\Scripts\python.exe' -B scripts/animar_politicas.py --dados resultados/cmp629-final-v2-20261011 --saida figuras/animacoes_final_v2
```

Animações usam renderização `rgb_array`, backend SDL dummy, checkpoint verificado, seed de exibição fixa. Vídeos não entram nas medidas de tempo de treinamento ou inferência. MP4 com fallback GIF.

## Notebook

Modo apresentação (`RUN_TRAINING=False`, `RUN_DEMO=False`) baixa resultados públicos. A reprodução acadêmica consiste em dados/checkpoints/protocolo publicados; a demonstração opcional é nova e não replica os históricos. A execução hospedada no Google Colab exige internet e não foi certificada por execução remota nesta máquina.
