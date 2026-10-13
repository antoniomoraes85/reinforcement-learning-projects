# Atualização — entrega final de 13/10/2026

Notebook, PDF e pacote portátil publicados em [entrega_final](entrega_final). Consulte [INTEGRACAO_GITHUB.md](entrega_final/INTEGRACAO_GITHUB.md) para o estado desta publicação. Resultados e treinamentos anteriores preservados. Colab hospedado ainda não testado nesta rodada.

---

# Estado canônico — CMP629 MountainCar

## Escopo e autorização
Em 11/10/2026, a ordem de finalização integral autorizou os 60 treinamentos após GO e publicação via branch/PR, inclusive integração após verificação. Substitui a espera por aprovação intermediária. Não autoriza apagar históricos ou publicar material protegido.

## Fontes e estado preservado
108 JSONs individuais da triagem localizados e validados. Manifesto SHA-256 em `relatorios/manifesto_historico_sha256.json`: 121 arquivos históricos. Nenhum treino da triagem foi repetido. O código anterior está no commit local `b25451d`; as alterações anteriores do usuário foram preservadas nesse snapshot. Pasta original não pertencia a repositório Git.

## Implementado
Branch local `codex/cmp629-fase-a`; clone de publicação isolado em `.publication`, branch `codex/cmp629-fase-a-20261011`. Monte Carlo intocado.
TD/traços/modelo empírico testados; checkpoint NPZ com SHA/config; avaliação comum v2; contadores separados; cache sem sobrescrita; logs; scripts de figuras/rollouts/Colab.
25 testes aprovados na emissão do gate. Seis smokes de 5 episódios executados em 10,906 s com dois workers, em `resultados/auditoria_v2`; não são execuções finais.

## Campanha final
`configs/protocolo_final_v2.json`, saída `resultados/cmp629-final-v2-20261011`.
60 configurações: 3 métodos × 2 grades × 10 seeds. Hiperparâmetros globais da triagem congelados. Treino: 1000 episódios; avaliação: 100 seeds comuns (800000–800099). Seed de exibição 900000, menor seed de treino; regra definida antes de assistir aos vídeos.
**Campanha concluída e validada: 60/60 execuções, 60 checkpoints, 120 CSVs; protocolo SHA-256 `fb1deb3ea35f1323438c10878f24225dfdb6fdebcd4c12cf50d0e714d2cf4094`.** Rodada de 12/10/2026 restrita à entrega: nenhum treino ou teste exploratório repetido. Os 240 arquivos brutos foram reconciliados e têm manifesto próprio.

## Decisões metodológicas
3 ações, TimeLimit 200, γ=0,99. Bootstrap em truncamento, não em terminal real. Traços acumulativos reiniciados a cada episódio. Modelo Dyna distribui desfechos empiricamente. Desempates greedy aleatórios reprodutíveis. Não inferir causalidade de state aliasing. `stable_episode` legado significa primeiro atingimento do limiar; `total_q_updates` tem unidades heterogêneas.

## Próxima ação
Publicar e verificar os artefatos finais na branch de entrega; validar notebook em kernel limpo local; integrar o PR #2 somente após os checks de entrega. Não há treinamento pendente. Colab hospedado não foi executado.

## Histórico
- 11/10/2026: inventário, baseline Git, manifesto, 25 testes, seis smokes, GO, início autorizado da campanha final. Figuras históricas corrigidas em pasta nova; oito MP4s de smoke reais produzidos (seis individuais e duas comparações), sem atribuição às políticas antigas.
- 12/10/2026: campanha final concluída antes desta rodada; 60/60 confirmados. Entrega revisada: 24 PNGs finais (12 gráficos, seis mapas Q/política, seis trajetórias) e oito MP4s verificados; seis trajetórias greedy conferidas com os checkpoints. Relatório complementado com IC, medianas e denominadores condicionais. Suíte prévia de 33 testes preservada, sem repetição.
