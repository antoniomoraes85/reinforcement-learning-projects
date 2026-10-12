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
Campanha iniciada após GO; atualizar este estado com a validação de conclusão. Nunca tratar resultados ainda ausentes como zero.

## Decisões metodológicas
3 ações, TimeLimit 200, γ=0,99. Bootstrap em truncamento, não em terminal real. Traços acumulativos reiniciados a cada episódio. Modelo Dyna distribui desfechos empiricamente. Desempates greedy aleatórios reprodutíveis. Não inferir causalidade de state aliasing. `stable_episode` legado significa primeiro atingimento do limiar; `total_q_updates` tem unidades heterogêneas.

## Próxima ação
Concluir e validar a campanha; analisar sem nova seleção; gerar animações finais; executar Colab em kernel limpo; publicar, verificar e integrar. Usar cache para preservar execuções completas em eventual retomada.

## Histórico
- 11/10/2026: inventário, baseline Git, manifesto, 25 testes, seis smokes, GO, início autorizado da campanha final. Figuras históricas corrigidas em pasta nova; oito MP4s de smoke reais produzidos (seis individuais e duas comparações), sem atribuição às políticas antigas.
