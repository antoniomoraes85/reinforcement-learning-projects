# Roteiro para os quatro integrantes

1. **Problema e protocolo:** mostrar MountainCar-v0, posição/velocidade, três ações, 200 passos, objetivo e recompensa. Explicar as duas grades e a perda de informação. Distinguir triagem (108) e campanha final (60); sem retuning pelo teste.
2. **Algoritmos e validade:** Q-learning off-policy; SARSA(λ) on-policy com traços; Dyna-Q model-based com contagens de múltiplos desfechos. Mostrar um cálculo TD e explicar terminal versus TimeLimit. N=0 deve coincidir com Q-learning; mencionar teste.
3. **Resultados:** abrir tabela por seed e painel da campanha final, depois curvas e sensibilidade da triagem. Diferenciar treino exploratório/avaliação greedy. Discutir média, dispersão e IC entre 10 políticas, não 1000 episódios como amostras independentes. Primeiro atingimento não é estabilidade.
4. **Demonstração e discussão:** vídeos finais lado a lado na seed predefinida; mostrar sucessos e falhas. Explicar custos reais versus planejamento e traços. Limitações: discretização, parâmetros globais, finitude da amostra, carga do hardware; aliasing não foi isolado causalmente. Fechar com reprodutibilidade e acesso a código/dados/checkpoints.

Antes de apresentar: conferir que notebook/figuras usam `final_v2`; smoke serve apenas para demonstrar funcionamento do pipeline. Não prometer resultados finais enquanto a validação 60/60 estiver ausente.

Perguntas de domínio: por que precisa de embalo? Como se distingue SARSA de Q-learning? O que o modelo Dyna conta? Por que não usar episódios como replicações? Qual evidência sustenta — e qual não sustenta — a conclusão apresentada?
