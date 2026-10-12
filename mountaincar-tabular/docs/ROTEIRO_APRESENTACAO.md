# Roteiro para os quatro integrantes

1. **Problema e protocolo:** mostrar MountainCar-v0, posição/velocidade, três ações, 200 passos, objetivo e recompensa. Explicar as duas grades e a perda de informação. Distinguir triagem (108) e campanha final (60); sem retuning pelo teste.
2. **Algoritmos e validade:** Q-learning off-policy; SARSA(λ) on-policy com traços; Dyna-Q model-based com contagens de múltiplos desfechos. Mostrar um cálculo TD e explicar terminal versus TimeLimit. N=0 deve coincidir com Q-learning; mencionar teste.
3. **Resultados:** abrir tabela por seed e painel da campanha final, depois curvas e sensibilidade da triagem. Diferenciar treino exploratório/avaliação greedy. Discutir média, dispersão e IC entre 10 políticas, não 1000 episódios como amostras independentes. Primeiro atingimento não é estabilidade.
4. **Demonstração e discussão:** vídeos finais lado a lado na seed predefinida; mostrar sucessos e falhas. Explicar custos reais versus planejamento e traços. Limitações: discretização, parâmetros globais, finitude da amostra, carga do hardware; aliasing não foi isolado causalmente. Fechar com reprodutibilidade e acesso a código/dados/checkpoints.

Campanha final **concluída e validada 60/60**. Notebook e gráficos principais usam `final_v2`; o smoke não integra os resultados finais.

Mensagem quantitativa: sucesso médio Q-learning 38,2%/73,6%, SARSA(λ) 83,7%/100%, Dyna-Q 0%/31,8% para 4×2/12×4. Mostrar pontos por seed e IC antes de interpretar médias. Dyna-Q 12×4 tem média 31,8%, mas mediana 1,5%: forte variabilidade, não uma política típica com 31,8%. SARSA 12×4 teve 100% observado nas dez políticas; isso não prova garantia populacional.

Nos vídeos de seed de treino 3100/reset 900000, SARSA chega em 185 passos (4×2) e 169 (12×4); os outros métodos falham dentro de 200 nesse caso. A seed foi predefinida, não selecionada por resultado. Abrir vídeo em tela cheia para ler as legendas.

Perguntas de domínio: por que precisa de embalo? Como se distingue SARSA de Q-learning? O que o modelo Dyna conta? Por que não usar episódios como replicações? Qual evidência sustenta — e qual não sustenta — a conclusão apresentada?
