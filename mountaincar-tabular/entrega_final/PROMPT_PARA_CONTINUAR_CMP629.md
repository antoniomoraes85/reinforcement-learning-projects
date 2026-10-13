# PROMPT PARA CONTINUAR CMP629

## Objetivo e restrições
Finalizar/manter a entrega do mini-projeto CMP629 MountainCar: notebook obrigatório, relatório acadêmico, pacote e documentação. PROIBIDO repetir treinamentos, alterar os resultados existentes, fabricar métricas ou misturar triagem e final. Dados brutos são imutáveis. Sem obrigação de publicar mudanças no GitHub; nenhuma publicação foi feita nesta rodada.

## Estado já produzido
Diretório de trabalho original: /workspace/scratch/6e0a9a68d9d5/ENTREGA_FINAL_CMP629
O ZIP mantém todos os caminhos relativos; ao extrair em outro computador, use a pasta extraída como raiz.
- CMP629_MOUNTAINCAR_ENTREGA_FINAL.ipynb: completo, 15 células de código executadas sequencialmente com outputs; tem dados e vídeos incorporados em base64/ZIP (hash verificado), dispensa downloads do repositório e retreino.
- CMP629_MOUNTAINCAR_RELATORIO_DEFINITIVO.pdf: completo em conteúdo, 36 páginas; LaTeX compilado com XeLaTeX. Formato A4 e regras aplicáveis ABNT; Nimbus Roman substitui Times New Roman. Percentis descritivos, boxplots e apêndice de código adicionados.
- apoio/: snapshot byte a byte de mountaincar-tabular, exceto cópia redundante do notebook antigo removida. Contém configurações, agentes, scripts, métricas, históricos, bundles ZIP e oito MP4s finais. Documentos em apoio são históricos, não representam o estado atual desta entrega.
- validacao_kernel_limpo/cmp629_reproduzido/: novos gráficos PNG/PDF e CSVs derivados sem treinamento.
- fonte_relatorio/relatorio.tex e construir_pdf.py: fonte e gerador do relatório; caminhos absolutos no TEX são regenerados pelo script.
- construir_notebook.py: gerador do notebook; reconstruí-lo apaga outputs, portanto reexecutar validação após alterações.
- validar_sem_sockets.py: execução alternativa utilizada neste ambiente; captura outputs IPython, sem servidor Jupyter.
- validar_notebook.py: tentativa de kernel separado; falha neste ambiente por rede/ZeroMQ, pode funcionar em outro computador.
- VALIDACAO_EXECUCAO.json, VALIDACAO_ENTREGA.json, REVISAO_VISUAL.md, STATUS_ENTREGA.md e MANIFESTO_ARQUIVOS.json: evidências.

## Entradas e proveniência
Enunciado oficial Mini-Projeto_ Modelagem e solução de problemas com Aprendizado por Reforço(8).pdf; relatório anterior CMP629_MiniProjeto_MountainCar_Padrao_Oficial(3).pdf; MountainCar_CMP629_Colab(2).ipynb; métricas/relatórios anexos. ZIP reinforcement-learning-projects-main.zip não constava localmente; substituído por clone do repositório indicado.
GitHub: https://github.com/antoniomoraes85/reinforcement-learning-projects/tree/19dcf17bfa589bb3cb16da4bd9f1862c501fddbd/mountaincar-tabular
As entradas originais desta sessão estão em ../upload; elas não são necessárias para executar o notebook final. O livro Sutton/Barto fornecido foi consultado para referências conceituais; não foi duplicado no ZIP.

## Resultados confirmados e imutáveis
108 execuções de triagem; 60 finais (3 métodos × 2 grades × 10 seeds), 1.000 episódios de treino/política, 100 resets de avaliação comuns. Seeds finais 3100..12100, passo 1000; avaliação 800000..800099.
Sucesso em 4×2: Q 38,2%; SARSA(λ) 83,7%; Dyna-Q 0%. Em 12×4: Q 73,6%; SARSA 100%; Dyna 31,8%.
Parâmetros: γ=0,99, ε inicial 1/mínimo 0,02. Q α=0,1 decay=0,99; SARSA α=0,02 λ=0,9 decay=0,995; Dyna α=0,05 N=5 decay=0,995.
Totais: 11.411.906 interações reais; 19.654.480 backups simulados; 52.575.778 componentes com traços. Não somar unidades heterogêneas.
Vídeos reais: seed de treino 3100/reset 900000, regra previamente definida. Não são estimativas da média.

## Correções desta rodada
Adequação ABNT, gráficos com legendas externas e novas distribuições por seed. CSV percentis_entre_seeds.csv: P10/P25/P50/P75/P90 por interpolação linear, n=10; percentis descritivos não são IC. Apêndice inclui código literal dos agentes e registro git_dirty da proveniência, com 11 hashes de módulos conferidos.
Notebook antes apresentava figuras externas e prosa breve. Foram incorporados o código dos agentes (Gymnasium opcional no import), dados/vídeos locais, explicação de modelagem, protocolo, triagem, custos e limitações.
Reconstruídas curvas exigidas: passos por episódio com falhas explicitadas; retornos/sucessos por interações no suporte comum de 30 execuções; atualizações por episódio/acumuladas com IC 95% e unidades separadas.
Não usar o contador Q legado como FLOPs. SARSA atualiza vetor denso, portanto componentes não nulos não representam todo o trabalho executado.
Diário distingue fatos e hipóteses; piloto de 24 execuções mencionado no relatório anterior não foi reapresentado como dado verificado por esta rodada.

## Validação executada
nbformat; execução sequencial em processo Python novo/IPython, 15 células sem erros; hashes dos dados incorporados; 60 linhas/60 hashes; dez seeds por célula; retorno=-passos; cumulativos e relações Dyna N=5; sucesso reconciliado; contagens da triagem; CRC dos bundles; integridade dos arquivos copiados; PDF renderizado e revisão visual.
Não repetir toda a auditoria histórica ou treinar de novo. Reexecutar apenas o necessário após alterações.

## Próximas ações
1. Colab hospedado: upload do notebook e Executar tudo. Registrar resultado real; não chamar funções de treinamento.
2. Obter notebook-base específico do professor apenas se indispensável e ajustar estrutura.
3. Confirmar integrantes e autoria intelectual. Não inventar nomes.
4. Se mudar algo, regenerar PDF/notebook, revisar páginas afetadas, atualizar manifesto e ZIP. Preservar última versão válida com escrita temporária + substituição.

## Comandos (na pasta extraída)
```bash
python -m pip install numpy pandas scipy matplotlib ipython nbformat nbclient ipykernel pymupdf
python validar_sem_sockets.py
python construir_pdf.py
```
`construir_pdf.py` requer XeLaTeX e fonte Nimbus Roman; lê os gráficos da pasta validacao_kernel_limpo/cmp629_reproduzido. Para reconstruir notebook, `python construir_notebook.py` ANTES da validação. O gerador de pacote original compara apoio com ../repo/mountaincar-tabular; em outra máquina, restaure esse snapshot ou substitua somente essa checagem por verificação dos hashes já registrados. Não retire validações para ocultar divergências.

## Salvamento
Nesta sessão os dois entregáveis e documentos foram salvos progressivamente. Atualizações devem preservar suas identidades quando disponíveis; o ZIP é a cópia portátil. Nunca afirmar salvamento/publicação sem confirmação da ferramenta.
