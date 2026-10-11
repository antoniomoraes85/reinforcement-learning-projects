# CMP629 — Mini-projeto MountainCar-v0

**UFRGS · Aprendizado por Reforço · atualização: 11/10/2026**

Comparação experimental de **Q-learning**, **SARSA(λ)** e **Dyna-Q**, considerando as discretizações **4×2** e **12×4** do ambiente Gymnasium `MountainCar-v0`.

> **Status científico: TRIAGEM E AUDITORIA.** Os resultados apresentados **não são** o experimento final da disciplina. Não utilizar para declarar superioridade geral dos algoritmos.

## Abrir no Google Colab

[**Notebook interativo MountainCar CMP629**](https://colab.research.google.com/github/antoniomoraes85/reinforcement-learning-projects/blob/main/mountaincar-tabular/notebooks/MountainCar_CMP629_Colab.ipynb)

O notebook:
- carrega e valida os dados experimentais;
- constrói gráficos comparativos por algoritmo e discretização;
- separa taxa de sucesso durante treinamento exploratório da avaliação da política;
- mostra efeitos de `N` no Dyna-Q;
- executa uma demonstração **nova e independente**, explicitamente rotulada como demonstração e não como parte das 108 execuções originais;
- exporta gráficos e tabelas.

## Evidência preservada

- **108/108 execuções** de triagem em resumos de resultados (3 algoritmos × 6 configurações por algoritmo × 2 discretizações × 3 seeds).
- O pacote original de resultados individuais JSON não estava entre os arquivos fornecidos: validação direta de JSONs por execução permanece pendente.
- `dados/triagem/selecionadas_por_discretizacao.csv`: médias de 3 seeds (50 episódios de avaliação por seed).
- `dados/triagem/resumo_execucoes.csv.gz`: CSV comprimido derivado de 108 resultados **por execução**, com todas as métricas preservadas mas sem a coluna de hashes de configuração (ver auditoria).
- `figuras/`: gráficos vetoriais de síntese. O notebook também regenera visualizações mais detalhadas dos resumos.
- `docs/AUDITORIA.md`: limitações, conclusões verificadas e pendências.

## Configuração do experimento original

| Aspecto | Configuração |
|---|---|
| Ambiente | MountainCar-v0, 3 ações discretas |
| Limite | 200 passos por episódio |
| Treinamento | 1.000 episódios por configuração |
| Avaliação | 50 episódios por política |
| γ | 0,99 |
| Seeds de triagem | 100, 1100, 2100 |
| Discretizações | 4×2, 12×4 |
| Seleção | Maior AUC média de treinamento; **uma configuração global por algoritmo** |

Hiperparâmetros selecionados originalmente:
- SARSA(λ): `α=0,02`, `λ=0,9`, `decay=0,995`;
- Q-learning: `α=0,1`, `decay=0,99`;
- Dyna-Q: `α=0,05`, `N=5`, `decay=0,995`.

## Limitações e próximos passos

1. Implementar e validar no projeto completo os ajustes indicados em `docs/AUDITORIA.md`.
2. Corrigir a nomenclatura da métrica "estabilidade" para **primeiro atingimento de limiar**.
3. Não confundir operações elementares de atualização Q entre SARSA(λ), Q-learning e Dyna-Q.
4. Executar testes integrados do Dyna-Q, incluindo equivalência com Q-learning quando `N=0`.
5. Só então congelar o protocolo e executar o experimento confirmatório.

**Origem:** análise de arquivos disponibilizados na conversa CMP629; dados de triagem preservados sem retreinamento.

**Escopo:** este diretório NÃO inclui o material didático protegido da disciplina, a dissertação MPC/RL ou arquivos pessoais.
