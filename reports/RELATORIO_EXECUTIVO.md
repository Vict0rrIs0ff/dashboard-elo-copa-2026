# Relatório executivo

## Tema

Classificação probabilística de seleções de elite com ratings Elo históricos das equipes classificadas para a Copa do Mundo de 2026.

## Resultado da limpeza

| Indicador | Quantidade |
|---|---:|
| Linhas originais | 4.683 |
| Duplicatas semânticas removidas | 48 |
| Retratos com menos de 30 partidas removidos | 728 |
| Linhas finais | 3.907 |
| Seleções distintas | 48 |

## Variável-alvo

`elite_top10 = 1` quando a seleção ocupa uma das dez primeiras posições do ranking Elo no retrato anual. Nos demais casos, `elite_top10 = 0`.

## Modelos

- Teorema de Bayes implementado manualmente;
- regressão logística;
- árvore de decisão.

## Síntese

A regressão logística obteve maior acurácia e ROC AUC. A árvore de decisão teve maior recall. O classificador bayesiano manual apresentou desempenho competitivo e permite demonstrar a priori, verossimilhança e posteriori de forma clara.
