# Roteiro de defesa presencial

## 1. Apresentação do problema

O projeto investiga a probabilidade de uma seleção pertencer ao Top 10 do ranking Elo em determinado retrato anual, considerando seu histórico esportivo.

## 2. Dataset

A base possui 4.683 linhas originais, 23 atributos e informações reais das 48 seleções classificadas para a Copa do Mundo de 2026.

## 3. Limpeza

Explique:

- ausência de valores nulos;
- conversão da data;
- remoção de 48 duplicatas semânticas;
- remoção de 728 retratos com menos de 30 partidas;
- criação de taxas e médias por partida;
- manutenção justificada dos outliers plausíveis.

## 4. Variável-alvo

`elite_top10 = 1` quando `rank <= 10`; nos demais casos, `elite_top10 = 0`.

## 5. Bayes

Apresente a priori, verossimilhança, posteriori e suavização de Laplace. Destaque que o cálculo foi implementado manualmente.

## 6. Modelos adicionais

Explique a regressão logística e a árvore de decisão. Mostre a comparação por acurácia, precisão, recall, F1-score, ROC AUC e matrizes de confusão.

## 7. Dashboard

Demonstre os gráficos e faça uma simulação alterando os atributos de uma seleção.

## 8. Conclusão

A regressão logística atingiu maior acurácia e ROC AUC. A árvore alcançou recall igual a 1,000. O Bayes manual apresentou resultado competitivo e evidencia o núcleo probabilístico do trabalho.
