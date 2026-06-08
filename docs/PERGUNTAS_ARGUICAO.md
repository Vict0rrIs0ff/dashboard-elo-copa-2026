# Perguntas prováveis da arguição

## Por que o dataset é válido se não possui valores ausentes?

Porque a limpeza não se limita à imputação. Foram identificadas duplicatas semânticas, retratos com históricos muito curtos, necessidade de conversão de tipos, validações de consistência, engenharia de atributos e detecção de outliers.

## Por que remover registros com menos de 30 partidas?

Taxas calculadas com poucos jogos podem oscilar muito. O filtro reduz instabilidade em proporções como taxa de vitórias e média de gols.

## Por que não remover automaticamente os outliers?

Em dados esportivos, valores extremos podem representar seleções historicamente muito fortes ou muito fracas. Sinalizar e analisar é mais correto do que excluir indiscriminadamente.

## Por que `rank` e `rating` atuais não entram nos modelos?

Porque a classe é derivada da posição no ranking. Utilizar essas variáveis produziria vazamento de informação e tornaria a classificação artificialmente fácil.

## Qual é a diferença entre priori, verossimilhança e posteriori?

A priori é a probabilidade inicial da classe. A verossimilhança é a probabilidade de observar os atributos sabendo a classe. A posteriori é a probabilidade atualizada da classe depois de considerar os atributos.

## O que é suavização de Laplace?

É a adição de uma pequena contagem para evitar probabilidades iguais a zero em combinações raras.

## Por que usar divisão temporal?

A base contém retratos anuais das mesmas seleções. Treinar com anos anteriores e testar com anos mais recentes é uma avaliação mais realista do que misturar livremente os períodos.

## Qual modelo foi melhor?

Depende da métrica. A regressão logística teve maior acurácia e ROC AUC. A árvore de decisão teve maior recall e maior F1-score. O Bayes manual permaneceu competitivo e cumpre a parte probabilística do projeto.
