# Relatório técnico: classificação probabilística de seleções de elite com ratings Elo históricos

## Integrantes

- **Victor de Pinho Sampaio**
- **Pedro Augusto Teixeira**

## 1. Descrição do dataset

O projeto utiliza o dataset **2026 FIFA World Cup — Historical Elo Ratings**, disponibilizado no Kaggle. O arquivo reúne informações históricas das 48 seleções classificadas para a Copa do Mundo de 2026. A versão bruta utilizada contém 4.683 registros e 23 atributos originais, com retratos entre 1901 e 2026.

Fonte original: https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings

O carregamento principal da base é realizado diretamente pelo KaggleHub. A cópia local do CSV permanece no repositório apenas como contingência e para garantir rastreabilidade.

Entre os atributos quantitativos estão ranking, rating Elo, quantidade de partidas, vitórias, derrotas, empates e gols. Entre os atributos qualitativos estão seleção, código do país e confederação. A coluna `is_host` identifica a condição de anfitriã.

A variável categórica alvo foi criada pela equipe:

```text
elite_top10 = 1, quando rank <= 10
elite_top10 = 0, quando rank > 10
```

O objetivo é estimar a probabilidade de uma seleção pertencer ao grupo de elite em determinado retrato anual, utilizando informações históricas e indicadores de desempenho acumulado.

## 2. Justificativa da escolha

O dataset é adequado porque contém dados reais de fonte aberta, apresenta variáveis quantitativas e qualitativas, exige tratamento justificável, permite engenharia de atributos, admite variável-alvo categórica interpretável, possui volume suficiente para EDA e classificação e possibilita dashboard interativo.

## 3. Tratamentos aplicados

### 3.1 Conversão de tipos

A coluna `snapshot_date` foi convertida de texto para data com `pd.to_datetime`. A conversão permite ordenar cronologicamente os registros e identificar retratos redundantes.

### 3.2 Valores ausentes

Foram verificadas ausências antes e depois do tratamento. Nenhum valor ausente foi identificado. Portanto, não houve imputação.

### 3.3 Consistência estrutural

Foram validadas as identidades:

```text
matches_home + matches_away + matches_neutral = matches_total
wins + losses + draws = matches_total
```

Também foram verificados gols negativos, ranking não positivo, datas inválidas e partidas não positivas. Nenhuma inconsistência estrutural foi encontrada.

### 3.4 Duplicatas semânticas

Não foram encontradas duplicatas exatas. Entretanto, foram identificadas 48 observações esportivamente equivalentes em 2026, diferenciadas apenas pela data do retrato. O pipeline ordena cronologicamente as datas e preserva o primeiro retrato equivalente.

### 3.5 Retratos com poucas partidas

Foram removidos registros com menos de 30 partidas acumuladas. A decisão reduz a instabilidade das taxas calculadas em históricos muito curtos.

### 3.6 Engenharia de atributos

Foram criadas taxas de vitórias, empates e derrotas, médias de gols marcados e sofridos, saldo de gols por partida e proporções de partidas em casa, fora e campo neutro.

### 3.7 Outliers

Os outliers foram sinalizados pelo método do intervalo interquartil. Foram mantidos porque podem representar trajetórias esportivas reais e relevantes.

### 3.8 Resultado da limpeza

| Indicador | Quantidade |
|---|---:|
| Linhas originais | 4.683 |
| Duplicatas semânticas removidas | 48 |
| Retratos com menos de 30 partidas removidos | 728 |
| Linhas finais | 3.907 |
| Seleções distintas | 48 |
| Observações fora do Top 10 | 3.052 |
| Observações de elite | 855 |

## 4. Análise exploratória

A classe `elite_top10` representa aproximadamente 21,9% das observações tratadas. O desequilíbrio moderado justifica a utilização de métricas adicionais à acurácia.

As correlações mais relevantes com a variável alvo são:

| Variável | Correlação com `elite_top10` |
|---|---:|
| Rating Elo médio histórico | 0,593 |
| Taxa de vitórias | 0,479 |
| Saldo de gols por partida | 0,448 |
| Gols marcados por partida | 0,445 |
| Gols sofridos por partida | -0,228 |

A análise por confederação mostra concentração histórica das observações de elite em CONMEBOL e UEFA. Essa leitura é descritiva e restrita às 48 seleções presentes no dataset.

## 5. Teorema de Bayes

O classificador bayesiano utiliza `rating_avg`, `win_rate`, `goal_diff_per_match`, `matches_total` e `confederation`. As variáveis quantitativas são discretizadas em três faixas: baixo, médio e alto.

Para cada classe `C` e conjunto de atributos `X`, aplica-se:

```text
P(C | X) = P(X | C) × P(C) / P(X)
```

A implementação calcula as probabilidades a priori, as verossimilhanças condicionais e as probabilidades a posteriori. A suavização de Laplace evita probabilidades iguais a zero. O cálculo foi implementado manualmente em `src/bayes_manual.py`.

## 6. Algoritmos de classificação

Além do Bayes manual, foram implementadas regressão logística e árvore de decisão. A avaliação utiliza treino até 2018 e teste entre 2019 e 2026.

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Árvore de decisão | 0,852 | 0,568 | 1,000 | 0,725 | 0,940 |
| Regressão logística | 0,865 | 0,610 | 0,853 | 0,711 | 0,956 |
| Bayes manual | 0,846 | 0,571 | 0,853 | 0,684 | 0,937 |

A regressão logística atingiu maior acurácia e maior ROC AUC. A árvore de decisão alcançou maior F1-score e recall igual a 1,000. O Bayes manual apresentou desempenho competitivo e explicita o raciocínio probabilístico.

## 7. Dashboard interativo em Streamlit

O produto principal do projeto é o dashboard em Streamlit. A aplicação foi estruturada para reunir, em uma única interface, resumo da limpeza, indicadores gerais, gráficos exploratórios, matriz de correlação, métricas dos modelos, matrizes de confusão, formulário de simulação e detalhamento das etapas do Bayes manual.

A escolha do Streamlit é coerente com a atividade porque permite que o usuário informe atributos de uma nova observação e compare, imediatamente, as probabilidades e classificações geradas pelos três métodos. O notebook foi mantido apenas como material complementar para reprodução técnica das análises.

## 8. Conclusões

Os indicadores históricos permitem classificar a probabilidade de uma seleção pertencer ao Top 10 do ranking Elo. O rating médio histórico, a taxa de vitórias e o saldo de gols por partida apresentaram relações relevantes com a classe de elite.

A comparação mostra que a escolha do modelo depende da métrica priorizada. O Bayes manual cumpre o núcleo probabilístico da disciplina, enquanto os outros modelos complementam a interpretação.

## 9. Limitações

- o dataset contém somente as 48 seleções classificadas para 2026;
- existem múltiplos retratos anuais de uma mesma seleção;
- a variável alvo Top 10 é uma definição operacional;
- a análise não estabelece causalidade;
- os resultados podem variar em outros recortes temporais.

## 10. Uso de IA

A equipe utilizou IA generativa como ferramenta de apoio para estruturação, revisão e documentação. A equipe é responsável pela execução, conferência, compreensão e defesa do conteúdo. A declaração completa está em `docs/DECLARACAO_USO_IA.md`.
