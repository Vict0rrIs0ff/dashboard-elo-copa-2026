# Conceitos aproveitados do notebook de referência

O notebook anterior foi utilizado como referência de organização didática. Foram aproveitados os seguintes elementos:

| Elemento do notebook anterior | Aplicação no projeto Elo Copa 2026 |
|---|---|
| `pandas`, `numpy`, `matplotlib`, `seaborn` | Tratamento, cálculos e gráficos exploratórios |
| `kagglehub` e `KaggleDatasetAdapter` | Carregamento principal diretamente da fonte do Kaggle, com cópia local de contingência |
| `df.shape`, `df.info()`, `df.describe()` | Diagnóstico inicial da base |
| `df.duplicated().sum()` | Verificação de duplicatas exatas |
| `df.isna().sum()` e percentual de nulos | Diagnóstico de valores ausentes |
| tabela de qualidade | Conferência dos tipos e ausências depois do tratamento |
| subconjuntos temáticos | Organização das análises por desempenho, confederação, tempo e modelagem |
| gráficos com títulos claros | Visualizações com objetivo analítico explícito |
| exportação do CSV tratado | Integração com o dashboard e os modelos |

Também foram incluídos conceitos exigidos especificamente nesta atividade:

- variável-alvo categórica derivada;
- engenharia de atributos;
- detecção de outliers pelo intervalo interquartil;
- Teorema de Bayes implementado manualmente;
- suavização de Laplace;
- regressão logística;
- árvore de decisão;
- divisão temporal de treino e teste;
- acurácia, precisão, recall, F1-score, ROC AUC e matriz de confusão;
- dashboard interativo com formulário para classificação de novos perfis.
