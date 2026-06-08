from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.bayes_manual import ManualCategoricalNaiveBayes
from src.data_processing import TARGET_COLUMN

TRAIN_END_YEAR = 2018

NUMERIC_FEATURES = [
    "matches_total",
    "win_rate",
    "draw_rate",
    "goals_for_per_match",
    "goals_against_per_match",
    "goal_diff_per_match",
    "neutral_share",
    "rating_avg",
]

CATEGORICAL_FEATURES = ["confederation", "is_host"]
BAYES_NUMERIC_FEATURES = ["rating_avg", "win_rate", "goal_diff_per_match", "matches_total"]
BAYES_CATEGORICAL_FEATURES = ["confederation"]


@dataclass
class ModelBundle:
    evaluation_models: dict[str, Any]
    live_models: dict[str, Any]
    metrics: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    train_rows: int
    test_rows: int


def all_model_features() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def bayes_features() -> list[str]:
    return BAYES_NUMERIC_FEATURES + BAYES_CATEGORICAL_FEATURES


def build_logistic_regression() -> Pipeline:
    preprocessing = ColumnTransformer(transformers=[("numeric", StandardScaler(), NUMERIC_FEATURES), ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    return Pipeline(steps=[("preprocessing", preprocessing), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])


def build_decision_tree() -> Pipeline:
    preprocessing = ColumnTransformer(transformers=[("numeric", "passthrough", NUMERIC_FEATURES), ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    return Pipeline(steps=[("preprocessing", preprocessing), ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42))])


def build_manual_bayes() -> ManualCategoricalNaiveBayes:
    return ManualCategoricalNaiveBayes(numeric_features=BAYES_NUMERIC_FEATURES, categorical_features=BAYES_CATEGORICAL_FEATURES, alpha=1.0)


def evaluate_predictions(y_true: pd.Series, prediction: np.ndarray, probability_elite: np.ndarray) -> dict[str, float]:
    return {
        "acuracia": float(accuracy_score(y_true, prediction)),
        "precisao": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability_elite)),
    }


def fit_model_bundle(data: pd.DataFrame) -> ModelBundle:
    train = data.loc[data["year"] <= TRAIN_END_YEAR].copy()
    test = data.loc[data["year"] > TRAIN_END_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("A divisão temporal exige observações antes e depois de 2018.")
    evaluation_models: dict[str, Any] = {"Bayes manual": build_manual_bayes(), "Regressão logística": build_logistic_regression(), "Árvore de decisão": build_decision_tree()}
    metrics_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    for name, model in evaluation_models.items():
        selected_features = bayes_features() if name == "Bayes manual" else all_model_features()
        model.fit(train[selected_features], train[TARGET_COLUMN])
        prediction = model.predict(test[selected_features])
        probability_elite = model.predict_proba(test[selected_features])[:, 1]
        metrics_rows.append({"modelo": name, **evaluate_predictions(test[TARGET_COLUMN], prediction, probability_elite)})
        confusion_matrices[name] = confusion_matrix(test[TARGET_COLUMN], prediction)
    live_models: dict[str, Any] = {"Bayes manual": build_manual_bayes(), "Regressão logística": build_logistic_regression(), "Árvore de decisão": build_decision_tree()}
    for name, model in live_models.items():
        selected_features = bayes_features() if name == "Bayes manual" else all_model_features()
        model.fit(data[selected_features], data[TARGET_COLUMN])
    metrics = pd.DataFrame(metrics_rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    return ModelBundle(evaluation_models=evaluation_models, live_models=live_models, metrics=metrics, confusion_matrices=confusion_matrices, train_rows=int(len(train)), test_rows=int(len(test)))


def predict_profile(bundle: ModelBundle, profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, model in bundle.live_models.items():
        selected_features = bayes_features() if name == "Bayes manual" else all_model_features()
        probability_elite = float(model.predict_proba(profile[selected_features])[:, 1][0])
        rows.append({"modelo": name, "probabilidade_elite": probability_elite, "classificacao": "Elite (Top 10)" if probability_elite >= 0.5 else "Fora do Top 10"})
    return pd.DataFrame(rows)
