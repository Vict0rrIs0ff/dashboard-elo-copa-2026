from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBin:
    lower_cut: float
    upper_cut: float


class ManualCategoricalNaiveBayes:
    LABELS = ["baixo", "medio", "alto"]

    def __init__(self, numeric_features: list[str], categorical_features: list[str], alpha: float = 1.0) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.classes_: list[int] = []
        self.bins_: dict[str, FeatureBin] = {}
        self.feature_values_: dict[str, list[str]] = {}
        self.priors_: dict[int, float] = {}
        self.class_counts_: dict[int, int] = {}
        self.likelihoods_: dict[str, dict[int, dict[str, float]]] = {}

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ManualCategoricalNaiveBayes":
        x = features.copy()
        y = target.astype(int).copy()
        self.classes_ = sorted(int(value) for value in y.unique())
        transformed = self._fit_transform(x)
        self.class_counts_ = {class_value: int((y == class_value).sum()) for class_value in self.classes_}
        self.priors_ = {class_value: self.class_counts_[class_value] / len(y) for class_value in self.classes_}
        self.likelihoods_ = {}
        for column in transformed.columns:
            values = sorted(transformed[column].astype(str).unique().tolist())
            self.feature_values_[column] = values
            self.likelihoods_[column] = {}
            for class_value in self.classes_:
                class_rows = transformed.loc[y == class_value, column].astype(str)
                denominator = len(class_rows) + self.alpha * len(values)
                self.likelihoods_[column][class_value] = {value: (int((class_rows == value).sum()) + self.alpha) / denominator for value in values}
        return self

    def _fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=features.index)
        for column in self.numeric_features:
            lower_cut, upper_cut = features[column].quantile([1 / 3, 2 / 3]).astype(float).tolist()
            if lower_cut >= upper_cut:
                lower_cut, upper_cut = features[column].quantile([0.25, 0.75]).astype(float).tolist()
            self.bins_[column] = FeatureBin(float(lower_cut), float(upper_cut))
            transformed[column] = self._apply_bin(features[column], self.bins_[column])
        for column in self.categorical_features:
            transformed[column] = features[column].astype(str)
        return transformed

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=features.index)
        for column in self.numeric_features:
            transformed[column] = self._apply_bin(features[column], self.bins_[column])
        for column in self.categorical_features:
            transformed[column] = features[column].astype(str)
        return transformed

    def _apply_bin(self, series: pd.Series, feature_bin: FeatureBin) -> pd.Series:
        return pd.cut(series, bins=[-np.inf, feature_bin.lower_cut, feature_bin.upper_cut, np.inf], labels=self.LABELS, include_lowest=True).astype(str)

    def _likelihood(self, feature: str, class_value: int, value: str) -> float:
        known_values = self.feature_values_[feature]
        probability = self.likelihoods_[feature][class_value].get(value)
        if probability is not None:
            return probability
        denominator = self.class_counts_[class_value] + self.alpha * (len(known_values) + 1)
        return self.alpha / denominator

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(features)
        result: list[list[float]] = []
        for _, row in transformed.iterrows():
            logs: dict[int, float] = {}
            for class_value in self.classes_:
                log_probability = math.log(self.priors_[class_value])
                for feature in transformed.columns:
                    log_probability += math.log(self._likelihood(feature, class_value, str(row[feature])))
                logs[class_value] = log_probability
            maximum = max(logs.values())
            exponentials = {class_value: math.exp(value - maximum) for class_value, value in logs.items()}
            total = sum(exponentials.values())
            result.append([exponentials[class_value] / total for class_value in self.classes_])
        return np.asarray(result)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(features)
        indexes = probabilities.argmax(axis=1)
        return np.asarray([self.classes_[index] for index in indexes])

    def explain_prediction(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(features) != 1:
            raise ValueError("A explicação detalhada exige exatamente uma observação.")
        transformed = self.transform(features)
        row = transformed.iloc[0]
        details: list[dict[str, Any]] = []
        for class_value in self.classes_:
            details.append({"etapa": "priori", "atributo": "classe", "valor_observado": str(class_value), "classe": class_value, "probabilidade": self.priors_[class_value]})
            for feature in transformed.columns:
                details.append({"etapa": "verossimilhanca", "atributo": feature, "valor_observado": str(row[feature]), "classe": class_value, "probabilidade": self._likelihood(feature, class_value, str(row[feature]))})
        posterior = self.predict_proba(features)[0]
        posterior_table = pd.DataFrame({"classe": self.classes_, "probabilidade_posteriori": posterior})
        return pd.DataFrame(details), posterior_table
