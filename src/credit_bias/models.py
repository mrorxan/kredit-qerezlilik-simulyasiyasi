"""Kredit skorinq modelləri və qərəzin azaldılması üsulları.

Qayda: bütün modellər yalnız bankın real həyatda bildiyi məlumatla —
tarixi etiketlə (y_hist) öyrədilir və tənzimlənir. Həqiqi etiket (y_true)
yalnız qiymətləndirmədə istifadə olunur.

Üsullar (Fairlearn və klassik ədəbiyyat):
  * unawareness   — qorunan əlaməti modeldən çıxarmaq;
  * reweighing    — Kamiran & Calders (2012): (qrup, etiket) cütlərinə çəki;
  * exp_gradient  — Agarwal və b. (2018): məhdudiyyətli optimallaşdırma (Fairlearn);
  * group_thresholds — Hardt və b. (2016) ruhunda qrup üzrə hədd seçimi
                       (bərabər imkan şərti ilə bank mənfəətinin maksimumu).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from fairlearn.reductions import (DemographicParity, ErrorRate, ExponentiatedGradient,
                                  TruePositiveRateParity)
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import GAIN, LOSS, THRESHOLD
from .simulator import FEATURES, GROUP

warnings.filterwarnings("ignore", category=UserWarning)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def make_gbm(seed: int = 0):
    return LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.9, random_state=seed, verbose=-1)


def cols(with_group: bool) -> list[str]:
    return FEATURES + ([GROUP] if with_group else [])


def _fit(model, X, y, w=None):
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):                       # sklearn Pipeline
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def reweighing_weights(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Kamiran & Calders: w(g, y) = P(g)·P(y) / P(g, y). Qrup və etiketi statistik asılısız edir."""
    y, g = np.asarray(y), np.asarray(g)
    w = np.empty(len(y), dtype=float)
    for gv in (0, 1):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            if m.any():
                w[m] = (g == gv).mean() * (y == yv).mean() / m.mean()
    return w


class ScoreModel:
    """Ehtimal verən model + sabit hədd (p > 0.8 → təsdiq)."""

    def __init__(self, base, with_group=False, reweigh=False):
        self.base, self.with_group, self.reweigh = base, with_group, reweigh

    def fit(self, df: pd.DataFrame, label="y_hist"):
        w = reweighing_weights(df[label], df[GROUP]) if self.reweigh else None
        _fit(self.base, df[cols(self.with_group)], df[label], w)
        return self

    def score(self, df):
        return self.base.predict_proba(df[cols(self.with_group)])[:, 1]

    def decide(self, df):
        return (self.score(df) > THRESHOLD).astype(int)


class ExpGradModel:
    """Fairlearn ExponentiatedGradient. Qorunan əlamət yalnız təlimdə istifadə olunur.
    Xərc-həssas məqsəd: səhv təsdiq (FP) = 4, səhv imtina (FN) = 1 → bankın p > 0.8 qaydası."""

    def __init__(self, constraint="tpr", seed=0):
        cons = TruePositiveRateParity() if constraint == "tpr" else DemographicParity()
        self.m = ExponentiatedGradient(make_lr(), constraints=cons,
                                       objective=ErrorRate(costs={"fp": LOSS, "fn": GAIN}),
                                       eps=0.01, max_iter=60,
                                       sample_weight_name="logisticregression__sample_weight")
        self.seed = seed

    def fit(self, df, label="y_hist"):
        self.m.fit(df[FEATURES], df[label], sensitive_features=df[GROUP])
        return self

    def score(self, df):
        return self.m._pmf_predict(df[FEATURES])[:, 1]

    def decide(self, df):
        return np.asarray(self.m.predict(df[FEATURES], random_state=self.seed)).astype(int)


class GroupThresholdModel:
    """Post-emal: hər qrup üçün ayrıca hədd. Validasiya verilənlərində (y_hist ilə)
    bərabər imkan (|TPR_A − TPR_B| ≤ tol) şərti daxilində bank mənfəətini maksimallaşdırır.
    Diqqət: qərar anında qorunan əlamətdən istifadə edir — hüquqi baxımdan mübahisəlidir."""

    def __init__(self, base, tol=0.02):
        self.inner, self.tol = ScoreModel(base, with_group=False), tol
        self.t = {0: THRESHOLD, 1: THRESHOLD}

    def fit(self, df, val, label="y_hist"):
        self.inner.fit(df, label)
        s, y, g = self.inner.score(val), val[label].to_numpy(), val[GROUP].to_numpy()
        grid = np.linspace(0.40, 0.98, 117)
        tpr = {k: np.array([((s > t) & (y == 1) & (g == k)).sum() / max(((y == 1) & (g == k)).sum(), 1)
                            for t in grid]) for k in (0, 1)}
        prof = {k: np.array([(GAIN * ((s > t) & (y == 1) & (g == k)) - LOSS * ((s > t) & (y == 0) & (g == k))).sum()
                             for t in grid]) for k in (0, 1)}
        ok = np.abs(tpr[0][:, None] - tpr[1][None, :]) <= self.tol
        total = np.where(ok, prof[0][:, None] + prof[1][None, :], -np.inf)
        i, j = np.unravel_index(np.argmax(total), total.shape)
        self.t = {0: float(grid[i]), 1: float(grid[j])}
        return self

    def score(self, df):
        return self.inner.score(df)

    def decide(self, df):
        s, g = self.score(df), df[GROUP].to_numpy()
        return (s > np.where(g == 1, self.t[1], self.t[0])).astype(int)
