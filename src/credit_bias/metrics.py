"""Ədalətlilik və bank göstəriciləri.

Metriklər əl ilə hesablanır (şəffaflıq üçün) və testlərdə Fairlearn
kitabxanasının nəticələri ilə tutuşdurulur.

Qərar qaydası: bank krediti uğurla qaytarılarsa +1 vahid qazanır,
defolt olarsa -4 vahid itirir. Gözlənilən mənfəət müsbət olur, əgər
P(qaytarır) > 4 / (1 + 4) = 0.8. Bu, illüstrativ fərziyyədir.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

GAIN, LOSS = 1.0, 4.0
THRESHOLD = LOSS / (GAIN + LOSS)        # 0.8


def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    d = mask_den.sum()
    return float(mask_num[mask_den].mean()) if d else float("nan")


def fairness_report(y: np.ndarray, approve: np.ndarray, group: np.ndarray,
                    score: np.ndarray | None = None) -> dict:
    """y — müqayisə üçün etiket (1 = qaytarır), approve — qərar (1 = təsdiq),
    group — 0 (A) / 1 (B). Qaytarır: qrup göstəriciləri və fərqlər."""
    y = np.asarray(y).astype(int)
    a = np.asarray(approve).astype(int)
    g = np.asarray(group).astype(int)
    A, B = g == 0, g == 1

    out = {
        "accuracy": float((a == y).mean()),
        "approval_A": float(a[A].mean()),
        "approval_B": float(a[B].mean()),
        # TPR: həqiqətən etibarlı olanlar arasında təsdiq payı (bərabər imkan)
        "tpr_A": _rate(a == 1, A & (y == 1)),
        "tpr_B": _rate(a == 1, B & (y == 1)),
        # FPR: krediti qaytarmayanlar arasında təsdiq payı
        "fpr_A": _rate(a == 1, A & (y == 0)),
        "fpr_B": _rate(a == 1, B & (y == 0)),
        # Bankın müraciətçi başına orta mənfəəti (şərti vahid)
        "profit": float((GAIN * ((a == 1) & (y == 1)) - LOSS * ((a == 1) & (y == 0))).mean()),
        # Haqsız imtina: hər 10 000 müraciətçidən krediti qaytaracaq, amma rədd edilənlər
        "unfair_reject_A_per10k": float(((a == 0) & (y == 1))[A].mean() * 1e4),
        "unfair_reject_B_per10k": float(((a == 0) & (y == 1))[B].mean() * 1e4),
    }
    out["dp_diff"] = out["approval_A"] - out["approval_B"]
    out["di_ratio"] = out["approval_B"] / out["approval_A"] if out["approval_A"] else float("nan")
    out["eo_diff"] = out["tpr_A"] - out["tpr_B"]
    out["eodds_diff"] = max(abs(out["eo_diff"]), abs(out["fpr_A"] - out["fpr_B"]))
    out["excess_unfair_reject_B_per10k"] = out["unfair_reject_B_per10k"] - out["unfair_reject_A_per10k"]
    if score is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
    return out


def summarize(runs: list[dict]) -> dict:
    """Bir neçə təsadüfi toxumun (seed) nəticələrini orta ± 95% etibarlılıq intervalı kimi ümumiləşdirir."""
    keys = runs[0].keys()
    res = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        m = float(np.nanmean(v))
        ci = float(1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))) if len(v) > 1 else 0.0
        res[k] = {"mean": m, "ci95": ci}
    return res
