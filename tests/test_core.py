"""Simulyator, metriklər və üsullar üçün avtomatik testlər (pytest)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fairlearn.metrics import (demographic_parity_difference, equalized_odds_difference,
                               true_positive_rate)
from fairlearn.metrics import MetricFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit_bias.metrics import THRESHOLD, fairness_report  # noqa: E402
from credit_bias.models import reweighing_weights  # noqa: E402
from credit_bias.simulator import SCENARIOS, BiasConfig, generate_population  # noqa: E402


@pytest.fixture(scope="module")
def pop():
    return generate_population(SCENARIOS["S5"], 30_000, np.random.default_rng(0))


def test_true_ability_is_equal_across_groups(pop):
    """Dizaynın əsası: həqiqi ödəmə səviyyəsi qruplar arasında eynidir."""
    r = pop.groupby("group").y_true.mean()
    assert abs(r[0] - r[1]) < 0.02


def test_label_bias_only_hurts_group_b():
    df = generate_population(BiasConfig(label_bias=0.3), 30_000, np.random.default_rng(1))
    a, b = df[df.group == 0], df[df.group == 1]
    assert (a.y_hist == a.y_true).all()                      # A-nın etiketləri toxunulmazdır
    flipped = ((b.y_true == 1) & (b.y_hist == 0)).sum() / (b.y_true == 1).sum()
    assert 0.27 < flipped < 0.33                              # ~30% etibarlı B "pis" yazılıb
    assert ((b.y_true == 0) & (b.y_hist == 1)).sum() == 0     # əks istiqamətdə təhrif yoxdur


def test_proxy_is_unrelated_to_ability(pop):
    assert abs(np.corrcoef(pop.zone_code, pop.y_true)[0, 1]) < 0.03
    assert np.corrcoef(pop.zone_code, pop.group)[0, 1] > 0.5


def test_counterfactual_twins_share_everything_but_group():
    cfg = BiasConfig(proxy_strength=0.0, measurement_bias=0.0)
    a = generate_population(cfg, 1000, np.random.default_rng(7), force_group=0)
    b = generate_population(cfg, 1000, np.random.default_rng(7), force_group=1)
    assert (a.y_true.values == b.y_true.values).all()
    assert np.allclose(a.log_income, b.log_income)


def test_metrics_match_fairlearn(pop):
    rng = np.random.default_rng(3)
    approve = (rng.random(len(pop)) < np.where(pop.group == 1, 0.3, 0.6)).astype(int)
    r = fairness_report(pop.y_true, approve, pop.group)
    assert r["dp_diff"] == pytest.approx(
        demographic_parity_difference(pop.y_true, approve, sensitive_features=pop.group), abs=1e-12)
    assert r["eodds_diff"] == pytest.approx(
        equalized_odds_difference(pop.y_true, approve, sensitive_features=pop.group), abs=1e-12)
    tpr = MetricFrame(metrics=true_positive_rate, y_true=pop.y_true, y_pred=approve,
                      sensitive_features=pop.group).by_group
    assert r["eo_diff"] == pytest.approx(tpr[0] - tpr[1], abs=1e-12)


def test_reweighing_makes_label_independent_of_group(pop):
    w = reweighing_weights(pop.y_hist, pop.group)
    rates = [np.average(pop.y_hist[pop.group == g], weights=w[pop.group == g]) for g in (0, 1)]
    assert rates[0] == pytest.approx(rates[1], abs=1e-9)


def test_threshold_matches_cost_assumption():
    assert THRESHOLD == pytest.approx(0.8)
