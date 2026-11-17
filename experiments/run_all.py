"""Bütün eksperimentləri işə salır və nəticələri results/ qovluğuna yazır.

İstifadə:
    python experiments/run_all.py            # tam (20 toxum)
    python experiments/run_all.py --quick    # sürətli yoxlama (3 toxum)

Eksperimentlər:
  E1  Qərəz mənbələrinin ablasiyası (S0–S5): hansı mexanizm nə qədər ədalətsizlik yaradır?
  E2  "Qrupu görməmək" (unawareness) kifayətdirmi? Proksi və SHAP təhlili.
  E3  "Əkiz" (counterfactual) testi: eyni insan, yalnız qrupu fərqli.
  E4  Qərəzin azaldılması üsullarının müqayisəsi (S5 ssenarisi).
  E5  Əks-əlaqə dövrəsi: model yalnız təsdiq etdiyi insanlardan öyrənir.
  E6  Real verilənlər: German Credit (UCI / OpenML credit-g).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_bias.metrics import THRESHOLD, fairness_report, summarize  # noqa: E402
from credit_bias.models import (ExpGradModel, GroupThresholdModel, ScoreModel,  # noqa: E402
                                make_gbm, make_lr)
from credit_bias.simulator import FEATURES, GROUP, SCENARIOS, generate_population  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
N_TRAIN, N_VAL, N_TEST = 20_000, 10_000, 20_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split(cfg, seed):
    rng = np.random.default_rng(seed)
    return (generate_population(cfg, N_TRAIN, rng), generate_population(cfg, N_VAL, rng),
            generate_population(cfg, N_TEST, rng))


def evaluate(model, te):
    a, s = model.decide(te), model.score(te)
    return {"true": fairness_report(te.y_true, a, te[GROUP], s),
            "hist": fairness_report(te.y_hist, a, te[GROUP], s)}


# --------------------------------------------------------------------------- E1
def e1_scenarios(seeds):
    out = {}
    for key, cfg in SCENARIOS.items():
        runs = {"LR": [], "GBM": []}
        for sd in seeds:
            tr, _, te = split(cfg, sd)
            runs["LR"].append(evaluate(ScoreModel(make_lr()).fit(tr), te)["true"])
            runs["GBM"].append(evaluate(ScoreModel(make_gbm(sd)).fit(tr), te)["true"])
        out[key] = {"config": cfg.as_dict(), **{m: summarize(r) for m, r in runs.items()}}
        g = out[key]["GBM"]
        log(f"E1 {key} {cfg.name:24s} DI={g['di_ratio']['mean']:.2f} EO={g['eo_diff']['mean']:.3f} "
            f"profit={g['profit']['mean']:.3f}")
    save("e1_scenarios", out)


# --------------------------------------------------------------------------- E2
def mean_abs_shap(model: ScoreModel, df) -> dict:
    X = df[model.base.feature_name_]
    contrib = model.base.predict(X, pred_contrib=True)[:, :-1]      # son sütun — baza dəyəri
    return dict(zip(model.base.feature_name_, np.abs(contrib).mean(axis=0).round(4).tolist()))


def e2_unawareness(seeds):
    cfg = SCENARIOS["S5"]
    variants = {
        "Qrup daxil": dict(with_group=True, drop=[]),
        "Qrupsuz (unawareness)": dict(with_group=False, drop=[]),
        "Qrupsuz + proksi silinib": dict(with_group=False, drop=["zone_code"]),
    }
    out, shap = {}, {}
    for name, v in variants.items():
        runs = []
        for sd in seeds:
            tr, _, te = split(cfg, sd)
            if v["drop"]:
                tr, te = tr.assign(zone_code=0.0), te.assign(zone_code=0.0)
            m = ScoreModel(make_gbm(sd), with_group=v["with_group"]).fit(tr)
            runs.append(evaluate(m, te)["true"])
            if sd == seeds[0]:
                shap[name] = mean_abs_shap(m, te)
        out[name] = summarize(runs)
        log(f"E2 {name:26s} DI={out[name]['di_ratio']['mean']:.2f} EO={out[name]['eo_diff']['mean']:.3f}")
    # proksinin qrupu nə dərəcədə "açıb göstərdiyi": zone_code-dan qrupu təxmin etmə dəqiqliyi
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tr, _, te = split(cfg, seeds[0])
    clf = LogisticRegression().fit(tr[FEATURES], tr[GROUP])
    proxy_auc = float(roc_auc_score(te[GROUP], clf.predict_proba(te[FEATURES])[:, 1]))
    corr = tr[FEATURES + [GROUP, "y_true"]].corr()[[GROUP, "y_true"]].round(3).to_dict()
    save("e2_unawareness", {"metrics": out, "shap_mean_abs": shap,
                            "group_recoverable_auc": proxy_auc, "correlations": corr})
    log(f"E2 qrup əlamətlərdən AUC={proxy_auc:.3f} ilə bərpa olunur")


# --------------------------------------------------------------------------- E3
def e3_counterfactual(seeds):
    """Eyni insanlar: bir dəfə B qrupunda, bir dəfə A qrupunda generasiya olunur."""
    cfg = SCENARIOS["S5"]
    rows = []
    for sd in seeds:
        tr, _, _ = split(cfg, sd)
        m = ScoreModel(make_gbm(sd)).fit(tr)
        as_b = generate_population(cfg, N_TEST, np.random.default_rng(10_000 + sd), force_group=1)
        as_a = generate_population(cfg, N_TEST, np.random.default_rng(10_000 + sd), force_group=0)
        assert np.array_equal(as_b.y_true, as_a.y_true)            # eyni insanlar
        sb, sa = m.score(as_b), m.score(as_a)
        db, da = sb > THRESHOLD, sa > THRESHOLD
        good = as_b.y_true.to_numpy() == 1
        rows.append({
            "approve_if_B": float(db.mean()), "approve_if_A": float(da.mean()),
            "flip_share": float((da & ~db).mean()),                  # A olsaydı təsdiq, B olduğu üçün rədd
            "flip_share_among_good": float((da & ~db)[good].mean()),
            "mean_score_drop": float((sa - sb).mean()),
        })
        if sd == seeds[0]:
            # "Humanist" nümunə: krediti qaytaracaq, A olsaydı təsdiq alacaq, B kimi rədd edilən biri
            idx = np.where(good & da & ~db)[0]
            i = int(idx[np.argmin(np.abs(sa[idx] - 0.9))])
            contrib = lambda df: dict(zip(FEATURES, m.base.predict(df[FEATURES].iloc[[i]], pred_contrib=True)[0, :-1].round(3).tolist()))
            example = {"features_as_A": as_a[FEATURES].iloc[i].round(3).to_dict(),
                       "features_as_B": as_b[FEATURES].iloc[i].round(3).to_dict(),
                       "score_as_A": float(sa[i]), "score_as_B": float(sb[i]),
                       "shap_as_A": contrib(as_a), "shap_as_B": contrib(as_b),
                       "y_true": int(as_b.y_true.iloc[i])}
    res = summarize(rows)
    save("e3_counterfactual", {"summary": res, "example": example})
    log(f"E3 B qrupunun {res['flip_share']['mean']:.1%}-i A qrupunda olsaydı təsdiq alardı")


# --------------------------------------------------------------------------- E4
def e4_mitigation(seeds):
    cfg = SCENARIOS["S5"]
    methods = {
        "Baza: LR (qrup daxil)": lambda sd, tr, va: ScoreModel(make_lr(), with_group=True).fit(tr),
        "Baza: GBM (qrupsuz)": lambda sd, tr, va: ScoreModel(make_gbm(sd)).fit(tr),
        "Reweighing + GBM": lambda sd, tr, va: ScoreModel(make_gbm(sd), reweigh=True).fit(tr),
        "ExpGrad (DP) + LR": lambda sd, tr, va: ExpGradModel("dp", sd).fit(tr),
        "ExpGrad (EO) + LR": lambda sd, tr, va: ExpGradModel("tpr", sd).fit(tr),
        "Qrup üzrə hədd (EO) + GBM": lambda sd, tr, va: GroupThresholdModel(make_gbm(sd)).fit(tr, va),
        "Oracle: GBM (həqiqi etiketlə)": lambda sd, tr, va: ScoreModel(make_gbm(sd)).fit(tr, label="y_true"),
    }
    out = {}
    for name, build in methods.items():
        runs_t, runs_h = [], []
        t0 = time.time()
        for sd in seeds:
            tr, va, te = split(cfg, sd)
            r = evaluate(build(sd, tr, va), te)
            runs_t.append(r["true"]); runs_h.append(r["hist"])
        out[name] = {"true": summarize(runs_t), "hist": summarize(runs_h)}
        t = out[name]["true"]
        log(f"E4 {name:30s} DI={t['di_ratio']['mean']:.2f} EO={t['eo_diff']['mean']:+.3f} "
            f"profit={t['profit']['mean']:.3f} auc={t['auc']['mean']:.3f} ({time.time()-t0:.0f}s)")
    save("e4_mitigation", out)


# --------------------------------------------------------------------------- E5
def e5_feedback(seeds, rounds=10, n_hist=10_000, n_round=5_000):
    """Seçici etiketlər (selective labels): bank yalnız təsdiq etdiyi müştərilərin
    real ödəmə nəticəsini görür. Rədd edilənlər haqqında məlumat heç vaxt yaranmır."""
    cfg = SCENARIOS["S5"]
    policies = {
        "Baza: GBM (qrupsuz)": lambda sd, pool, va: ScoreModel(make_gbm(sd)).fit(pool),
        "Qrup üzrə hədd (EO)": lambda sd, pool, va: GroupThresholdModel(make_gbm(sd)).fit(pool, va),
        "Baza + 10% kəşfiyyat": "explore",
    }
    out = {p: [] for p in policies}
    for p, build in policies.items():
        for sd in seeds:
            rng = np.random.default_rng(500 + sd)
            pool = generate_population(cfg, n_hist, rng)                 # tarixi (qərəzli) etiketlər
            traj = []
            cum_unfair_b = 0
            for t in range(1, rounds + 1):
                va = pool.sample(min(len(pool), 5000), random_state=sd + t)
                m = (ScoreModel(make_gbm(sd)).fit(pool) if build == "explore" else build(sd, pool, va))
                new = generate_population(cfg, n_round, rng)
                a = m.decide(new)
                if build == "explore":                                  # rədd edilənlərin 10%-ni sınaq üçün təsdiq et
                    a = np.where((a == 0) & (rng.random(len(a)) < 0.10), 1, a)
                r = fairness_report(new.y_true, a, new[GROUP])
                cum_unfair_b += int(((a == 0) & (new.y_true == 1) & (new[GROUP] == 1)).sum())
                traj.append({"round": t, "approval_B": r["approval_B"], "approval_A": r["approval_A"],
                             "eo_diff": r["eo_diff"], "di_ratio": r["di_ratio"], "profit": r["profit"],
                             "cum_unfair_reject_B": cum_unfair_b,
                             "pool_share_B": float(pool[GROUP].mean()), "pool_size": len(pool)})
                obs = new[a == 1].assign(y_hist=lambda d: d.y_true)    # yalnız təsdiq edilənlərin nəticəsi görünür
                pool = pd.concat([pool, obs], ignore_index=True)
            out[p].append(traj)
        last = np.mean([tr[-1]["eo_diff"] for tr in out[p]]); first = np.mean([tr[0]["eo_diff"] for tr in out[p]])
        log(f"E5 {p:24s} EO: 1-ci dövr {first:+.3f} → {rounds}-cu dövr {last:+.3f}")
    # toxumlar üzrə orta trayektoriya
    agg = {}
    for p, trajs in out.items():
        keys = [k for k in trajs[0][0] if k != "round"]
        agg[p] = [{"round": t + 1, **{k: {"mean": float(np.mean([tr[t][k] for tr in trajs])),
                                          "ci95": float(1.96 * np.std([tr[t][k] for tr in trajs], ddof=1) / np.sqrt(len(trajs)))}
                                      for k in keys}} for t in range(rounds)]
    save("e5_feedback", agg)


# --------------------------------------------------------------------------- E6
def e6_german(repeats):
    """Real verilənlər: UCI Statlog German Credit (1000 müraciətçi). Rəsmi xərc matrisi:
    pis müştərini 'yaxşı' saymaq = 5, yaxşını 'pis' saymaq = 1 → təsdiq həddi p > 5/6."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import RepeatedStratifiedKFold
    from fairlearn.reductions import ErrorRate, ExponentiatedGradient, TruePositiveRateParity
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

    ds = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(ROOT / "data"))
    X_raw, y = ds.data.copy(), (ds.target == "good").astype(int).to_numpy()
    sex = X_raw["personal_status"].astype(str).str.startswith("female").astype(int).to_numpy()   # 1 = qadın
    young = (X_raw["age"] <= 25).astype(int).to_numpy()                                         # 1 = ≤25 yaş
    thr = 5 / 6

    def design(drop):
        X = X_raw.drop(columns=drop)
        return pd.get_dummies(X, drop_first=True).astype(float)

    variants = {"Bütün əlamətlər": [], "Yaş və cins çıxarılıb": ["age", "personal_status"]}
    out = {"n": int(len(y)), "good_rate": float(y.mean()),
           "share_female": float(sex.mean()), "share_young": float(young.mean()),
           "base_rate": {"female": float(y[sex == 1].mean()), "male": float(y[sex == 0].mean()),
                         "young": float(y[young == 1].mean()), "older": float(y[young == 0].mean())}}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    for vname, drop in variants.items():
        X = design(drop)
        for mname in ["LR", "GBM", "ExpGrad (EO, yaş) + LR"]:
            res = {"young": [], "sex": [], "auc": [], "profit": []}
            for k, (i_tr, i_te) in enumerate(cv.split(X, y)):
                if mname == "LR":
                    m = make_lr().fit(X.iloc[i_tr], y[i_tr]); s = m.predict_proba(X.iloc[i_te])[:, 1]; a = s > thr
                elif mname == "GBM":
                    m = make_gbm(k).set_params(n_estimators=150, min_child_samples=20).fit(X.iloc[i_tr], y[i_tr])
                    s = m.predict_proba(X.iloc[i_te])[:, 1]; a = s > thr
                else:
                    eg = ExponentiatedGradient(make_lr(), TruePositiveRateParity(),
                                               objective=ErrorRate(costs={"fp": 5, "fn": 1}), eps=0.02, max_iter=60,
                                               sample_weight_name="logisticregression__sample_weight")
                    eg.fit(X.iloc[i_tr], y[i_tr], sensitive_features=young[i_tr])
                    s = eg._pmf_predict(X.iloc[i_te])[:, 1]; a = np.asarray(eg.predict(X.iloc[i_te], random_state=k)) == 1
                yt = y[i_te]
                for attr, gv in (("young", young[i_te]), ("sex", sex[i_te])):
                    mf = MetricFrame(metrics={"sel": selection_rate, "tpr": true_positive_rate},
                                     y_true=yt, y_pred=a.astype(int), sensitive_features=gv)
                    bg = mf.by_group
                    res[attr].append({"approval_0": float(bg.loc[0, "sel"]), "approval_1": float(bg.loc[1, "sel"]),
                                      "di_ratio": float(bg.loc[1, "sel"] / bg.loc[0, "sel"]),
                                      "eo_diff": float(bg.loc[0, "tpr"] - bg.loc[1, "tpr"])})
                from sklearn.metrics import roc_auc_score
                res["auc"].append(float(roc_auc_score(yt, s)))
                res["profit"].append(float((1 * (a & (yt == 1)) - 5 * (a & (yt == 0))).mean()))
            key = f"{vname} | {mname}"
            out[key] = {"young": summarize(res["young"]), "sex": summarize(res["sex"]),
                        "auc": summarize([{"v": v} for v in res["auc"]])["v"],
                        "profit": summarize([{"v": v} for v in res["profit"]])["v"]}
            o = out[key]
            log(f"E6 {key:45s} AUC={o['auc']['mean']:.3f} DI(gənc)={o['young']['di_ratio']['mean']:.2f} "
                f"DI(qadın)={o['sex']['di_ratio']['mean']:.2f}")
    # SHAP: yaşın modeldə rolu (LightGBM daxili TreeSHAP)
    X = design([])
    m = make_gbm(0).set_params(n_estimators=150, min_child_samples=20).fit(X, y)
    contrib = m.predict(X, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
    age_i = list(X.columns).index("age")
    out["shap_top10"] = imp.head(10).round(4).to_dict()
    out["shap_age_mean_young"] = float(contrib[young == 1, age_i].mean())
    out["shap_age_mean_older"] = float(contrib[young == 0, age_i].mean())
    save("e6_german", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    seeds = list(range(3 if args.quick else 20))
    steps = {"e1": lambda: e1_scenarios(seeds), "e2": lambda: e2_unawareness(seeds),
             "e3": lambda: e3_counterfactual(seeds), "e4": lambda: e4_mitigation(seeds),
             "e5": lambda: e5_feedback(seeds[:3] if args.quick else seeds[:10]),
             "e6": lambda: e6_german(2 if args.quick else 10)}
    t0 = time.time()
    for k, f in steps.items():
        if args.only is None or k in args.only:
            f()
    import fairlearn, lightgbm, sklearn
    save("run_info", {"seeds": len(seeds), "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                      "threshold": THRESHOLD, "python": platform.python_version(),
                      "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                      "lightgbm": lightgbm.__version__, "fairlearn": fairlearn.__version__,
                      "runtime_sec": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d")})
    log(f"Hazırdır: {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
