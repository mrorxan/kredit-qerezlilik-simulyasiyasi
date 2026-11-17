"""results/*.json fayllarından təqdimat üçün qrafiklər yaradır (results/figures/*.png)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palitra validate_palette.js ilə yoxlanılıb (CVD və kontrast PASS)
COL_A, COL_B, COL_C = "#2F64B0", "#C8741F", "#8A4FB8"
INK, INK2, MUTED, GRID = "#1F2433", "#4A5163", "#7A8194", "#E3E7EE"
NEUTRAL = "#9AA6BF"

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def az(v, d=2):
    """Azərbaycan yazı qaydası: onluq kəsr vergüllə."""
    return f"{v:.{d}f}".replace(".", ",")


AZ_TICK = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",").replace("-", "−"))


def comma_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(AZ_TICK)
    if y:
        ax.yaxis.set_major_formatter(AZ_TICK)


# --------------------------------------------------------------------- F1: ssenarilər
def fig_scenarios():
    d = load("e1_scenarios")
    keys = list(d)
    labels = [f"{k}\n{d[k]['config']['name']}" for k in keys]
    a = [d[k]["GBM"]["approval_A"]["mean"] for k in keys]
    b = [d[k]["GBM"]["approval_B"]["mean"] for k in keys]
    ea = [d[k]["GBM"]["approval_A"]["ci95"] for k in keys]
    eb = [d[k]["GBM"]["approval_B"]["ci95"] for k in keys]
    x = np.arange(len(keys)); w = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - w/2 - 0.01, a, w, color=COL_A, label="A qrupu", yerr=ea, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.bar(x + w/2 + 0.01, b, w, color=COL_B, label="B qrupu (qorunan)", yerr=eb, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    for i, k in enumerate(keys):
        di = d[k]["GBM"]["di_ratio"]["mean"]
        ax.text(x[i], max(a[i], b[i]) + 0.045, f"DI = {az(di)}", ha="center", fontsize=11.5,
                color=("#B23A2E" if di < 0.8 else INK2), fontweight="bold" if di < 0.8 else "normal")
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("Kredit təsdiqi payı")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.62)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.12))
    fig.savefig(FIG / "f1_scenarios.png"); plt.close(fig)


# --------------------------------------------------------------------- F2: SHAP, proksi
NAMES = {"log_income": "Gəlir (log)", "debt_to_income": "Borc / gəlir", "credit_history_years": "Kredit tarixçəsi",
         "late_payments": "Gecikmiş ödənişlər", "employment_years": "İş stajı", "zone_code": "Zona kodu (proksi)",
         "group": "Qrup"}


def fig_shap_proxy():
    d = load("e2_unawareness")["shap_mean_abs"]["Qrupsuz (unawareness)"]
    items = sorted(d.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    y = np.arange(len(items))
    cols = [COL_B if k == "zone_code" else NEUTRAL for k, _ in items]
    ax.barh(y, [v for _, v in items], height=0.55, color=cols)
    for i, (k, v) in enumerate(items):
        ax.text(v + 0.006, i, az(v, 3), va="center", fontsize=11, color=INK)
    ax.set_yticks(y, [NAMES[k] for k, _ in items], fontsize=12)
    ax.set_xlabel("Orta |SHAP| töhfəsi (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(d.values()) * 1.18)
    comma_axes(ax, y=False)
    fig.savefig(FIG / "f2_shap_proxy.png"); plt.close(fig)


# --------------------------------------------------------------------- F3: əkiz nümunə
def fig_twin():
    ex = load("e3_counterfactual")["example"]
    feats = list(ex["shap_as_A"])
    y = np.arange(len(feats))[::-1]; h = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.barh(y + h/2 + 0.01, [ex["shap_as_A"][f] for f in feats], h, color=COL_A, label=f"A qrupunda olsaydı (bal {az(ex['score_as_A'])})")
    ax.barh(y - h/2 - 0.01, [ex["shap_as_B"][f] for f in feats], h, color=COL_B, label=f"B qrupunda (bal {az(ex['score_as_B'])})")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [NAMES[f] for f in feats], fontsize=12)
    ax.set_xlabel("SHAP töhfəsi: (+) təsdiqə, (−) imtinaya doğru")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=10.5)
    comma_axes(ax, y=False)
    fig.savefig(FIG / "f3_twin_shap.png"); plt.close(fig)


# --------------------------------------------------------------------- F4: üsulların müqayisəsi
SHORT = {"Baza: LR (qrup daxil)": "LR (qrup daxil)", "Baza: GBM (qrupsuz)": "GBM (qrupsuz)",
         "Reweighing + GBM": "Reweighing", "ExpGrad (DP) + LR": "ExpGrad (DP)",
         "ExpGrad (EO) + LR": "ExpGrad (EO)", "Qrup üzrə hədd (EO) + GBM": "Qrup üzrə hədd",
         "Oracle: GBM (həqiqi etiketlə)": "Oracle (həqiqi etiket)"}


def fig_mitigation():
    d = load("e4_mitigation")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    offs = {"LR (qrup daxil)": (10, 6), "GBM (qrupsuz)": (10, -16), "Reweighing": (10, 10),
            "ExpGrad (DP)": (12, -12), "ExpGrad (EO)": (12, -26), "Qrup üzrə hədd": (12, -14),
            "Oracle (həqiqi etiket)": (12, 4)}
    for name, r in d.items():
        t = r["true"]; s = SHORT[name]
        kind = "base" if name.startswith("Baza") else ("oracle" if name.startswith("Oracle") else "fair")
        col = {"base": COL_B, "fair": COL_A, "oracle": COL_C}[kind]
        ax.errorbar(t["eo_diff"]["mean"], t["profit"]["mean"], xerr=t["eo_diff"]["ci95"], yerr=t["profit"]["ci95"],
                    fmt="o", ms=9, color=col, mec="white", mew=2, ecolor=col, elinewidth=1.2, capsize=3)
        ox, oy = offs.get(s, (8, 4))
        if s == "ExpGrad (EO)":
            s = None
        if s == "ExpGrad (DP)":
            s = "ExpGrad (DP və EO)"
        if s is None:
            continue
        ax.annotate(s, (t["eo_diff"]["mean"], t["profit"]["mean"]), xytext=(ox, oy), textcoords="offset points",
                    fontsize=11, color=INK, ha="left" if ox > 0 else "right")
    ax.set_xlabel("Bərabər imkan fərqi (TPR_A − TPR_B), həqiqi etiketlə  →  daha ədalətsiz")
    ax.set_ylabel("Bankın mənfəəti, şərti vahid / müraciət")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=c, ms=8, label=l) for c, l in
          ((COL_B, "Baza modellər"), (COL_A, "Qərəzi azaldan üsullar"), (COL_C, "Oracle (yuxarı hədd)"))]
    ax.legend(handles=hs, loc="lower left", fontsize=10.5)
    ax.set_xlim(-0.05, 0.45)
    comma_axes(ax)
    fig.savefig(FIG / "f4_mitigation.png"); plt.close(fig)


# --------------------------------------------------------------------- F5: əks-əlaqə dövrəsi
def fig_feedback():
    d = load("e5_feedback")
    cols = [COL_B, COL_A, COL_C]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for (name, traj), c in zip(d.items(), cols):
        r = [t["round"] for t in traj]
        m = np.array([t["eo_diff"]["mean"] for t in traj]); e = np.array([t["eo_diff"]["ci95"] for t in traj])
        axs[0].plot(r, m, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.5, label=name)
        axs[0].fill_between(r, m - e, m + e, color=c, alpha=0.10, lw=0)
        u = [t["cum_unfair_reject_B"]["mean"] for t in traj]
        axs[1].plot(r, u, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.5, label=name)
        axs[1].annotate(f"{u[-1]:,.0f}".replace(",", " "), (r[-1], u[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=11, color=INK)
    axs[0].axhline(0, color=INK2, lw=1)
    comma_axes(axs[0], x=False)
    axs[0].set_ylabel("Bərabər imkan fərqi (TPR_A − TPR_B)")
    axs[1].set_ylabel("Rədd edilən etibarlı B müraciətçiləri (cəm)")
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", " ")))
    for ax in axs:
        ax.set_xlabel("Dövr (hər dövrdə 5 000 yeni müraciət)")
        ax.set_xticks(range(1, len(traj) + 1))
    axs[1].set_xlim(0.5, len(traj) + 1.3)
    axs[0].legend(loc="upper right", fontsize=10.5)
    fig.tight_layout(w_pad=3)
    fig.savefig(FIG / "f5_feedback.png"); plt.close(fig)


# --------------------------------------------------------------------- F6: German Credit
def fig_german():
    d = load("e6_german")
    keys = [k for k in d if " | " in k]
    lab = [k.replace("Bütün əlamətlər", "Bütün\nəlamətlər").replace("Yaş və cins çıxarılıb", "Yaş, cins\nçıxarılıb")
           .replace(" | ", "\n").replace("ExpGrad (EO, yaş) + LR", "ExpGrad (EO)") for k in keys]
    y1 = [d[k]["young"]["di_ratio"]["mean"] for k in keys]; e1 = [d[k]["young"]["di_ratio"]["ci95"] for k in keys]
    y2 = [d[k]["sex"]["di_ratio"]["mean"] for k in keys]; e2 = [d[k]["sex"]["di_ratio"]["ci95"] for k in keys]
    x = np.arange(len(keys)); w = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.bar(x - w/2 - 0.01, y1, w, color=COL_A, label="Gənclər (≤25) / yaşlılar", yerr=e1, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.bar(x + w/2 + 0.01, y2, w, color=COL_B, label="Qadınlar / kişilər", yerr=e2, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    for i in range(len(keys)):
        ax.text(x[i] - w/2, y1[i] + e1[i] + 0.02, az(y1[i]), ha="center", fontsize=10.5, color=INK)
        ax.text(x[i] + w/2, y2[i] + e2[i] + 0.02, az(y2[i]), ha="center", fontsize=10.5, color=INK)
    ax.axhline(0.8, color="#B23A2E", lw=1.5, label="4/5 qaydası (0,8)")
    ax.set_xticks(x, lab, fontsize=10.5)
    ax.set_ylabel("Təsdiq nisbəti (DI)")
    ax.set_ylim(0, 1.25)
    comma_axes(ax, x=False)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(FIG / "f6_german.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_scenarios, fig_shap_proxy, fig_twin, fig_mitigation, fig_feedback, fig_german):
        try:
            f(); print("OK", f.__name__)
        except FileNotFoundError as e:
            print("SKIP", f.__name__, e)
