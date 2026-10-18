"""Re-specified designs after the basic DiD failed its pre-trend and placebo checks.

Why: insulin was already losing share to newer diabetes drugs before 2023, and
the control drugs were on different trends from each other (DPP-4 placebo -20%).
A level DiD reads that pre-existing decline as a cap effect.

Design A, trend break. Each drug class gets its own linear trend. The cap effect
is how far insulin's 2023 value sits from its own trend line. Placebos rerun with
the same trends.

Design B, triple difference by age. Under-65 Medicare enrollees qualify through
disability and are mostly on the low-income subsidy, which already capped their
insulin copays, so the cap mostly hit 65+. Comparing 65+ with under-65 within the
same state, drug and year cancels any drug-level trend, including substitution
toward GLP-1s and SGLT2s, because it hits both age groups. Under-65 non-LIS
enrollees were also treated, so B understates the full effect (a lower bound).

Outputs in output/: results_v2.csv, results_v2.json, event_study_v2.png
"""
import json
import warnings
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CAP, REF = 2023, 2022
VCOV = {"CRV1": "state_fips"}
INK, INK2, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6"


def load():
    df = duckdb.sql(f"SELECT * FROM '{ROOT / 'data' / 'panel.parquet'}'").df()
    first = df["year"].min()
    w = df.loc[df["year"] == first, ["unit", "claims"]].rename(columns={"claims": "w"})
    df = df.merge(w, on="unit")
    df["year"] = df["year"].astype(int)
    df["t"] = df["year"] - first
    df["post"] = (df["year"] >= CAP).astype(int)
    df["treat_post"] = df["insulin"] * df["post"]
    return df


def est(m, term, label, extra=None):
    b, se = m.coef()[term], m.se()[term]
    lo, hi = m.confint().loc[term]
    return {"spec": label, "coef": b, "se": se, "p": m.pvalue()[term],
            "pct": 100 * np.expm1(b), "pct_lo": 100 * np.expm1(lo), "pct_hi": 100 * np.expm1(hi),
            "n": m._N, **(extra or {})}


def class_trends(df):
    """Dummy x t columns, one per drug class, so each class gets its own slope."""
    # One control class is the reference slope (the state x year effects absorb the
    # common trend), so metformin gets no column of its own.
    cols = []
    for c in sorted(set(df["drug_class"].unique()) - {"metformin"}):
        col = f"tr_{c}"
        df[col] = (df["drug_class"] == c).astype(int) * df["t"]
        cols.append(col)
    return cols


def design_a(df):
    rows = []
    d = df.dropna(subset=["log_fills"]).copy()
    tr = class_trends(d)
    rhs = " + ".join(tr)
    fe = "unit + state_fips^year"

    for y, lab in [("log_fills", "A: 30-day fills"), ("log_claims", "A: claims"),
                   ("log_oop_per_fill", "A: OOP per fill")]:
        dd = d.dropna(subset=[y])
        m = pf.feols(f"{y} ~ treat_post + {rhs} | {fe}", data=dd, vcov=VCOV, weights="w")
        rows.append(est(m, "treat_post", lab, {"outcome": y}))

    # Senior Savings Model years get their own insulin shift, so the trend is
    # pinned down by 2019 and 2020 only. Stricter, noisier.
    d["ssm"] = d["insulin"] * d["year"].isin([2021, 2022]).astype(int)
    m = pf.feols(f"log_fills ~ treat_post + ssm + {rhs} | {fe}", data=d, vcov=VCOV, weights="w")
    rows.append(est(m, "treat_post", "A: fills, SSM years shifted", {"outcome": "log_fills"}))

    m = pf.feols(f"log_fills ~ treat_post + {rhs} | {fe}", data=d, vcov=VCOV)
    rows.append(est(m, "treat_post", "A: fills, unweighted", {"outcome": "log_fills"}))

    # Placebo in time: fake break in 2022 on pre-cap years. SSM could make this positive.
    pre = d[d["year"] < CAP].copy()
    pre["fake"] = pre["insulin"] * (pre["year"] == 2022).astype(int)
    m = pf.feols(f"log_fills ~ fake + {rhs} | {fe}", data=pre, vcov=VCOV, weights="w")
    rows.append(est(m, "fake", "A placebo: fake 2022 break", {"outcome": "log_fills"}))

    # Placebo treatment: DPP-4 gets the fake break, insulin removed. Expect ~0.
    ctrl = d[d["insulin"] == 0].copy()
    ctrl["fake"] = (ctrl["drug_class"] == "dpp4").astype(int) * ctrl["post"]
    tr_c = [c for c in tr if c != "tr_insulin"]
    m = pf.feols(f"log_fills ~ fake + {' + '.join(tr_c)} | {fe}", data=ctrl, vcov=VCOV, weights="w")
    rows.append(est(m, "fake", "A placebo: DPP-4 fake break", {"outcome": "log_fills"}))
    return rows


def age_long(df):
    a = df[df["age_split_ok"] & (df["ge65_fills"] > 0) & (df["u65_fills"] > 0)].copy()
    n_years = df["year"].nunique()
    a = a[a.groupby("unit")["year"].transform("count") == n_years]
    long = pd.concat([
        a.assign(ge65=1, lf=np.log(a["ge65_fills"])),
        a.assign(ge65=0, lf=np.log(a["u65_fills"])),
    ])
    long["unit_age"] = long["unit"] + "|" + long["ge65"].astype(str)
    long["ddd"] = long["insulin"] * long["post"] * long["ge65"]
    long["ins65"] = long["insulin"] * long["ge65"]
    return long


def design_b(df):
    long = age_long(df)
    # unit^year absorbs every drug-in-state-year shock (substitution, launches, supply).
    # state^year^ge65 absorbs age-group shocks common to all drugs in a state.
    fe = "unit_age + unit^year + state_fips^year^ge65"
    m = pf.feols("lf ~ ddd | " + fe, data=long, vcov=VCOV, weights="w")
    row = est(m, "ddd", "B: triple diff, 65+ vs under 65", {"outcome": "log_fills"})
    row["units"] = int(long["unit"].nunique())

    es = pf.feols(f"lf ~ i(year, ins65, ref={REF}) | {fe}", data=long, vcov=VCOV, weights="w")
    coefs, ci = es.coef(), es.confint()
    idx = list(coefs.index)
    yr = lambda n: int(str(n).split("::")[1].split(":")[0].replace("]", ""))
    pts = [{"year": REF, "coef": 0.0, "lo": 0.0, "hi": 0.0}] + [
        {"year": yr(n), "coef": coefs[n], "lo": ci.loc[n].iloc[0], "hi": ci.loc[n].iloc[1]} for n in idx]
    pre = [n for n in idx if yr(n) < REF]
    sel = [idx.index(n) for n in pre]
    b = coefs[pre].values
    wald = float(b @ np.linalg.solve(es._vcov[np.ix_(sel, sel)], b))
    p_pre = float(chi2.sf(wald, len(pre)))
    return row, pd.DataFrame(pts).sort_values("year"), p_pre


def oop_event_study(df):
    d = df.dropna(subset=["log_oop_per_fill"])
    m = pf.feols(f"log_oop_per_fill ~ i(year, insulin, ref={REF}) | unit + state_fips^year",
                 data=d, vcov=VCOV, weights="w")
    ci = m.confint()
    yr = lambda n: int(str(n).split("::")[1].split(":")[0].replace("]", ""))
    pts = [{"year": REF, "coef": 0.0, "lo": 0.0, "hi": 0.0}] + [
        {"year": yr(n), "coef": b, "lo": ci.loc[n].iloc[0], "hi": ci.loc[n].iloc[1]} for n, b in m.coef().items()]
    return pd.DataFrame(pts).sort_values("year")


def plot(es_oop, es_ddd, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    spec = [
        (axes[0], es_oop, "Insulin out-of-pocket cost per fill", "vs other diabetes drugs"),
        (axes[1], es_ddd, "Insulin fills, age 65+ vs under 65", "triple difference"),
    ]
    for ax, es, title, sub in spec:
        ax.set_facecolor(SURFACE)
        pct = 100 * np.expm1(es[["coef", "lo", "hi"]])
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(CAP - 0.5, color=GRID, lw=1.5, ls="--")
        ax.fill_between(es["year"], pct["lo"], pct["hi"], color=BLUE, alpha=0.15, lw=0)
        ax.plot(es["year"], pct["coef"], color=BLUE, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(f"{title}\n", loc="left", color=INK, fontsize=12)
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9)
        ax.set_ylabel(f"% change relative to {REF}", color=INK2, fontsize=9)
        ax.annotate("$35 cap", xy=(CAP - 0.45, 0.04), xycoords=("data", "axes fraction"),
                    color=INK2, fontsize=9)
        ax.set_xticks(es["year"])
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
    fig.text(0.01, 0.01, "Medicare Part D, state x drug panel. Shaded band is the 95% CI, "
             "SEs clustered by state. Shift in 2021 reflects the Senior Savings Model.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    df = load()
    rows = design_a(df)
    b_row, es_ddd, p_pre_ddd = design_b(df)
    rows.append(b_row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results_v2.csv", index=False)
    print(table[["spec", "pct", "pct_lo", "pct_hi", "p", "n"]].round(3).to_string(index=False))

    es_oop = oop_event_study(df)
    es_ddd.to_csv(OUT / "event_study_ddd.csv", index=False)
    plot(es_oop, es_ddd, OUT / "event_study_v2.png")
    print("\nTriple-diff event study (% vs 2022):")
    print(es_ddd.assign(pct=lambda x: 100 * np.expm1(x["coef"]), lo=lambda x: 100 * np.expm1(x["lo"]),
                        hi=lambda x: 100 * np.expm1(x["hi"]))[["year", "pct", "lo", "hi"]].round(2).to_string(index=False))
    print(f"Triple-diff pre-trend joint test p-value: {p_pre_ddd:.3f}")

    t = table.set_index("spec")
    ins = df[df["insulin"] == 1].groupby("year")
    oop = (ins["oop"].sum() / ins["fills"].sum()).round(2)
    summary = {
        "states": int(df["state_fips"].nunique()),
        "drugs": int(df["generic"].nunique()),
        "insulin_generics": int(df.loc[df["insulin"] == 1, "generic"].nunique()),
        "units": int(df["unit"].nunique()),
        "claims_in_panel": float(df["claims"].sum()),
        "insulin_oop_per_fill_by_year": {int(k): float(v) for k, v in oop.items()},
        "A_fills_pct": float(t.loc["A: 30-day fills", "pct"]),
        "A_fills_ci": [float(t.loc["A: 30-day fills", "pct_lo"]), float(t.loc["A: 30-day fills", "pct_hi"])],
        "A_placebo_2022_pct": float(t.loc["A placebo: fake 2022 break", "pct"]),
        "A_placebo_dpp4_pct": float(t.loc["A placebo: DPP-4 fake break", "pct"]),
        "B_ddd_pct": float(t.loc["B: triple diff, 65+ vs under 65", "pct"]),
        "B_ddd_ci": [float(t.loc["B: triple diff, 65+ vs under 65", "pct_lo"]),
                     float(t.loc["B: triple diff, 65+ vs under 65", "pct_hi"])],
        "B_pretrend_p": p_pre_ddd,
    }
    (OUT / "results_v2.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
