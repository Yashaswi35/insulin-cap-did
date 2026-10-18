"""Difference-in-differences evaluation of the 2023 Medicare Part D $35 insulin cap.

Unit: state x generic drug. Treated: insulin. Control: non-GLP-1 diabetes drugs.
Fixed effects: unit, plus state x year (absorbs anything that hit a whole state
in a given year, such as enrollment shifts or COVID). SEs clustered by state.
Weights: each cell's claims in the first sample year, so the estimate reflects
where the volume is. Unweighted results are reported as a robustness check.

Outputs in output/: results.csv, results.json, event_study.png
"""
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CAP_YEAR = 2023
REF_YEAR = 2022
FE = "unit + state_fips^year"
VCOV = {"CRV1": "state_fips"}

INK, INK2, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6", "#eb6834"


def load() -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{ROOT / 'data' / 'panel.parquet'}'").df()
    first = df["year"].min()
    w = df.loc[df["year"] == first, ["unit", "claims"]].rename(columns={"claims": "w"})
    df = df.merge(w, on="unit")
    df["post"] = (df["year"] >= CAP_YEAR).astype(int)
    df["treat_post"] = df["insulin"] * df["post"]
    return df


def did(df, y, label, weights="w", treat="treat_post"):
    d = df.dropna(subset=[y])
    m = pf.feols(f"{y} ~ {treat} | {FE}", data=d, vcov=VCOV, weights=weights)
    b, se = m.coef()[treat], m.se()[treat]
    lo, hi = m.confint().loc[treat]
    return {
        "spec": label, "outcome": y, "coef": b, "se": se, "p": m.pvalue()[treat],
        "ci_lo": lo, "ci_hi": hi, "pct": 100 * np.expm1(b) if y.startswith("log") else np.nan,
        "pct_lo": 100 * np.expm1(lo) if y.startswith("log") else np.nan,
        "pct_hi": 100 * np.expm1(hi) if y.startswith("log") else np.nan,
        "n": m._N, "units": d["unit"].nunique(), "states": d["state_fips"].nunique(),
    }


def event_study(df, y):
    d = df.dropna(subset=[y])
    m = pf.feols(f"{y} ~ i(year, insulin, ref={REF_YEAR}) | {FE}", data=d, vcov=VCOV, weights="w")
    ci = m.confint()
    rows = [{"year": REF_YEAR, "coef": 0.0, "lo": 0.0, "hi": 0.0}]
    for name, b in m.coef().items():
        yr = int(str(name).split("::")[1].split(":")[0].replace("]", ""))
        rows.append({"year": yr, "coef": b, "lo": ci.loc[name].iloc[0], "hi": ci.loc[name].iloc[1]})
    es = pd.DataFrame(rows).sort_values("year")
    # joint test that all pre-period coefficients are zero
    pre = [n for n in m.coef().index if int(str(n).split("::")[1].split(":")[0].replace("]", "")) < REF_YEAR]
    p_pre = np.nan
    if pre:
        b = m.coef()[pre].values
        V = m._vcov[np.ix_([list(m.coef().index).index(p) for p in pre], [list(m.coef().index).index(p) for p in pre])]
        wald = float(b @ np.linalg.solve(V, b))
        from scipy.stats import chi2
        p_pre = float(chi2.sf(wald, len(pre)))
    return es, p_pre


def plot(es_fills, es_oop, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [
        (axes[0], es_oop, BLUE, "Out-of-pocket cost per 30-day fill"),
        (axes[1], es_fills, BLUE, "30-day fills"),
    ]
    for ax, es, color, title in panels:
        ax.set_facecolor(SURFACE)
        pct = 100 * np.expm1(es[["coef", "lo", "hi"]])
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(CAP_YEAR - 0.5, color=GRID, lw=1.5, ls="--")
        ax.fill_between(es["year"], pct["lo"], pct["hi"], color=color, alpha=0.15, lw=0)
        ax.plot(es["year"], pct["coef"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(title, loc="left", color=INK, fontsize=12)
        ax.set_ylabel(f"Insulin vs control, % change from {REF_YEAR}", color=INK2, fontsize=9)
        ax.text(CAP_YEAR - 0.45, ax.get_ylim()[1], " $35 cap", va="top", color=INK2, fontsize=9)
        ax.set_xticks(es["year"])
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
    fig.text(0.01, 0.01, "Event study, state x drug panel, unit and state x year fixed effects. "
             "Shaded band is the 95% CI, SEs clustered by state.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    years = sorted(df["year"].unique())
    print(f"Panel: {df['unit'].nunique()} state x drug units, {df['state_fips'].nunique()} states, years {years}")
    if CAP_YEAR not in years:
        raise SystemExit("No post-cap year in the data.")

    res = []
    # 1. First stage: did the cap actually lower what beneficiaries paid for insulin?
    res.append(did(df, "log_oop_per_fill", "Main: OOP per fill"))
    # 2. Main outcome: did insulin use rise relative to other diabetes drugs?
    res.append(did(df, "log_fills", "Main: 30-day fills"))
    res.append(did(df, "log_claims", "Main: claims"))

    # 3. Robustness
    res.append(did(df, "log_fills", "Unweighted", weights=None))
    no_sglt2 = df[df["drug_class"] != "sglt2"]
    res.append(did(no_sglt2, "log_fills", "Controls exclude SGLT2"))
    res.append(did(df[df["year"] != 2020], "log_fills", "Drop 2020 (COVID)"))

    # 4. Placebo in time: pretend the cap started in 2021, pre-cap years only.
    #    Caveat: the voluntary Senior Savings Model capped insulin in some plans from 2021,
    #    so a small positive placebo is possible and would bias the main estimate toward zero.
    pre = df[df["year"] < CAP_YEAR].copy()
    if pre["year"].nunique() >= 3:
        pre["fake"] = pre["insulin"] * (pre["year"] >= 2021).astype(int)
        res.append(did(pre, "log_fills", "Placebo: fake 2021 start", treat="fake"))

    # 5. Placebo treatment: give the 'cap' to DPP-4s instead, insulin removed. Expect ~0.
    ctrl = df[df["insulin"] == 0].copy()
    ctrl["fake"] = (ctrl["drug_class"] == "dpp4").astype(int) * ctrl["post"]
    res.append(did(ctrl, "log_fills", "Placebo: DPP-4 as fake treated", treat="fake"))

    # 6. Dose response by age. Under-65 Medicare enrollees qualify through disability and
    #    are far more likely to get the low-income subsidy, which already capped copays.
    #    Expect a smaller effect for under-65 than for 65+.
    age = df[df["age_split_ok"] & (df["u65_fills"] > 0) & (df["ge65_fills"] > 0)].copy()
    keep = age.groupby("unit")["year"].transform("count") == len(years)
    age = age[keep]
    age["log_ge65_fills"] = np.log(age["ge65_fills"])
    age["log_u65_fills"] = np.log(age["u65_fills"])
    res.append(did(age, "log_ge65_fills", "Age 65+ fills"))
    res.append(did(age, "log_u65_fills", "Under 65 fills"))

    table = pd.DataFrame(res)
    table.to_csv(OUT / "results.csv", index=False)
    print("\n", table[["spec", "outcome", "pct", "pct_lo", "pct_hi", "p", "units", "n"]].round(3).to_string(index=False))

    # Event studies
    es_f, p_pre_f = event_study(df, "log_fills")
    es_o, p_pre_o = event_study(df, "log_oop_per_fill")
    es_f.to_csv(OUT / "event_study_fills.csv", index=False)
    es_o.to_csv(OUT / "event_study_oop.csv", index=False)
    plot(es_f, es_o, OUT / "event_study.png")
    print(f"\nPre-trend joint test p-value: fills {p_pre_f:.3f}, OOP {p_pre_o:.3f}")

    # Headline numbers for the brief and the resume
    main_f = table.set_index("spec").loc["Main: 30-day fills"]
    post_ins = df[(df["insulin"] == 1) & (df["post"] == 1)]
    extra = post_ins["fills"].sum() * (1 - 1 / np.exp(main_f["coef"]))
    oop = df[df["insulin"] == 1].groupby("year").apply(lambda g: g["oop"].sum() / g["fills"].sum(), include_groups=False)
    summary = {
        "years": [int(y) for y in years],
        "states": int(df["state_fips"].nunique()),
        "units": int(df["unit"].nunique()),
        "drugs": int(df["generic"].nunique()),
        "insulin_generics": int(df.loc[df["insulin"] == 1, "generic"].nunique()),
        "claims_in_panel": float(df["claims"].sum()),
        "fills_effect_pct": float(main_f["pct"]),
        "fills_effect_ci": [float(main_f["pct_lo"]), float(main_f["pct_hi"])],
        "fills_effect_p": float(main_f["p"]),
        "oop_effect_pct": float(table.set_index("spec").loc["Main: OOP per fill", "pct"]),
        "insulin_oop_per_fill_by_year": {int(k): round(float(v), 2) for k, v in oop.items()},
        "implied_extra_insulin_fills_post": float(extra),
        "pretrend_p_fills": p_pre_f,
        "pretrend_p_oop": p_pre_o,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
