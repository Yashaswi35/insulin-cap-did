"""Build the analysis panel with DuckDB and print descriptive trends.

Writes data/panel.parquet and output/descriptive_trends.csv.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"


REQUIRED = [
    "Prscrbr_Geo_Lvl", "Prscrbr_Geo_Cd", "Prscrbr_Geo_Desc", "Brnd_Name", "Gnrc_Name",
    "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes",
    "GE65_Sprsn_Flag", "GE65_Tot_30day_Fills", "LIS_Bene_Cst_Shr", "NonLIS_Bene_Cst_Shr",
]


def load_raw(con, files):
    """Load each year on its own, clean header names, and stack into table src.

    CMS files differ slightly by year (byte-order marks, stray spaces), which
    breaks a single glob read, so each file is checked separately.
    """
    parts = []
    for i, f in enumerate(files):
        year = int(f.stem[-4:])
        rel = con.read_csv(str(f), all_varchar=True, header=True)
        clean = {c: c.replace("﻿", "").replace('"', "").strip() for c in rel.columns}
        missing = [c for c in REQUIRED if c not in clean.values()]
        if missing:
            head = f.read_bytes()[:300]
            raise SystemExit(
                f"\n{f.name} is missing columns {missing[:4]}...\n"
                f"Size: {f.stat().st_size / 1e6:.1f} MB. First bytes: {head!r}\n"
                f"Delete it and rerun src/01_download.py --years {year}"
            )
        cols = ", ".join(f'"{old}" AS {new}' for old, new in clean.items() if new in REQUIRED)
        con.register(f"r{i}", rel)
        parts.append(f"SELECT {year} AS year, {cols} FROM r{i}")
    con.execute("CREATE OR REPLACE TABLE src AS " + " UNION ALL ".join(parts))
    print(con.sql("SELECT year, count(*) n_rows FROM src GROUP BY 1 ORDER BY 1").df().to_string(index=False))


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(RAW.glob("partd_geo_drug_*.csv"))
    if not files:
        raise SystemExit("No raw files. Run src/01_download.py first.")
    print("Years found:", [f.stem[-4:] for f in files])

    con = duckdb.connect()
    load_raw(con, files)
    con.execute((ROOT / "sql" / "build_panel.sql").read_text())

    # Sanity checks: what got classified, and what diabetes-looking names were left out
    print("\nRows by class (state x generic x year, before balancing):")
    print(con.sql("SELECT drug_class, count(*) n, count(DISTINCT generic) generics FROM panel_raw GROUP BY 1 ORDER BY 1").df().to_string(index=False))
    print("\nInsulin generics kept:")
    print(con.sql("SELECT DISTINCT generic FROM panel WHERE insulin = 1 ORDER BY 1").df().to_string(index=False))

    stats = con.sql("""
        SELECT count(DISTINCT state_fips) states, count(DISTINCT generic) generics,
               count(DISTINCT unit) units, count(*) n_rows,
               count(DISTINCT unit) FILTER (WHERE insulin = 1) insulin_units
        FROM panel""").df()
    print("\nBalanced panel:\n", stats.to_string(index=False))

    con.execute(f"COPY panel TO '{ROOT / 'data' / 'panel.parquet'}' (FORMAT parquet)")

    trends = con.sql("""
        SELECT year,
               CASE WHEN insulin = 1 THEN 'insulin' ELSE 'control' END AS grp,
               sum(fills)                  AS fills,
               sum(claims)                 AS claims,
               sum(oop) / sum(fills)       AS oop_per_fill,
               sum(nonlis_oop) / sum(fills) AS nonlis_oop_per_fill,
               sum(drug_cost) / sum(fills) AS cost_per_fill
        FROM panel GROUP BY 1, 2 ORDER BY 2, 1""").df()
    trends.to_csv(OUT / "descriptive_trends.csv", index=False)
    print("\nDescriptive trends:\n", trends.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
