"""Download CMS Medicare Part D Prescribers by Geography and Drug files.

One CSV per data year (state x brand x generic drug), roughly 20 to 40 MB each.
Tries the CMS catalog first so the newest year (2024, if posted) is picked up,
then falls back to known URLs for 2019 to 2023.

Usage: python src/01_download.py [--years 2019 2020 2021 2022 2023 2024]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TITLE = "Medicare Part D Prescribers - by Geography and Drug"
CATALOG = "https://data.cms.gov/data.json"

KNOWN = {
    2019: "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY19_Geo.csv",
    2020: "https://data.cms.gov/sites/default/files/2022-07/ca71b7df-4d48-4c2d-aded-2ca22285739c/MUP_DPR_RY22_P04_V10_DY20_Geo.csv",
    2021: "https://data.cms.gov/sites/default/files/2023-04/3d3ebd5b-b4bf-45b4-876d-afa7916d1b72/MUP_DPR_RY23_P04_V10_DY21_Geo.csv",
    2022: "https://data.cms.gov/sites/default/files/2024-05/410ed206-d782-4dba-b442-6bcd45ae2016/MUP_DPR_RY24_P04_V10_DY22_Geo.csv",
    2023: "https://data.cms.gov/sites/default/files/2025-04/9fe6b8a6-0cb9-4b7c-9760-87800da010a8/MUP_DPR_RY25_P04_V10_DY23_Geo.csv",
}


def catalog_urls() -> dict:
    """Return {year: csv_url} from the CMS catalog, or {} if it can't be read."""
    try:
        cat = requests.get(CATALOG, timeout=120).json()
    except Exception as e:  # catalog is large and occasionally slow
        print(f"  catalog unavailable ({e}), using known URLs")
        return {}
    out = {}
    for ds in cat.get("dataset", []):
        if ds.get("title", "").strip() != TITLE:
            continue
        for d in ds.get("distribution", []):
            url = d.get("downloadURL", "")
            if not url.lower().endswith(".csv"):
                continue
            m = re.search(r"_DY(\d{2})_", url) or re.search(r"(20\d{2})", d.get("temporal", ""))
            if m:
                y = int(m.group(1))
                out[2000 + y if y < 100 else y] = url
    return out


def valid(path: Path) -> bool:
    """A real file is over 1 MB and its header names the geography column."""
    if path.stat().st_size < 1_000_000:
        return False
    with open(path, "rb") as f:
        return b"Prscrbr_Geo_Lvl" in f.read(500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    urls = {**KNOWN, **catalog_urls()}
    missing = []
    for y in args.years:
        dest = RAW / f"partd_geo_drug_{y}.csv"
        if dest.exists() and valid(dest):
            print(f"{y}: already downloaded")
            continue
        url = urls.get(y)
        if not url:
            missing.append(y)
            continue
        print(f"{y}: downloading {url}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"  saved {dest.stat().st_size / 1e6:.1f} MB")
        if not valid(dest):
            print(f"  WARNING: {dest.name} does not look like the CMS file, skipping it. First bytes: {dest.read_bytes()[:200]!r}")
            dest.unlink()
            missing.append(y)

    (RAW / "sources.json").write_text(json.dumps({y: urls.get(y) for y in args.years}, indent=2))
    if missing:
        print(f"Not posted yet or not found: {missing}. The analysis runs on the years that downloaded.")
    if not any(RAW.glob("partd_geo_drug_*.csv")):
        sys.exit("No files downloaded.")


if __name__ == "__main__":
    main()
