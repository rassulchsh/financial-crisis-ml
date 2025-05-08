from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ───────────────────────── logging ──────────────────────────
LOG_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=LOG_FMT,
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("quick_clean_panel")

# ───────────────────────── constants ────────────────────────
TARGET = "crisisJST"
KEYS = ["iso", "year", "crisisID"]
NEWS_VARS = ["avg_label", "n_articles"]

# ───────────────────────── helpers ──────────────────────────


def ensure_news_vars(df: pd.DataFrame) -> None:
    for c in NEWS_VARS:
        if c not in df.columns:
            log.info("Column %-12s absent → initialised with NaN", c)
            df[c] = np.nan


def force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns.difference(KEYS):
        if df[col].dtype == "object":
            ser = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d\.\-\+eE]", "", regex=True)   # strip junk
            )
            df[col] = pd.to_numeric(ser, errors="coerce")
    return df


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Country-wise ffill/bfill then median-impute remaining NaNs."""
    num = df.select_dtypes("number").columns.difference([TARGET])
    df = df.sort_values(["iso", "year"])
    df[num] = df.groupby("iso")[num].ffill()
    df[num] = df.groupby("iso")[num].bfill()
    df[num] = df[num].fillna(df[num].median())
    return df


def vif_prune(df: pd.DataFrame, thresh: float = 5.0) -> List[str]:
    """Iteratively drop numeric columns with VIF > *thresh*."""
    to_check = (
        df.select_dtypes("number")
        .drop(columns=[TARGET], errors="ignore")
        .loc[:, df.std(numeric_only=True) > 0]
        .columns
        .tolist()
    )
    removed: List[str] = []

    while True:
        X = df[to_check].values
        vifs = [variance_inflation_factor(X, i) for i in range(len(to_check))]
        worst_vif = max(vifs)
        if worst_vif < thresh:
            break
        culprit = to_check[vifs.index(worst_vif)]
        removed.append(culprit)
        to_check.remove(culprit)

    if removed:
        log.info("Dropped %d high-VIF cols (>%.1f): %s",
                 len(removed), thresh, ", ".join(removed))
        df = df.drop(columns=removed)

    return df


# ───────────────────────── workflow ─────────────────────────
def clean_panel(in_path: Path,
                out_path: Path,
                skip_vif: bool = False) -> None:

    log.info("Loading %s", in_path)
    df = pd.read_csv(in_path, low_memory=False)

    # 0. guarantee news-factor columns
    df = ensure_news_vars(df)

    # 1. numeric coercion
    df = force_numeric(df)

    # 2. drop post-crisis
    if "remove" in df.columns:
        pre = len(df)
        df = df[df.remove == 0].drop(columns=["remove"])
        log.info("Dropped %d post-crisis rows (remove==1)", pre - len(df))

    # 3. fill missing macro/news values
    df = fill_gaps(df)

    # 4. VIF pruning
    if not skip_vif:
        df = vif_prune(df)

    # 5. save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved cleaned panel → %s  (%d rows, %d columns)",
             out_path, *df.shape)
    log.info("Ready for modelling – add scaling / one-hot etc. in sklearn pipeline.")


# ───────────────────────── CLI ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast cleaner for jst_enriched panel.")
    ap.add_argument("--in-file",  type=Path,
                    default="code/data_pipeline/jst_enriched.csv")
    ap.add_argument("--out-file", type=Path,
                    default="code/data_pipeline/jst_clean_revised.csv")
    ap.add_argument("--skip-vif", action="store_true",
                    help="Skip VIF pruning (faster)")
    args = ap.parse_args()

    clean_panel(args.in_file, args.out_file, skip_vif=args.skip_vif)


if __name__ == "__main__":
    main()
