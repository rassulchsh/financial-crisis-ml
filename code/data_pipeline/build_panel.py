from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pycountry
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ───────────────────────── logging
LOG = logging.getLogger("build_training_panel")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ───────────────────────── helpers


def iso2_to_iso3(code: str | None) -> str | None:
    if not code or len(code) != 2:
        return None
    try:
        return pycountry.countries.get(alpha_2=code.upper()).alpha_3
    except Exception:
        return None


def pull_sentiment(uri: str, db: str, coll: str) -> pd.DataFrame:
    """Return DF[iso, year, news_sentiment, n_news_articles]."""
    try:
        cli = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        cli.admin.command("ping")
        LOG.info("MongoDB ✅  %s / %s.%s", uri, db, coll)
    except PyMongoError as e:
        LOG.error("Mongo connection failed: %s", e)
        raise

    cursor = cli[db][coll].find(
        {},
        {
            "_id": 0,
            "country": 1,          # ISO-2
            "year": 1,
            "avg_label": 1,
            "n_articles": 1,
        },
    )
    df = pd.DataFrame(list(cursor))
    cli.close()

    if df.empty:
        LOG.warning("⚠️  sentiment collection is empty")
        return df

    df["iso"] = df.country.map(iso2_to_iso3)
    df = df.dropna(subset=["iso"])
    df = df.rename(
        columns={
            "avg_label": "news_sentiment",
            "n_articles": "n_news_articles",
        }
    )
    df = df[["iso", "year", "news_sentiment", "n_news_articles"]]
    LOG.info("Sentiment factor pulled: %s", df.shape)
    return df


def pull_alt_crisis() -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import altCrisisData as ac

    lv = ac.getLaevenValencia()
    esrb = ac.getESRB()
    return lv, esrb


# ───────────────────────── build routine
def build_panel(
    jst_path: Path,
    macro_path: Path,
    sentiment_df: pd.DataFrame,
    out_path: Path,
) -> None:
    # base JST
    jst = pd.read_excel(jst_path, sheet_name=0)
    LOG.info("JST base: %s", jst.shape)

    # macro add-ons
    macro = pd.read_csv(macro_path)
    macro_keys = {"iso", "year"}
    if not macro_keys.issubset(macro.columns):
        raise ValueError(f"{macro_path} missing keys {macro_keys}")
    LOG.info("Macro merge file: %s", macro.shape)

    # crisis flags
    lv, esrb = pull_alt_crisis()
    LOG.info("Laeven-Valencia: %s – ESRB: %s", lv.shape, esrb.shape)

    # join everything
    panel = (
        jst
        .merge(macro, on=["iso", "year"], how="left", suffixes=("", "_macro"))
        .merge(sentiment_df, on=["iso", "year"], how="left")
        .merge(lv,   on=["iso", "year"], how="left")
        .merge(esrb, on=["iso", "year"], how="left")
        .sort_values(["iso", "year"])
        .reset_index(drop=True)
    )

    # guarantee crisis dummy columns exist
    for col in ["crisis_banking", "crisis_esrb"]:
        if col not in panel.columns:
            panel[col] = pd.NA

    LOG.info("Final enriched panel: %s  (saving…)", panel.shape)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    LOG.info("✅  %s written", out_path)


# ───────────────────────── CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich JST panel with macro, sentiment and alt-crisis data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--jst",   required=True, help="Path to JSTdatasetR6.xlsx")
    ap.add_argument("--macro", required=True,
                    help="merged_macro_data_clean.csv")
    ap.add_argument("--out",   required=True, help="output.csv")
    ap.add_argument("--mongo-uri",  default="mongodb://localhost:27017")
    ap.add_argument("--mongo-db",   default="sentiment_news")
    ap.add_argument("--sent-coll",  default="country_year_sentiment")
    args = ap.parse_args()

    sent_df = pull_sentiment(args.mongo_uri, args.mongo_db, args.sent_coll)
    build_panel(Path(args.jst), Path(args.macro), sent_df, Path(args.out))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:          # noqa
        LOG.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
