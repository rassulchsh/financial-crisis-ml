import pandas as pd
import re
import uuid
import yaml
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
_cfg = yaml.safe_load(open("config/twitter_loader.yaml"))

TICKER_RE = re.compile(_cfg["ticker_regex"])


def _load_raw() -> pd.DataFrame:
    p = _cfg["dataset_path"]
    df = pd.read_csv(os.path.join(p, "sent_train.csv"))
    df = pd.concat(
        [df, pd.read_csv(os.path.join(p, "sent_valid.csv"))], ignore_index=True)
    if _cfg["sample_size"]:
        df = df.sample(_cfg["sample_size"], random_state=42)
    return df


def _extract_tickers(text: str) -> List[str]:
    return list({m.group(0) for m in TICKER_RE.finditer(text)})


def load_kaggle_finance() -> List[Dict[str, Any]]:
    df = _load_raw()
    recs: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        recs.append({
            "_id":             f"kaggle_fin_{idx}",
            "tweet_id":        None,
            "created_at":      None,
            "text":            row["text"],
            "cleaned_text":    None,
            "tickers":         _extract_tickers(row["text"]),
            "companies":       [],
            "sentiment_label": int(row["label"]),
            "source":          "kaggle_fin_news"
        })
    logger.info("Loaded %d Kaggle finance tweets", len(recs))
    return recs
