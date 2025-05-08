from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pycountry
import spacy
from langdetect import detect, LangDetectException
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError, BulkWriteError
from tqdm import tqdm

# ─────────────────────────── Config ────────────────────────────
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "sentiment_news"
DEFAULT_COLL_NAME = "historical_articles"

SPACY_MODEL = "en_core_web_sm"
BATCH_SIZE_DEFAULT = 500

COUNTRY_FIELD = "country_iso_alpha2"
LANGUAGE_FIELD = "language"
PROCESSED_FLAG_FIELD = "processed_at_cleaner"

LOG_FILE = "gdelt_cleaner.log"
BREAK_TIES_ALPHA = True
# ─────────────────────────── Logging ───────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"))
logging.getLogger("").addHandler(_console)
logger = logging.getLogger(__name__)

# ───────────────────────── Mongo helpers ───────────────────────


def get_mongo_collection(uri: str, db_name: str, coll_name: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    db = client[db_name]
    coll = db[coll_name]
    coll.create_index([(PROCESSED_FLAG_FIELD, 1)],
                      sparse=True, background=True)
    logger.info("MongoDB ✅  %s/%s", db_name, coll_name)
    return coll, client

# ───────────────────────── Resources  ──────────────────────────


def load_spacy(model_name: str = SPACY_MODEL) -> spacy.language.Language:
    nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    nlp.max_length = 200_000
    logger.info("spaCy model loaded: %s (max_length=%d)",
                model_name, nlp.max_length)
    return nlp


def build_country_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return two dicts: {name→iso2}, {demonym→iso2}."""
    name2 = {}
    demonym2 = {}

    for c in pycountry.countries:
        code = c.alpha_2
        name2[c.name.lower()] = code
        if hasattr(c, "common_name"):
            name2[c.common_name.lower()] = code
        if hasattr(c, "official_name"):
            name2[c.official_name.lower()] = code

    name2.update({
        "united states": "US", "u.s.": "US", "u.s.a.": "US", "usa": "US",
        "united kingdom": "GB", "great britain": "GB", "uk": "GB", "u.k.": "GB",
        "south korea": "KR", "north korea": "KP",
    })
    demonyms = {
        "american": "US", "british": "GB", "german": "DE", "french": "FR",
        "russian": "RU", "chinese": "CN", "japanese": "JP", "canadian": "CA",
        "australian": "AU", "indian": "IN", "pakistani": "PK", "turkish": "TR",
        "spanish": "ES", "italian": "IT",
    }
    demonym2.update(demonyms)
    logger.info("Country lookup built: %d names, %d demonyms",
                len(name2), len(demonym2))
    return name2, demonym2


# ──────────────────────── NLP utilities ────────────────────────
_url_re = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
_ws_re = re.compile(r"\s+")


def clean_text_basic(text: Optional[str]) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text = text.lower()
    text = _url_re.sub("", text)
    text = _ws_re.sub(" ", text).strip()
    return text


def detect_language(text: str) -> Optional[str]:
    """Return ISO-639-1 language code or None."""
    snippet = clean_text_basic(text)
    if not snippet or len(snippet) < 20:
        return None
    try:
        return detect(snippet)
    except LangDetectException:
        return None
    except Exception as e:
        logger.debug("langdetect error: %s", e)
        return None


def extract_primary_country(
    doc: Dict[str, Any],
    nlp: spacy.language.Language,
    name_lookup: Dict[str, str],
    demonym_lookup: Dict[str, str],
) -> Optional[str]:
    joined = f"{doc.get('title','')}. {doc.get('body','')}"
    cleaned = clean_text_basic(joined)
    if not cleaned:
        return None

    try:
        spacy_doc = nlp(cleaned[: nlp.max_length])
    except Exception as e:
        logger.warning("spaCy failure on _id=%s : %s", doc.get("_id"), e)
        return None

    mentions = []
    for ent in spacy_doc.ents:
        if ent.label_ != "GPE":
            continue
        token = ent.text.lower()
        if token in name_lookup:
            mentions.append(name_lookup[token])
        elif token in demonym_lookup:
            mentions.append(demonym_lookup[token])
        else:
            for name, code in name_lookup.items():
                if re.search(rf"\b{name}\b", token):
                    mentions.append(code)
                    break

    if not mentions:
        return None

    counts = Counter(mentions).most_common()
    top, top_cnt = counts[0]
    if len(counts) > 1 and counts[1][1] == top_cnt:
        if BREAK_TIES_ALPHA:
            tied = [code for code, cnt in counts if cnt == top_cnt]
            return sorted(tied)[0]
        return None
    return top

# ───────────────────────── Core routine ────────────────────────


def process_articles(
    coll,
    nlp,
    name_lookup,
    demonym_lookup,
    batch_size: int,
    limit: Optional[int],
    force: bool,
) -> None:
    if force:
        query = {}
    else:
        query = {
            "$or": [
                {PROCESSED_FLAG_FIELD: {"$exists": False}},
                {COUNTRY_FIELD: {"$exists": False}},
                {COUNTRY_FIELD: None},
                {COUNTRY_FIELD: ""},
            ]
        }

    total = coll.count_documents(query)
    if limit:
        total = min(total, limit)
    if total == 0:
        logger.info("No articles match the query – all done.")
        return

    logger.info("Articles to process: %d", total)
    cursor = coll.find(query, batch_size=batch_size).limit(limit or 0)

    ops: List[UpdateOne] = []
    now = datetime.utcnow()

    stats = Counter()
    with tqdm(total=total, unit="doc", desc="Cleaning") as bar:
        for doc in cursor:
            stats["seen"] += 1
            text_combo = f"{doc.get('title','')} {doc.get('body','')}"
            lang = detect_language(text_combo) or "und"
            if lang == "en":
                stats["en"] += 1
                country = extract_primary_country(
                    doc, nlp, name_lookup, demonym_lookup)
                if country:
                    stats["country"] += 1
            else:
                stats["non_en"] += 1
                country = None

            update = {
                "$set": {
                    LANGUAGE_FIELD: lang,
                    COUNTRY_FIELD: country,
                    PROCESSED_FLAG_FIELD: now,
                }
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, update))

            if len(ops) >= batch_size:
                _flush_bulk(coll, ops, stats)
                bar.update(len(ops))
                ops = []

        if ops:
            _flush_bulk(coll, ops, stats)
            bar.update(len(ops))

    logger.info("--- Cleaner summary ---")
    for k, v in stats.items():
        logger.info("%-15s : %d", k, v)


def _flush_bulk(coll, ops, stats):
    try:
        res = coll.bulk_write(ops, ordered=False)
        stats["updated"] += res.modified_count
    except BulkWriteError as bwe:
        stats["failed"] += len(ops) - bwe.details.get("nModified", 0)
        logger.error("Bulk write error: %s", bwe.details)
    except PyMongoError as pme:
        stats["failed"] += len(ops)
        logger.error("Mongo error: %s", pme)

# ──────────────────────────   CLI   ────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect language & country for GDELT news articles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--mongo-uri", default=os.getenv("MONGO_URI", DEFAULT_MONGO_URI))
    ap.add_argument("--db-name", default=DEFAULT_DB_NAME)
    ap.add_argument("--coll-name", default=DEFAULT_COLL_NAME)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force-reprocess", action="store_true")
    ap.add_argument("--spacy-model", default=SPACY_MODEL)

    args = ap.parse_args()

    logger.info("------- gdelt_cleaner starting -------")
    logger.info("Mongo   : %s / %s / %s", args.mongo_uri,
                args.db_name, args.coll_name)
    logger.info("Batch   : %d", args.batch_size)
    if args.limit:
        logger.info("Limit   : %d", args.limit)
    if args.force_reprocess:
        logger.info("Force   : True (ignoring previous processed flag)")

    client = None
    try:
        nlp = load_spacy(args.spacy_model)
        name_lookup, demonym_lookup = build_country_lookup()
        coll, client = get_mongo_collection(
            args.mongo_uri, args.db_name, args.coll_name)

        process_articles(
            coll,
            nlp,
            name_lookup,
            demonym_lookup,
            args.batch_size,
            args.limit,
            args.force_reprocess,
        )

        logger.info("------- gdelt_cleaner finished -------")
    except (PyMongoError, OSError, ImportError) as e:
        logger.critical("Fatal init error: %s", e, exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(0)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
