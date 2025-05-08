from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Union

import torch
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from tqdm import tqdm
from code.training.sentiment_factory import get_predictor

# ───────── config ──────────
DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB = "sentiment_news"
DEFAULT_COLL = "historical_articles"
BATCH_SIZE = 256
POST_FIELD = "sentiment_at"
LABEL_FIELD = "sentiment_label"

# ───────── logging ─────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("news_scoring")

# ───────── helpers ─────────


def connect(uri: str, db: str, coll: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client, client[db][coll]


def batched(iterable, size: int):
    """Yield successive lists of length ≤ size."""
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ───────── main routine ─────────
def score_articles(
    coll,
    predictor,
    batch_size: int,
    force: bool,
):
    query = {
        "language": "en",
        "country_iso_alpha2": {"$nin": [None, ""]},
    }
    if not force:
        query[POST_FIELD] = {"$exists": False}

    total = coll.count_documents(query)
    if total == 0:
        log.info("No articles need sentiment scoring – done.")
        return

    log.info("Articles to score: %d", total)

    cursor = coll.find(query, batch_size=batch_size, projection=[
                       "_id", "title", "body"])

    stats = {"scored": 0, "updated": 0, "failed": 0}
    now = datetime.utcnow()

    with tqdm(total=total, unit="doc", desc="scoring") as bar:
        for docs in batched(cursor, batch_size):
            ids = [d["_id"] for d in docs]
            texts = [
                f"{d.get('title','')} {d.get('body','')}" for d in docs]

            try:
                labels: List[int] = predictor(texts)
            except Exception as e:
                log.error("Prediction failure on batch starting _id=%s : %s",
                          ids[0], e, exc_info=True)
                stats["failed"] += len(docs)
                bar.update(len(docs))
                continue

            ops = [
                UpdateOne(
                    {"_id": _id},
                    {"$set": {LABEL_FIELD: lbl, POST_FIELD: now}},
                )
                for _id, lbl in zip(ids, labels)
            ]

            try:
                res = coll.bulk_write(ops, ordered=False)
                stats["updated"] += res.modified_count
            except BulkWriteError as bwe:
                log.error("Bulk-write error: %s", bwe.details)
            except PyMongoError as pme:
                log.error("Mongo error: %s", pme)

            stats["scored"] += len(docs)
            bar.update(len(docs))

    # summary
    log.info("--- Sentiment scoring summary ---")
    for k, v in stats.items():
        log.info("%-8s : %d", k, v)


# ───────── CLI ─────────
def cli():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Attach sentiment labels to cleaned news articles.",
    )
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", DEFAULT_URI))
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--coll", default=DEFAULT_COLL)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--model", choices=["finbert", "baseline"],
                    default="finbert", help="Which sentiment model to use")
    ap.add_argument("--force", action="store_true",
                    help="Re-score even if sentiment already present")
    args = ap.parse_args()

    log.info("---------- news_scoring ----------")
    log.info("Mongo   : %s / %s / %s", args.mongo_uri, args.db, args.coll)
    log.info("Model   : %s", args.model)
    log.info("Batch   : %d", args.batch)
    if args.force:
        log.info("Force   : true (overwrite existing labels)")

    client = None
    try:
        predictor = get_predictor(args.model)
        _ = predictor("smoke test")

        client, coll = connect(args.mongo_uri, args.db, args.coll)
        score_articles(coll, predictor, args.batch, args.force)

        log.info("------------ finished ------------")
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        log.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    cli()
