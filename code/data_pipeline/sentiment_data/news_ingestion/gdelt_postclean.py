from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict

from pymongo import MongoClient, UpdateOne, DeleteOne
from pymongo.errors import BulkWriteError, PyMongoError
from tqdm import tqdm

# ─────────── config ───────────
DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB = "sentiment_news"
DEFAULT_IN_COLL = "historical_articles"
DEFAULT_OUT_COLL = None
BATCH = 1_000

FIELDS_TO_UNSET: List[str] = [
    "country",
    "gdelt_themes",
]

POST_FLAG = "cleaned_at_post"

# ─────────── logging ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("postclean")

# ─────────── helpers ──────────


def connect(uri: str, db: str, coll: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client, client[db][coll]


def build_queries() -> Dict[str, dict]:
    """Queries for good vs bad documents (idempotent)."""
    good = {
        POST_FLAG: {"$exists": False},
        "language": "en",
        "country_iso_alpha2": {"$nin": [None, ""]},
    }
    bad = {"$and": [{"$nor": [good]}, {POST_FLAG: {"$exists": False}}]}
    return {"good": good, "bad": bad}


def process(
    coll_in,
    coll_out,
    copy_mode: bool,
    batch: int,
):
    q = build_queries()
    total_good = coll_in.count_documents(q["good"])
    total_bad = coll_in.count_documents(q["bad"])
    log.info("Docs to **keep** : %d", total_good)
    log.info("Docs to **drop** : %d", total_bad)

    if total_bad:
        log.info("Deleting unwanted documents…")
        bad_cursor = coll_in.find(q["bad"], {"_id": 1}, batch_size=batch)
        ops: List[DeleteOne] = []
        with tqdm(total=total_bad, unit="doc", desc="deleting") as bar:
            for d in bad_cursor:
                ops.append(DeleteOne({"_id": d["_id"]}))
                if len(ops) >= batch:
                    _flush_delete(coll_in, ops)
                    bar.update(len(ops))
                    ops = []
            if ops:
                _flush_delete(coll_in, ops)
                bar.update(len(ops))

    log.info("Cleaning remaining documents…")
    good_cursor = coll_in.find(q["good"], batch_size=batch)
    ops_in: List[UpdateOne] = []
    ops_out: List[dict] = []
    ts = datetime.utcnow()

    with tqdm(total=total_good, unit="doc", desc="cleaning") as bar:
        for doc in good_cursor:
            update = {"$set": {POST_FLAG: ts}, "$unset": {
                f: "" for f in FIELDS_TO_UNSET}}

            if copy_mode:
                cleaned = {k: v for k, v in doc.items()
                           if k not in FIELDS_TO_UNSET}
                cleaned[POST_FLAG] = ts
                ops_out.append(cleaned)
            else:
                ops_in.append(UpdateOne({"_id": doc["_id"]}, update))

            if len(ops_in) >= batch:
                _flush_bulk(coll_in, ops_in)
                bar.update(len(ops_in))
                ops_in = []

            if len(ops_out) >= batch:
                coll_out.insert_many(ops_out, ordered=False)
                bar.update(len(ops_out))
                ops_out = []

        # leftovers
        if ops_in:
            _flush_bulk(coll_in, ops_in)
            bar.update(len(ops_in))
        if ops_out:
            coll_out.insert_many(ops_out, ordered=False)
            bar.update(len(ops_out))

    log.info("Post-clean finished.")


def _flush_bulk(coll, ops):
    try:
        res = coll.bulk_write(ops, ordered=False)
        log.debug("Bulk-update: matched=%d modified=%d",
                  res.matched_count, res.modified_count)
    except BulkWriteError as bwe:
        log.error("Bulk write error: %s", bwe.details)
    finally:
        ops.clear()


def _flush_delete(coll, ops):
    try:
        coll.bulk_write(ops, ordered=False)
    except BulkWriteError as bwe:
        log.error("Bulk delete error: %s", bwe.details)
    finally:
        ops.clear()


# ──────────────── CLI ─────────────────
def cli():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", DEFAULT_URI))
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--in-coll", default=DEFAULT_IN_COLL,
                    help="Collection containing output of gdelt_cleaner.")
    ap.add_argument("--out-coll", default=DEFAULT_OUT_COLL,
                    help="If set, write clean docs into this collection "
                         "instead of updating in place and deleting bad docs.")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="Mongo bulk-op size.")
    args = ap.parse_args()

    client, coll_in = connect(args.mongo_uri, args.db, args.in_coll)
    copy_mode = bool(args.out_coll)

    coll_out = None
    if copy_mode:
        coll_out = client[args.db][args.out_coll]
        coll_out.create_index(POST_FLAG)
        log.info("Copy-mode: cleaned docs will be inserted into %s/%s",
                 args.db, args.out_coll)

    try:
        process(coll_in, coll_out, copy_mode, args.batch)
    finally:
        client.close()


if __name__ == "__main__":
    cli()
