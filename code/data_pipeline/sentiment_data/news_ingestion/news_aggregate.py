from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from tqdm import tqdm

# ───────── Config ─────────
DEF_URI = "mongodb://localhost:27017"
DEF_DB = "sentiment_news"
DEF_SOURCE = "historical_articles"
DEF_TARGET = "country_year_sentiment"

LOG = logging.getLogger("news_aggregate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def connect(uri: str, db: str):
    cli = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    cli.admin.command("ping")
    return cli, cli[db]


def aggregate(coll_src, coll_tgt):
    """
    Build {country, year} aggregates with Mongo aggregation pipeline
    and bulk-write them to `coll_tgt`.
    """
    LOG.info("Running aggregation pipeline inside MongoDB …")
    pipe = [
        # 1 ▸ filter cleaned & scored docs
        {
            "$match": {
                "language": "en",
                "country_iso_alpha2": {"$nin": [None, ""]},
                "sentiment_label": {"$in": [-1, 0, 1]},
            }
        },
        # 2 ▸ project needed fields + derive 'year'
        {
            "$project": {
                "_id": 0,
                "country": "$country_iso_alpha2",
                "year": {"$year": "$published_date"},
                "sentiment_label": 1,
            }
        },
        # 3 ▸ group
        {
            "$group": {
                "_id": {"country": "$country", "year": "$year"},
                "avg_label": {"$avg": "$sentiment_label"},
                "n_articles": {"$sum": 1},
            }
        },
    ]

    docs = list(coll_src.aggregate(pipe, allowDiskUse=True))
    LOG.info("Aggregation produced %d country-year rows.", len(docs))

    if not docs:
        LOG.info("Nothing to write – exiting.")
        return

    ts = datetime.utcnow()
    ops = [
        UpdateOne(
            {
                "country": d["_id"]["country"],
                "year": d["_id"]["year"],
            },
            {
                "$set": {
                    "avg_label": d["avg_label"],
                    "n_articles": d["n_articles"],
                    "updated_at": ts,
                }
            },
            upsert=True,
        )
        for d in docs
    ]

    LOG.info("Writing results to %s …", coll_tgt.full_name)
    try:
        res = coll_tgt.bulk_write(ops, ordered=False)
        LOG.info(
            "Upserted: %d | Modified: %d",
            res.upserted_count,
            res.modified_count,
        )
    except BulkWriteError as bwe:
        LOG.error("Bulk-write error: %s", bwe.details)
    except PyMongoError as pme:
        LOG.error("Mongo error: %s", pme)
    LOG.info("Done.")


def cli():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Aggregate article sentiment into country-year factors.",
    )
    ap.add_argument("--mongo-uri", default=DEF_URI)
    ap.add_argument("--db", default=DEF_DB)
    ap.add_argument("--source-coll", default=DEF_SOURCE)
    ap.add_argument("--target-coll", default=DEF_TARGET)
    args = ap.parse_args()

    LOG.info("--- news_aggregate starting ---")
    LOG.info("Mongo  : %s / %s", args.mongo_uri, args.db)
    LOG.info("Source : %s", args.source_coll)
    LOG.info("Target : %s", args.target_coll)

    client = None
    try:
        client, db = connect(args.mongo_uri, args.db)
        aggregate(db[args.source_coll], db[args.target_coll])
        LOG.info("--- news_aggregate finished ---")
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        LOG.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    cli()
