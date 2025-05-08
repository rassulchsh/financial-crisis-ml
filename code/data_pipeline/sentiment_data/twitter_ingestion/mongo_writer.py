from pymongo import MongoClient, errors as PyMongoErrors
from pymongo.collection import Collection
import config
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def connect_mongo() -> Collection | None:
    """Connects to MongoDB and returns the collection object, ensuring necessary indexes."""
    try:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
        # Validate connection
        client.admin.command('ismaster')

        db = client[config.DATABASE_NAME]
        collection = db[config.COLLECTION_NAME]

        # Ensure compound index for fast sentiment_label + source queries
        collection.create_index(
            [("sentiment_label", 1), ("source", 1)],
            background=True,
            name="lbl_src_idx"
        )

        logger.info(
            f"MongoDB connected to {config.DATABASE_NAME}/{config.COLLECTION_NAME}, indexes ensured.")
        return collection

    except PyMongoErrors.ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An error occurred during MongoDB connection: {e}", exc_info=True)
        return None


def insert_documents(collection: Collection, docs: List[Dict[str, Any]]) -> int:
    """
    Inserts multiple documents into the collection, handling duplicates gracefully.
    Returns the number of new documents inserted.
    """
    if not docs:
        logger.info("No documents provided for insertion.")
        return 0

    inserted_count = 0
    duplicate_count = 0

    try:
        result = collection.insert_many(docs, ordered=False)
        inserted_count = len(result.inserted_ids)
        logger.debug(f"Inserted {inserted_count} new documents.")

    except PyMongoErrors.BulkWriteError as bwe:
        write_errors = bwe.details.get('writeErrors', [])
        duplicates = sum(1 for err in write_errors if err.get('code') == 11000)
        other_errors = len(write_errors) - duplicates

        inserted_count = bwe.details.get('nInserted', 0)
        duplicate_count = duplicates

        logger.warning(
            f"Bulk write error: inserted={inserted_count}, duplicates_skipped={duplicates}, other_errors={other_errors}"
        )

    except PyMongoErrors.PyMongoError as pme:
        logger.error(f"PyMongo error during insert_many: {pme}", exc_info=True)
        return 0
    except Exception as e:
        logger.error(
            f"Unexpected error during insert_documents: {e}", exc_info=True)
        return 0

    logger.info(
        f"Insertion complete. New: {inserted_count}, Duplicates skipped: {duplicate_count}")
    return inserted_count
