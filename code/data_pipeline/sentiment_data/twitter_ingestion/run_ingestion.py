import logging
import os
import time

# --------------------------------------------------------------------- #
#  Local modules
# --------------------------------------------------------------------- #
from twitter_loader import load_kaggle_finance
from text_cleaner import clean_text
from mongo_writer import connect_mongo, insert_documents

# --------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------- #
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(
            LOG_DIR, "twitter_ingestion_kaggle.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------- #


def main() -> None:
    logger.info("--- Kaggle Twitter-Finance Ingestion Pipeline Started ---")
    t0 = time.time()

    # 1) Load Kaggle data ------------------------------------------------
    tweets = load_kaggle_finance()
    logger.info("Cleaning %d Kaggle tweets", len(tweets))

    # 2) Clean / enrich --------------------------------------------------
    for tw in tweets:
        tw["cleaned_text"] = clean_text(tw["text"])

    # 3) Mongo connection ------------------------------------------------
    collection = connect_mongo()
    if collection is None:
        logger.critical("MongoDB connection failed – aborting.")
        return

    # 4) Insert with duplicate safety -----------------------------------
    inserted = insert_documents(collection, tweets)
    logger.info("Mongo insertion complete – %d new docs added", inserted)

    # 5) Summary ---------------------------------------------------------
    logger.info("--- Pipeline finished in %.1f s ---", time.time() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
