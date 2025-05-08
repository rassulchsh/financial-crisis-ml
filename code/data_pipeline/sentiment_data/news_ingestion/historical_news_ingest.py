from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, date
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from newspaper import Article, Config as NewspaperConfig
from newspaper.article import ArticleException
from requests.exceptions import RequestException
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from tqdm import tqdm


LOG_FILE = "historical_news_ingest.log"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "sentiment_news"
DEFAULT_COLL_NAME = "historical_articles"


ARTICLE_FETCH_LIMIT_PER_DAY = 500
FETCH_DELAY_SECONDS = 0.5

GKG_COLUMNS: List[str] = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "ExtrasXML", "NumMentions",
    "SourceArticleCharCnt", "TranslationInfo_XML",
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
    ],
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger('').addHandler(console_handler)

logger = logging.getLogger(__name__)


newspaper_config = NewspaperConfig()
newspaper_config.browser_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
)
newspaper_config.request_timeout = 15
newspaper_config.fetch_images = False
newspaper_config.memoize_articles = False

MONGO_URI = os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
DB_NAME = DEFAULT_DB_NAME
COLL_NAME = DEFAULT_COLL_NAME


def get_mongo_collection():
    """Connect to MongoDB and return the requested collection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        logger.info("MongoDB connection successful to %s/%s.",
                    DB_NAME, COLL_NAME)
        db = client[DB_NAME]
        collection = db[COLL_NAME]
        collection.create_index("source_url", unique=True, background=True)
        collection.create_index("published_date", background=True)
        logger.info(
            "Ensured indexes exist on 'source_url' and 'published_date'")
        return collection
    except PyMongoError as exc:
        logger.error("MongoDB connection failed (%s): %s", MONGO_URI, exc)
        raise


def generate_daily_dates(start: str, end: str) -> list[str]:
    """Return list of YYYYMMDD strings from *start* to *end* (inclusive)."""
    try:
        d0 = datetime.strptime(start, "%Y%m%d")
        d1 = datetime.strptime(end, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYYMMDD.")
        return []

    if d0 > d1:
        logger.warning(
            "Start date (%s) is after end date (%s). No dates generated.", start, end)
        return []

    result: list[str] = []
    current_date = d0
    while current_date <= d1:
        result.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    return result


def generate_gdelt_url(date_str: str) -> str:
    """Return GDELT GKG zip URL for *date_str* (YYYYMMDD)."""
    return f"http://data.gdeltproject.org/gkg/{date_str}.gkg.csv.zip"


def download_and_extract_gdelt(url: str) -> Optional[pd.DataFrame]:
    """Download *url*, unzip the CSV, return a DataFrame (str columns)."""
    logger.info("Downloading GDELT URL -> %s", url)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except RequestException as exc:
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response.status_code == 404:
            logger.warning(
                "GDELT file not found (404): %s - Likely a future date or missing data.", url)
        else:
            logger.error("Request failed for %s: %s", url, exc)
        return None

    try:
        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            csv_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if not csv_name:
                logger.error("No .csv file found in zip: %s", url)
                return None

            with zf.open(csv_name) as fh:
                first_line_bytes = fh.readline()
                fh.seek(0)
                first_line_str = first_line_bytes.decode(
                    "utf-8", "ignore").upper()
                has_header = first_line_str.startswith("GKGRECORDID\t") or \
                    first_line_str.startswith("DATE\t")

                header_option = 0 if has_header else None
                column_names = None if has_header else range(
                    30)

                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=header_option,
                    names=column_names,
                    dtype=str,
                    low_memory=False,
                    quoting=3,
                    on_bad_lines="warn",
                )
        logger.info("Extracted %d rows from %s", len(df), csv_name)
        return df
    except zipfile.BadZipFile:
        logger.error("Bad zip file downloaded from %s", url)
        return None
    except pd.errors.ParserError as exc:
        logger.error("Pandas parsing error for %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.error("Unpack or parse error for %s: %s", url, exc)
        return None


def process_gdelt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with columns:
       source_url · published_date (date) · tone (float|None) · gdelt_themes (list[str])"""
    if df is None or df.empty:
        return pd.DataFrame()

    col_date = "DATE" if "DATE" in df.columns else None
    col_url = next((c for c in ("DocumentIdentifier",
                                "SOURCEURL", "SOURCEURLS", "SourceUrls")
                    if c in df.columns), None)
    col_tone = "V2Tone" if "V2Tone" in df.columns else (
        "TONE" if "TONE" in df.columns else None)
    col_themes = "V2Themes" if "V2Themes" in df.columns else (
                 "THEMES" if "THEMES" in df.columns else None)

    if col_url is None:
        sample = df.sample(min(len(df), 2000), random_state=7)
        urlish = sample.apply(
            lambda s: s.astype(str).str.contains(r'^https?://', na=False)
        )
        scores = urlish.mean()
        if (scores > 0.10).any():
            col_url = scores.idxmax()

    if col_date is None or col_url is None:
        logger.warning("Could not find DATE and URL columns – skipping frame.")
        return pd.DataFrame()

    subset = {"date_raw": col_date, "url_raw": col_url}
    if col_themes:
        subset["themes_raw"] = col_themes
    if col_tone:
        subset["tone_raw"] = col_tone

    work = df[list(subset.values())].copy()
    work.columns = list(subset.keys())

    work["source_url"] = (
        work["url_raw"].astype(str)
        .str.extract(r'(https?://[^\s;]+)', expand=False)
        .str.strip()
    )
    work = work[work["source_url"].str.contains(
        r'^https?://[^/]+\.[^/]+', na=False)]

    work["published_date"] = (
        pd.to_datetime(work["date_raw"].str[:14],
                       format="%Y%m%d%H%M%S", errors="coerce")
        .fillna(pd.to_datetime(work["date_raw"], format="%Y%m%d", errors="coerce"))
        .dt.date
    )

    if "tone_raw" in work.columns:
        work["tone"] = pd.to_numeric(
            work["tone_raw"].astype(str).str.split(",", n=1).str[0],
            errors="coerce"
        )
    else:
        work["tone"] = None

    if "themes_raw" in work.columns:
        work["gdelt_themes"] = work["themes_raw"].apply(
            lambda x: [t for t in str(x).split(
                ';') if t] if pd.notnull(x) and x else []
        )
    else:
        work["gdelt_themes"] = [[] for _ in range(len(work))]

    tidy = work.dropna(subset=["source_url", "published_date"])[
        ["source_url", "published_date", "tone", "gdelt_themes"]
    ].reset_index(drop=True)

    return tidy


def fetch_article_body(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Download article with *newspaper3k*, return (title, body)."""
    title: Optional[str] = None
    body: Optional[str] = None
    try:
        logger.debug("Fetching article: %s", url)
        art = Article(url, config=newspaper_config)
        art.download()
        art.parse()
        title = art.title.strip() if art.title else None
        body = art.text.strip() if art.text else None
        if title and body:
            logger.debug("Successfully parsed article: %s", url)
            return title, body
        else:
            logger.debug(
                "Article parsing yielded empty title or body for %s", url)
            return None, None
    except ArticleException as exc:
        logger.debug("Newspaper3k ArticleException for %s: %s", url, exc)
        return None, None
    except (RequestException, TimeoutError, ConnectionError) as exc:
        logger.debug("Network/Request error fetching %s: %s", url, exc)
        return None, None
    except Exception as exc:
        logger.warning("Unexpected error fetching/parsing %s: %s (%s)",
                       url, exc, type(exc).__name__, exc_info=False)
        return None, None
    finally:
        time.sleep(FETCH_DELAY_SECONDS)


def ingest_gdelt_for_date(date_str: str, collection) -> None:
    """Downloads, processes, and ingests articles for a single GDELT date."""
    logger.info("Starting ingestion process for date: %s", date_str)
    df_raw = download_and_extract_gdelt(generate_gdelt_url(date_str))
    if df_raw is None or df_raw.empty:
        logger.warning("No raw data extracted for %s, skipping.", date_str)
        return

    df = process_gdelt_dataframe(df_raw)
    if df.empty:
        logger.info("No processable article rows found for %s.", date_str)
        return

    logger.info("%d candidate articles identified for %s", len(df), date_str)

    inserted_count = 0
    attempted_count = 0
    already_exists_count = 0
    fetch_failed_count = 0
    processed_urls: set[str] = set()

    rows_to_process = len(df)
    if rows_to_process == 0:
        logger.info(
            "Zero valid article candidates after processing for %s.", date_str)
        return

    progress_bar = tqdm(df.iterrows(), total=rows_to_process,
                        desc=f"Processing {date_str}", unit="url", leave=True)

    for _, row in progress_bar:
        attempted_count += 1
        raw_url = row["source_url"]

        if not isinstance(raw_url, str) or not raw_url.startswith(('http://', 'https://')):
            logger.debug("Skipping invalid or non-HTTP(S) URL: %s", raw_url)
            continue

        parsed_url = urlparse(raw_url)
        if not parsed_url.netloc:
            logger.debug("Skipping URL with missing domain: %s", raw_url)
            continue

        if raw_url in processed_urls:
            continue
        processed_urls.add(raw_url)

        try:
            if collection.find_one({"source_url": raw_url}, {"_id": 1}):
                logger.debug(
                    "URL already exists in DB, skipping fetch: %s", raw_url)
                already_exists_count += 1
                continue
        except PyMongoError as exc:
            logger.warning(
                "Mongo find_one check failed for %s: %s. Skipping URL.", raw_url, exc)
            continue
        title, body = fetch_article_body(raw_url)

        if not title or not body:
            fetch_failed_count += 1
            continue

        publish_date_obj = row["published_date"]
        publish_datetime = datetime.combine(publish_date_obj, datetime.min.time()) \
            if isinstance(publish_date_obj, date) else None

        if publish_datetime is None:
            logger.warning(
                "Could not determine valid publish datetime for %s, skipping insert.", raw_url)
            continue

        doc = {
            "title": title,
            "body": body,
            "country": None,
            "source_domain": parsed_url.netloc,
            "published_date": publish_datetime,
            "topic": None,
            "tone": row["tone"],
            "gdelt_themes": row["gdelt_themes"],
            "source_url": raw_url,
            "ingested_at": datetime.utcnow(),
        }

        try:
            collection.insert_one(doc)
            inserted_count += 1
            logger.debug("Successfully inserted article from %s", raw_url)
            progress_bar.set_postfix({
                "Inserted": inserted_count,
                "Exists": already_exists_count,
                "Failed": fetch_failed_count
            })
        except PyMongoError as exc:
            logger.warning("Mongo insert failed for %s: %s", raw_url, exc)

    progress_bar.close()
    logger.info(
        "Finished %s – Candidates: %d, Attempted: %d, Inserted: %d, Already Existed: %d, Fetch Failed: %d",
        date_str, len(
            df), attempted_count, inserted_count, already_exists_count, fetch_failed_count
    )


def ingest_gdelt(start: str, end: str, collection):
    """Iterates through dates and calls the daily ingestion function."""
    logger.info("Starting GDELT ingestion range: %s -> %s", start, end)
    date_list = generate_daily_dates(start, end)
    if not date_list:
        logger.error(
            "No dates generated for the specified range. Exiting GDELT ingestion.")
        return
    for day_str in tqdm(date_list, desc="Overall GDELT Progress", unit="day"):
        ingest_gdelt_for_date(day_str, collection)

    logger.info("Completed GDELT ingestion for range: %s -> %s", start, end)


def main():

    global MONGO_URI, DB_NAME, COLL_NAME, FETCH_DELAY_SECONDS

    """Parses arguments and orchestrates the ingestion process."""
    parser = argparse.ArgumentParser(
        description="Ingest historical news into MongoDB from GDELT & Common Crawl. "
                    "Processes 2015-2025 GDELT range by default with no daily limits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["gdelt", "commoncrawl", "all"],
                        default="all", help="Which data source(s) to ingest")
    parser.add_argument("--gdelt-start", default="20250101",
                        help="GDELT start date (YYYYMMDD)")
    parser.add_argument("--gdelt-end",   default="20251201",
                        help="GDELT end date (YYYYMMDD)")
    # Common Crawl arguments
    parser.add_argument("--cc-start", type=int, default=2022,
                        help="Common Crawl start year")
    parser.add_argument("--cc-end",   type=int, default=2022,
                        help="Common Crawl end year")
    # MongoDB connection arguments
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", DEFAULT_MONGO_URI),
                        help="MongoDB connection URI")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME,
                        help="Mongo database name")
    parser.add_argument("--coll-name", default=DEFAULT_COLL_NAME,
                        help="Mongo collection name")
    parser.add_argument("--fetch-delay", type=float, default=FETCH_DELAY_SECONDS,
                        help="DeYou successfully finished creating your Next.js lay in seconds between article fetch attempts.")

    args = parser.parse_args()

    MONGO_URI = args.mongo_uri
    DB_NAME = args.db_name
    COLL_NAME = args.coll_name
    FETCH_DELAY_SECONDS = args.fetch_delay

    logger.info("--- Historical News Ingestion Initializing ---")
    logger.info("Data Source(s): %s", args.source)
    logger.info("Mongo Target: %s / %s / %s", MONGO_URI, DB_NAME, COLL_NAME)
    if args.source in {"gdelt", "all"}:
        logger.info("GDELT Range: %s to %s", args.gdelt_start, args.gdelt_end)
    if args.source in {"commoncrawl", "all"}:
        logger.info("Common Crawl Range: %d to %d", args.cc_start, args.cc_end)
    logger.info("Fetch Delay: %.2f seconds", FETCH_DELAY_SECONDS)
    logger.info("Logging to file: %s (DEBUG level)", LOG_FILE)
    logger.info("Logging to console: INFO level")

    try:
        collection = get_mongo_collection()

        if args.source in {"gdelt", "all"}:
            ingest_gdelt(args.gdelt_start, args.gdelt_end, collection)
        else:
            logger.info("Skipping GDELT ingestion as per --source argument.")

        logger.info("--- Ingestion Process Finished ---")

    except PyMongoError as exc:
        logger.critical("Halting due to MongoDB error: %s", exc, exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("--- Ingestion Interrupted by User (Ctrl+C) ---")
        sys.exit(0)
    except Exception as exc:
        logger.critical(
            "An unexpected critical error occurred: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
