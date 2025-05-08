# code/generate_explanations.py
import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Any, Tuple, Dict

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, BulkWriteError
from tqdm import tqdm
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import HarmCategory, HarmBlockThreshold

try:
    from code.models.lstm_arch import build_lstm
    from code.prep_helpers import reshape_for_lstm
    if not callable(build_lstm) or not callable(reshape_for_lstm):
        raise ImportError("Required functions not callable.")
    print("Successfully imported build_lstm and reshape_for_lstm (for potential unpickling).")
except ImportError as e:
    print(
        f"WARNING: Could not import model/prep functions: {e}. Ensure files exist if needed elsewhere.")

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s")
log = logging.getLogger("ExplanationGenerator")
load_dotenv()  # Load .env file

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", message=".*is fitted with feature names, but X does not.*")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

# --- Constants ---
KEYS = ["iso", "year"]
MODEL_DIR = Path("code/fitted_models")
OUTPUT_DB = os.getenv("MONGO_DB_NAME", "financial_crisis")
OUTPUT_COLLECTION = "eri_explanations_nosha"
LLM_PROVIDER = 'google'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
MONGO_BATCH_SIZE = 500

KEY_FEATURES_FOR_PROMPT = [
    'gdpgrowth_wb',
    'inflation_annual_cpi_wb',
    'total_debt_to_gdp_weighted_average',
    'gfdd_sm_01',
    'news_sentiment',
    'systloan_gdp_jst',
    'net_foreign_assets_imf'
]

# --- Helper Functions ---


def load_all_data(eri_path: Path, panel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    log.info(f"Loading ERI scores from: {eri_path}")
    if not eri_path.is_file():
        raise FileNotFoundError(f"ERI file not found: {eri_path}")
    eri_df = pd.read_csv(eri_path)
    log.info(f"Loaded {len(eri_df)} ERI scores.")

    log.info(f"Loading full panel data from: {panel_path}")
    if not panel_path.is_file():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    panel_df = pd.read_csv(panel_path, low_memory=False)
    log.info(f"Loaded panel data shape: {panel_df.shape}")

    # --- Type Conversions & Validation ---
    try:
        eri_df['year'] = eri_df['year'].astype(int)
        panel_df['year'] = pd.to_numeric(
            panel_df['year'], errors='coerce').astype('Int64')
        eri_df['iso'] = eri_df['iso'].astype(str).str.upper()
        panel_df['iso'] = panel_df['iso'].astype(str).str.upper()
        assert {'iso', 'year'}.issubset(
            eri_df.columns), "ERI file missing key columns."
        assert {'iso', 'year'}.issubset(
            panel_df.columns), "Panel file missing key columns."
    except KeyError as e:
        raise ValueError(f"Missing key column: {e}")

    # --- Identify Numeric Training Columns ---
    all_numeric_cols = []
    log.info("Extracting full feature list from saved preprocessor...")
    try:
        xgb_pipeline_path = MODEL_DIR / "XGBoost_pipeline.joblib"
        assert xgb_pipeline_path.is_file(
        ), f"XGB model not found: {xgb_pipeline_path}"
        xgb_pipeline = joblib.load(xgb_pipeline_path)
        preprocessor_step = xgb_pipeline.named_steps['prep']
        all_numeric_cols = preprocessor_step.transformers_[0][2]
        assert isinstance(all_numeric_cols, list), "Expected list."
        log.info(
            f"Extracted {len(all_numeric_cols)} numeric feature names model was trained on.")
    except Exception as e:
        log.error(f"CRITICAL ERROR extracting features: {e}", exc_info=True)
        raise ValueError("Feature extraction failed.") from e
    assert all_numeric_cols, "Numeric columns list empty."

    # --- Filter Panel Data ---
    relevant_cols = KEYS + all_numeric_cols
    missing = [c for c in relevant_cols if c not in panel_df.columns]
    assert not missing, f"Panel missing required columns: {missing}"
    panel_features_df = panel_df[relevant_cols].copy().dropna(subset=['year'])
    panel_features_df['year'] = panel_features_df['year'].astype(int)

    log.info(
        f"Filtered panel data: {panel_features_df.shape[1]} cols, {len(panel_features_df)} rows.")
    return eri_df, panel_features_df, all_numeric_cols


def generate_explanation_from_data(llm_client, country: str, year: int, eri_score: float,
                                   feature_data: Dict[str, Any]) -> str:
    safety_settings = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
    generation_config = genai.types.GenerationConfig(
        temperature=0.7)

    prompt = f"You are an economic analyst providing concise explanations for a non-expert audience.\n\n"
    prompt += f"Analyze the Economic Risk Index (ERI) for {country} in {year}.\n\n"
    prompt += f"Context:\n"
    prompt += f"- The calculated ERI Score is {eri_score:.3f} (Scale 0-1, higher score indicates higher economic risk).\n"
    prompt += f"- Consider the following economic indicators for {country} in {year}:\n"
    key_features_found = False
    for feature_name in KEY_FEATURES_FOR_PROMPT:
        value = feature_data.get(feature_name)
        if pd.notna(value):
            prompt += f"  - {feature_name}: {value:.2f}\n"
            key_features_found = True
        else:
            prompt += f"  - {feature_name}: (Data unavailable)\n"
    if not key_features_found:
        prompt += "\n  (Note: Key indicator data is largely unavailable for this period.)\n"

    prompt += f"""
Task:
Based *only* on the provided ERI score and the economic indicators listed above, write a concise possible explanation (strictly 2-3 sentences) for why {country} might have received this risk score in {year}. Speculate on how the available indicators (like GDP growth, inflation, debt, sentiment, etc.) could contribute to the assessed risk level (positively or negatively). Acknowledge if key data is unavailable. Do *not* mention specific models or analysis techniques.
"""

    max_retries = 2
    retry_delay = 3
    for attempt in range(max_retries + 1):
        try:
            response = llm_client.generate_content(
                prompt, safety_settings=safety_settings, generation_config=generation_config)
            if not response.candidates:
                block_reason = "Unknown"
                try:
                    block_reason = response.prompt_feedback.block_reason or "No candidates"
                except Exception:
                    pass
                log.warning(
                    f"LLM response blocked {country}/{year}: {block_reason}")
                return f"Error: LLM response blocked ({block_reason})."
            explanation = response.text
            return explanation.strip()
        # Catch specific rate limit error
        except ResourceExhausted as rate_limit_e:
            log.error(
                f"RATE LIMIT HIT or quota exceeded for {country}/{year} on attempt {attempt+1}: {rate_limit_e}.")
            raise rate_limit_e  # Reraise to be caught by main loop
        except Exception as e:
            log.warning(
                f"LLM API call attempt {attempt+1} fail {country}/{year}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                # Return error string
                return f"Error: Failed explanation generation ({e})."
    return "Error: Explanation generation failed."  # Fallback


def store_in_mongo(mongo_client, db_name, coll_name, data_list):
    """Stores the list of results in MongoDB using bulk operations."""
    if not data_list:
        log.warning("No data to store.")
        return 0
    try:
        db = mongo_client[db_name]
        collection = db[coll_name]
        log.info(f"Storing {len(data_list)} in {db_name}.{coll_name}")
        try:
            collection.create_index(
                [("iso", 1), ("year", -1)], unique=True, background=True)
        except Exception as idx_e:
            log.warning(f"Index creation warn: {idx_e}")

        from pymongo import UpdateOne
        bulk_ops = []
        for r in data_list:
            filter_q = {"iso": r["iso"], "year": r["year"]}
            # Store only relevant fields for this version
            update_q = {"$set": {"eriScore": r["eriScore"], "explanation": r["explanation"], "updatedAt": pd.Timestamp.utcnow()},
                        "$setOnInsert": {"iso": r["iso"], "year": r["year"], "createdAt": pd.Timestamp.utcnow()}}
            bulk_ops.append(UpdateOne(filter_q, update_q, upsert=True))

        if bulk_ops:
            try:
                result = collection.bulk_write(bulk_ops, ordered=False)
                inserted = result.upserted_count
                matched = result.matched_count
                log.info(
                    f"Mongo bulk: Inserted={inserted}, Matched/Updated={matched}")
                return inserted + matched
            except BulkWriteError as bwe:
                failed = len(bwe.details.get('writeErrors', []))
                log.error(
                    f"MongoDB bulk write error ({failed} failures): {bwe.details}", exc_info=False)
                successful_ops = bwe.details.get(
                    'nUpserted', 0) + bwe.details.get('nMatched', 0)
                return successful_ops  # Return successes even if some failed
            except Exception as bulk_e:
                log.error(f"Mongo bulk write failed: {bulk_e}", exc_info=True)
                return 0
        else:
            log.info("No Mongo ops needed.")
            return 0
    except Exception as e:
        log.error(f"Mongo connection/setup error: {e}", exc_info=True)
        return 0


def get_already_processed(collection) -> set:
    """Fetches (iso, year) tuples for records already in MongoDB."""
    log.info(
        f"Querying MongoDB for already processed records in {collection.name}...")
    processed = set()
    try:
        cursor = collection.find({}, {"iso": 1, "year": 1, "_id": 0})
        for doc in cursor:
            if 'iso' in doc and 'year' in doc:
                processed.add((doc['iso'], doc['year']))
        log.info(f"Found {len(processed)} records already processed.")
    except Exception as e:
        log.error(
            f"Failed to query MongoDB for processed records: {e}", exc_info=True)
        log.warning(
            "Proceeding without resuming capability due to DB query error.")
    return processed

# --- Main Execution ---


def main(eri_file: Path, panel_file: Path, limit: int | None = None):
    """Main orchestrator function with resume capability."""
    log.info(
        "--- Starting ERI Explanation Generation Pipeline (No SHAP, Resumable) ---")
    start_time = time.time()
    processed_this_run = 0
    error_count = 0
    skipped_count = 0
    results_batch = []

    mongo_client = None
    processed_set = set()

    try:
        # 1. Connect to MongoDB early for progress check
        log.info("Step 0: Connecting to MongoDB to check progress...")
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        try:
            mongo_client = MongoClient(
                mongo_uri, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command('ping')
            log.info("MongoDB connected.")
            db = mongo_client[OUTPUT_DB]
            collection = db[OUTPUT_COLLECTION]
            processed_set = get_already_processed(collection)
        except ConnectionFailure as conn_e:
            log.error(
                f"MongoDB connection failed: {conn_e}. Cannot check progress or save results.")
            sys.exit(1)
        except Exception as db_e:
            log.error(f"MongoDB setup error: {db_e}", exc_info=True)
            sys.exit(1)

        # 2. Load Data
        log.info("Step 1: Loading data...")
        eri_scores_df, panel_features_df, _ = load_all_data(
            eri_file, panel_file)
        log.info("Step 2: Skipping model loading.")

        # 3. Merge Data
        log.info("Step 3: Merging data...")
        merged_df = pd.merge(
            eri_scores_df, panel_features_df, on=KEYS, how="left")
        log.info(f"Merged data shape: {merged_df.shape}")

        total_records_to_consider = len(merged_df)
        iteration_df = merged_df
        if limit:
            log.warning(
                f"Limiting run to first {limit} records overall (including already processed).")
            iteration_df = merged_df.head(limit)
            total_records_to_consider = len(iteration_df)

        if iteration_df.empty:
            log.error("No data to process. Exiting.")
            return

        # 4. Initialize LLM Client
        log.info("Step 4: Initializing LLM...")
        api_key = os.getenv("GOOGLE_API_KEY")
        assert api_key, "GOOGLE_API_KEY missing."
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        log.info(f"Initialized LLM: {GEMINI_MODEL_NAME}.")

        # 5. Generate Explanations Iteratively
        log.info("Step 5: Generating LLM explanations (skipping processed)...")
        for index, row in tqdm(iteration_df.iterrows(), total=total_records_to_consider, desc="Generating Explanations"):
            iso, year, eri_score = row.get(
                'iso'), row.get('year'), row.get('eri')

            # --- Resume Check ---
            if (iso, int(year)) in processed_set:
                skipped_count += 1
                log.debug(f"Skipping {iso}/{year}: Already processed.")
                continue

            if pd.isna(eri_score):
                log.warning(
                    f"Skipping row {index} ({iso}/{year}): missing ERI score.")
                error_count += 1
                continue

            # --- Process Row ---
            try:
                feature_context = {feat: row.get(
                    feat) for feat in KEY_FEATURES_FOR_PROMPT if feat in row}
                explanation = generate_explanation_from_data(
                    llm, iso, year, eri_score, feature_context)

                if explanation.startswith("Error:"):
                    log.error(
                        f"LLM explanation failed for {iso}/{year}: {explanation}")
                    error_count += 1
                else:
                    results_batch.append({"iso": iso, "year": int(
                        year), "eriScore": float(eri_score), "explanation": explanation})
                    processed_this_run += 1

                # Store results in batches
                if len(results_batch) >= MONGO_BATCH_SIZE:
                    log.info(
                        f"Storing batch of {len(results_batch)} results...")
                    if store_in_mongo(mongo_client, OUTPUT_DB, OUTPUT_COLLECTION, results_batch) > 0:
                        for record in results_batch:
                            processed_set.add((record['iso'], record['year']))
                        results_batch = []  # Clear batch
                    else:
                        log.error(
                            "Failed to store batch to MongoDB. Exiting to prevent data loss.")
                        sys.exit(1)  # Exit if batch store fails critically

            # --- Specific Rate Limit Handling ---
            except ResourceExhausted as rate_limit_e:
                log.error(
                    f"RATE LIMIT HIT for {iso}/{year}: {rate_limit_e}. Saving progress and exiting.")
                error_count += 1
                # Attempt to save the current batch before exiting
                if results_batch:
                    log.info(
                        f"Attempting save of final {len(results_batch)} results before exit...")
                    store_in_mongo(mongo_client, OUTPUT_DB,
                                   OUTPUT_COLLECTION, results_batch)
                sys.exit(1)  # Exit gracefully
            except Exception as row_e:
                log.error(
                    f"Failed row {index} ({iso}/{year}): {row_e}", exc_info=False)
                error_count += 1

            # API Rate limiting
            time.sleep(1.1)

        log.info(f"Finished generation loop.")

        # 6. Store Final Batch
        if results_batch:
            log.info("Step 6: Storing final batch results...")
            store_in_mongo(mongo_client, OUTPUT_DB,
                           OUTPUT_COLLECTION, results_batch)
        else:
            log.info("Step 6: No final batch to store.")

    # Catch critical errors
    except FileNotFoundError as e:
        log.critical(f"File not found: {e}")
    except ValueError as e:
        log.critical(f"Data/config error: {e}")
    except KeyboardInterrupt:
        log.warning("--- Pipeline interrupted by user ---")
    except Exception as e:
        log.critical(f"Unexpected critical error: {e}", exc_info=True)
    finally:  # Log summary & close DB
        if mongo_client:
            mongo_client.close()
            log.info("MongoDB connection closed.")
        total_time = time.time() - start_time
        log.info(f"--- Explanation Generation Pipeline Finished ---")
        log.info(f"Total Time: {total_time:.2f}s")
        log.info(f"Records Processed This Run: {processed_this_run}")
        log.info(f"Records Skipped (Done Previously): {skipped_count}")
        log.info(f"Processing Errors This Run: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ERI explanations using LLMs (No SHAP, Resumable).")
    parser.add_argument("--eri-file", type=Path,
                        required=True, help="Path to ERI scores file")
    parser.add_argument("--panel-file", type=Path,
                        required=True, help="Path to the full panel data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N records overall")
    args = parser.parse_args()
    # Argument Validation
    if not args.eri_file.is_file():
        log.error(f"ERI file missing: {args.eri_file}")
        sys.exit(1)
    if not args.panel_file.is_file():
        log.error(f"Panel file missing: {args.panel_file}")
        sys.exit(1)

    main(args.eri_file, args.panel_file, args.limit)
