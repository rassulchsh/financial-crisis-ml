import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/clean_merged_data.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE = "merged_macro_data.csv"
OUTPUT_FILE = "merged_macro_data_clean.csv"

CRITICAL_COLUMNS = ["gdp_current_usd_wb",
                    "gdp_nominal_usd_imf", "inflation_annual_cpi_wb"]
KNOWN_STRING_COLUMNS = ["iso"]

# ----------------------------------------------------------------------------
# Data Loading Function
# ----------------------------------------------------------------------------


def load_data(filepath: str) -> pd.DataFrame:
    logger.info("Loading merged dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded data shape: %s", df.shape)
    return df

# ----------------------------------------------------------------------------
# Data Cleaning Functions
# ----------------------------------------------------------------------------


def standardize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing country codes...")
    if "iso" in df.columns:
        df["iso"] = df["iso"].astype(str).str.upper()
    else:
        logger.warning("Column 'iso' not found in the dataset.")
    logger.info("Country codes standardized.")
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting eligible columns to numeric...")
    for col in df.columns:
        if col not in KNOWN_STRING_COLUMNS and col.lower() != "year":
            try:
                # Remove possible commas and convert to numeric
                df[col] = pd.to_numeric(df[col].replace(
                    {",": ""}, regex=True), errors="coerce")
                logger.debug("Converted column '%s' to numeric.", col)
            except Exception as ex:
                logger.warning("Could not convert column '%s': %s", col, ex)
    logger.info("Numeric conversion complete.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing values...")
    initial_count = df.shape[0]

    existing_critical = [col for col in CRITICAL_COLUMNS if col in df.columns]
    if existing_critical:
        df = df.dropna(subset=existing_critical)
        dropped = initial_count - df.shape[0]
        logger.info("Dropped %d rows due to missing values in critical columns %s; remaining rows: %d",
                    dropped, existing_critical, df.shape[0])
    else:
        logger.warning("None of the specified critical columns %s found in the dataset. Skipping row drop step.",
                       CRITICAL_COLUMNS)

    if "year" not in df.columns:
        logger.warning(
            "Column 'year' not found. Skipping temporal imputation.")
    else:
        df = df.sort_values(by=["iso", "year"])
        for col in df.columns:
            if col not in KNOWN_STRING_COLUMNS and col.lower() != "year":
                df[col] = df.groupby("iso")[col].fillna(
                    method="ffill").fillna(method="bfill")
        logger.info(
            "Missing values imputed (ffill then bfill within each country).")

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping duplicate rows (if any)...")
    initial_count = df.shape[0]
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows; final row count: %d",
                initial_count - df.shape[0], df.shape[0])
    return df

# ----------------------------------------------------------------------------
# Preprocessing Pipeline
# ----------------------------------------------------------------------------


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing of merged macro data...")
    df = standardize_country_codes(df)
    df = convert_numeric_columns(df)
    df = handle_missing_values(df)
    df = drop_duplicates(df)
    logger.info("Preprocessing complete. Final data shape: %s", df.shape)
    return df

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------


def main():
    try:
        df = load_data(INPUT_FILE)
        df_clean = preprocess_data(df)
        logger.info("Saving cleaned data to %s", OUTPUT_FILE)
        df_clean.to_csv(OUTPUT_FILE, index=False)
        logger.info("Cleaned data saved successfully.")
    except Exception as e:
        logger.exception("Preprocessing and cleaning failed!")
        raise


if __name__ == "__main__":
    main()
