#!/usr/bin/env python
import logging
from logging.handlers import RotatingFileHandler
import os
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Logging Setup
# ----------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/transform_world_bank.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_format)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------
# 2. Configuration
# ----------------------------------------------------------------------------
INPUT_CSV = "world_bank_data_final.csv"      # Raw data file from previous step
# Output file with clean & engineered data
OUTPUT_CSV = "wb_clean_transformed.csv"

# Critical columns: rows missing these values will be dropped.
CRITICAL_COLUMNS = ['gdp_current_usd', 'inflation_annual_cpi']

# List of numeric columns that we expect to convert
NUMERIC_COLUMNS = [
    'gdp_current_usd', 'gdp_growth_annual', 'inflation_annual_cpi',
    'credit_private_gdp', 'broad_money_gdp', 'current_account_gdp',
    'trade_percent_gdp', 'exchange_rate_usd', 'unemployment_rate',
    'gov_debt_percent_gdp', 'gov_revenue_percent_gdp'
]

# ----------------------------------------------------------------------------
# 3. Data Cleaning Functions
# ----------------------------------------------------------------------------


def drop_critical_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in critical columns."""
    logger.info("Dropping rows with critical NaNs: " +
                ", ".join(CRITICAL_COLUMNS))
    initial_count = df.shape[0]
    df_clean = df.dropna(subset=CRITICAL_COLUMNS)
    final_count = df_clean.shape[0]
    logger.info(
        f"Dropped {initial_count - final_count} rows; remaining rows: {final_count}")
    return df_clean


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that expected numeric columns are properly converted to numbers."""
    logger.info("Converting numeric columns to proper numeric types...")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    logger.info("Numeric conversion complete.")
    return df


def standardize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure country codes are in consistent ISO3 format (upper case)."""
    logger.info("Standardizing country codes to ISO3...")
    if 'iso' in df.columns:
        df['iso'] = df['iso'].astype(str).str.upper()
    logger.info("Country codes standardized.")
    return df

# ----------------------------------------------------------------------------
# 4. Feature Engineering Functions
# ----------------------------------------------------------------------------


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag-based features:
      - gdp_growth_pctDiff5: Percentage difference in GDP growth compared to 5 years prior.
      - debt_gdp_ratioDiff5: Percentage difference in government debt-to-GDP ratio compared to 5 years prior.
    """
    logger.info("Creating lag-based features...")
    # Sort by country and year (ensure year is numeric)
    df = df.sort_values(by=['iso', 'year'])

    # Calculate features for each country group
    df['gdp_growth_pctDiff5'] = df.groupby('iso')['gdp_growth_annual'].apply(
        lambda group: (group - group.shift(5)) / group.shift(5) * 100
    )
    df['debt_gdp_ratioDiff5'] = df.groupby('iso')['gov_debt_percent_gdp'].apply(
        lambda group: (group - group.shift(5)) / group.shift(5) * 100
    )
    logger.info("Lag-based features created.")
    return df


def create_gdp_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create GDP ratio features:
      - credit_gdp_ratio: Ratio of credit (credit_private_gdp) to GDP.
      - broad_money_gdp_ratio: Ratio of broad money to GDP.
    The ratios are expressed as percentages.
    """
    logger.info("Creating GDP ratio features...")
    if 'credit_private_gdp' in df.columns and 'gdp_current_usd' in df.columns:
        df['credit_gdp_ratio'] = df['credit_private_gdp'] / \
            df['gdp_current_usd'] * 100
    if 'broad_money_gdp' in df.columns and 'gdp_current_usd' in df.columns:
        df['broad_money_gdp_ratio'] = df['broad_money_gdp'] / \
            df['gdp_current_usd'] * 100
    logger.info("GDP ratio features created.")
    return df

# ----------------------------------------------------------------------------
# 5. Main Transformation Process
# ----------------------------------------------------------------------------


def main():
    try:
        logger.info("Starting World Bank data transformation process...")

        # Load raw data
        logger.info(f"Loading raw data from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Raw data shape: {df.shape}")

        # Data Cleaning
        df = drop_critical_nans(df)
        df = convert_numeric_columns(df)
        df = standardize_country_codes(df)

        # Feature Engineering
        df = create_lag_features(df)
        df = create_gdp_ratios(df)

        # Save the clean & transformed data
        logger.info(f"Saving transformed data to {OUTPUT_CSV}")
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(
            f"Data transformation complete. Final data shape: {df.shape}")

    except Exception as e:
        logger.exception("Data transformation failed!")
        raise


# ----------------------------------------------------------------------------
# 6. Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
