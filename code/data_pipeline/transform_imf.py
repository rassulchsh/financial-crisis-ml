import logging
from logging.handlers import RotatingFileHandler
import os
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Logging Setup
# ----------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/transform_imf.log"

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
INPUT_CSV = "imf_weo_data.csv"
OUTPUT_CSV = "imf_clean_transformed.csv"

CRITICAL_COLUMNS = ['gdp_nominal_usd', 'gdp_growth_real']
NUMERIC_COLUMNS = [
    'gdp_nominal_usd', 'gdp_growth_real', 'gdp_per_capita',
    'inflation_yoy', 'inflation_eop', 'fiscal_deficit_pct_gdp',
    'gov_debt_pct_gdp', 'gov_revenue_pct_gdp',
    'current_account_pct_gdp', 'exports_usd', 'unemployment_rate'
]

# ----------------------------------------------------------------------------
# 3. Data Cleaning
# ----------------------------------------------------------------------------


def drop_critical_nans(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        f"Dropping rows with NaNs in critical columns: {CRITICAL_COLUMNS}")
    initial = df.shape[0]
    df = df.dropna(subset=CRITICAL_COLUMNS)
    logger.info(
        f"Dropped {initial - df.shape[0]} rows; Remaining: {df.shape[0]}")
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting columns to numeric...")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def standardize_iso_codes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing country codes (ISO)...")
    if 'iso' in df.columns:
        df['iso'] = df['iso'].astype(str).str.upper()
    return df

# ----------------------------------------------------------------------------
# 4. Feature Engineering
# ----------------------------------------------------------------------------


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating lag-based features...")
    df = df.sort_values(by=['iso', 'year'])
    df['gdp_growth_real_diff5'] = df.groupby('iso')['gdp_growth_real'].transform(
        lambda x: (x - x.shift(5)) / x.shift(5) * 100
    )
    df['gov_debt_change_5y'] = df.groupby('iso')['gov_debt_pct_gdp'].transform(
        lambda x: (x - x.shift(5)) / x.shift(5) * 100
    )
    return df


def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating ratio-based indicators...")
    if 'gov_revenue_pct_gdp' in df.columns and 'fiscal_deficit_pct_gdp' in df.columns:
        df['net_gov_balance_pct_gdp'] = df['gov_revenue_pct_gdp'] - \
            df['fiscal_deficit_pct_gdp']
    return df

# ----------------------------------------------------------------------------
# 5. Main Transformation Pipeline
# ----------------------------------------------------------------------------


def main():
    try:
        logger.info("Starting IMF data transformation...")

        # Load raw data
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {INPUT_CSV} with shape: {df.shape}")

        # Cleaning
        df = drop_critical_nans(df)
        df = convert_numeric_columns(df)
        df = standardize_iso_codes(df)

        # Feature Engineering
        df = create_lag_features(df)
        df = create_ratios(df)

        # Save clean output
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(
            f"Transformed data saved to: {OUTPUT_CSV} | Final shape: {df.shape}")

    except Exception as e:
        logger.exception("IMF data transformation failed!")
        raise


# ----------------------------------------------------------------------------
# 6. Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
