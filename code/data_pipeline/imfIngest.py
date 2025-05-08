import logging
import os
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Logging Setup
# ----------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/imf_ingest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 2. Configuration
# ----------------------------------------------------------------------------
RAW_FILE = "WEOApr2024all.xls"
OUTPUT_FILE = "imf_weo_data.csv"

JST_COUNTRIES = ["DEU", "USA", "BRA", "IND",
                 "GBR", "FRA", "ITA", "JPN", "CAN", "AUS"]

IMF_INDICATORS = {
    "NGDPD": "gdp_nominal_usd",
    "NGDP_RPCH": "gdp_growth_real",
    "NGDPDPC": "gdp_per_capita",
    "PCPIPCH": "inflation_yoy",
    "PCPIEPCH": "inflation_eop",
    "GGXONLB_NGDP": "fiscal_deficit_pct_gdp",
    "GGXWDG_NGDP": "gov_debt_pct_gdp",
    "GGXGRT_NGDP": "gov_revenue_pct_gdp",
    "BCA_NGDPD": "current_account_pct_gdp",
    "NGDP_FOB_USD": "exports_usd",
    "UNEMP": "unemployment_rate",
    "LP": "unemployment_rate"
}

RAW_SUBJECT_CODE_COL = "WEO Subject Code"
RAW_SUBJECT_NAME_COL = "Subject Descriptor"

# ----------------------------------------------------------------------------
# 3. Parsing Function
# ----------------------------------------------------------------------------


def parse_weo_file(local_filename: str) -> pd.DataFrame:
    logger.info("Attempting to read WEO file as UTF-16 tab-delimited text...")
    try:
        df_raw = pd.read_csv(local_filename, sep="\t", encoding="utf-16")
        logger.info("File read successfully with utf-16 encoding.")
    except UnicodeError as e:
        logger.warning(
            "Failed to read with utf-16 encoding due to: %s. Trying with utf-16-le encoding...", e)
        df_raw = pd.read_csv(local_filename, sep="\t", encoding="utf-16-le")
        logger.info("File read successfully with utf-16-le encoding.")
    logger.info(f"File shape: {df_raw.shape}")

    if "ISO" not in df_raw.columns:
        raise ValueError("Column 'ISO' not found in IMF file.")

    df_raw["iso"] = df_raw["ISO"].astype(str).str.upper()

    # Identify year columns dynamically.
    year_cols = [col for col in df_raw.columns if col.isdigit()]
    if not year_cols:
        raise ValueError("No year columns found in IMF file.")

    # Melt to long format.
    df_long = df_raw.melt(
        id_vars=["iso", RAW_SUBJECT_CODE_COL, RAW_SUBJECT_NAME_COL],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    df_long["year"] = df_long["year"].astype(int)

    # Convert values to numeric, coercing errors (e.g., '--') to NaN.
    df_long["value"] = pd.to_numeric(
        df_long["value"].astype(str).str.replace(",", ""), errors="coerce"
    )

    logger.info(f"Data melted to long format. Shape: {df_long.shape}")
    return df_long

# ----------------------------------------------------------------------------
# 4. Processing Function
# ----------------------------------------------------------------------------


def process_weo_data(df_long: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filtering for selected IMF indicators...")
    df_filtered = df_long[df_long[RAW_SUBJECT_CODE_COL].isin(
        IMF_INDICATORS.keys())].copy()
    df_filtered["indicator"] = df_filtered[RAW_SUBJECT_CODE_COL].map(
        IMF_INDICATORS)

    # Pivot to wide format.
    logger.info("Pivoting data by country-year...")
    df_pivot = df_filtered.pivot_table(
        index=["iso", "year"],
        columns="indicator",
        values="value",
        aggfunc="first"
    ).reset_index()
    df_pivot.columns.name = None

    # Filter target countries.
    df_final = df_pivot[df_pivot["iso"].isin(JST_COUNTRIES)].copy()
    logger.info(f"Final processed data shape: {df_final.shape}")
    return df_final

# ----------------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------------


def main():
    try:
        logger.info("=== IMF WEO Data Ingestion Started ===")
        df_long = parse_weo_file(RAW_FILE)
        df_processed = process_weo_data(df_long)
        df_processed.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Data saved to {OUTPUT_FILE}")
        logger.info("=== IMF WEO Ingestion Completed ===")
    except Exception as e:
        logger.exception("IMF ingestion failed.")
        raise


if __name__ == "__main__":
    main()
