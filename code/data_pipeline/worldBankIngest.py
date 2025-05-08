import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict
import os
import pandas as pd
import requests
from datetime import datetime


def set_default_timeout(timeout: int = 10):
    original_request = requests.Session.request

    def request_with_timeout(self, *args, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return original_request(self, *args, **kwargs)

    requests.Session.request = request_with_timeout


set_default_timeout(10)
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/data_pipeline.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

WB_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_annual",
    "FP.CPI.TOTL.ZG": "inflation_annual_cpi",
    "FS.AST.PRVT.GD.ZS": "credit_private_gdp",
    "FM.LBL.BMNY.GD.ZS": "broad_money_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "NE.TRD.GNFS.ZS": "trade_percent_gdp",
    "PA.NUS.FCRF": "exchange_rate_usd",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "GC.DOD.TOTL.GD.ZS": "gov_debt_percent_gdp",
    "GC.REV.XGRT.GD.ZS": "gov_revenue_percent_gdp"
}

# Adjust the date range as needed.
WB_START_YEAR = 1960
WB_END_YEAR = 2023

# Using 3-letter ISO codes:
JST_COUNTRIES = [
    "DEU", "USA", "GBR", "FRA", "ITA", "JPN", "CAN", "AUS",
    "BRA", "IND", "IDN", "MEX", "TUR", "CHN", "RUS"
]

OUTPUT_CSV = "world_bank_data_final.csv"

# ----------------------------------------------------------------------------
# 3. Data Fetching Function (Using Direct API Calls and Indicator-wise Concatenation)
# ----------------------------------------------------------------------------


def fetch_world_bank_data(
    countries: List[str],
    indicators: Dict[str, str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Fetches indicator data from the World Bank API for each country and indicator.
    For each indicator, data from all countries is concatenated before merging
    the indicator DataFrames on 'iso' and 'year'.
    """
    logger.info(
        f"Fetching data for countries={countries}, years={start_year}-{end_year}")
    base_url = "https://api.worldbank.org/v2/country"

    # Dictionary to hold a concatenated DataFrame per indicator (using the friendly column name)
    indicator_dfs = {}

    for indicator, col_name in indicators.items():
        indicator_frames = []
        for country in countries:
            # Build the URL with HTTPS and required parameters
            url = (
                f"{base_url}/{country}/indicator/{indicator}"
                f"?format=json&date={start_year}:{end_year}&per_page=1000"
            )
            logger.info(f"Fetching URL: {url}")
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    logger.error(
                        f"Failed to fetch data for {indicator} in {country}: {response.status_code} - {response.text}")
                    continue

                json_data = response.json()
                if not json_data or len(json_data) < 2 or not isinstance(json_data[1], list):
                    logger.warning(
                        f"No data available for {indicator} in {country}")
                    continue

                df = pd.DataFrame(json_data[1])
                if df.empty:
                    logger.warning(
                        f"Empty DataFrame for {indicator} in {country}")
                    continue

                # Keep only necessary columns and rename
                df = df[['countryiso3code', 'date', 'value']]
                df.rename(columns={'countryiso3code': 'iso',
                          'date': 'year', 'value': col_name}, inplace=True)
                # Ensure the 'year' column is numeric for proper merging
                df['year'] = df['year'].astype(int)
                indicator_frames.append(df)
            except Exception as e:
                logger.error(
                    f"Exception while fetching data for {indicator} in {country}: {str(e)}", exc_info=True)

        if indicator_frames:
            # Concatenate all data for this indicator and remove duplicate rows for same iso-year if any
            indicator_df = pd.concat(indicator_frames, ignore_index=True)
            indicator_df = indicator_df.drop_duplicates(subset=['iso', 'year'])
            indicator_dfs[col_name] = indicator_df
        else:
            logger.warning(f"No data found for indicator {indicator}")

    if not indicator_dfs:
        logger.error("No data fetched for any indicator.")
        raise ValueError("Empty DataFrame from World Bank API.")

    # Merge all indicator DataFrames on 'iso' and 'year'
    merged_df = None
    for col_name, df in indicator_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=[
                                 'iso', 'year'], how='outer')

    logger.info(
        f"Successfully merged data. Final DataFrame shape: {merged_df.shape}")
    return merged_df

# ----------------------------------------------------------------------------
# 4. Preparation Function (Optional Additional Cleaning)
# ----------------------------------------------------------------------------


def prepare_world_bank_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform any additional cleaning if needed.
    """
    logger.info("Starting data preparation...")
    if df.empty:
        logger.warning("Received empty DataFrame for preparation.")
        return df

    # Additional cleaning can be performed here if necessary.
    logger.info(
        f"Data preparation complete. Final DataFrame shape: {df.shape}")
    return df

# ----------------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------------


def main() -> None:
    """Main data pipeline execution."""
    try:
        logger.info("=== Starting World Bank Data Pipeline ===")
        logger.info(f"Countries (ISO3): {JST_COUNTRIES}")
        logger.info(f"Date range: {WB_START_YEAR}-{WB_END_YEAR}")

        # 1) Fetch data from the API
        raw_df = fetch_world_bank_data(
            countries=JST_COUNTRIES,
            indicators=WB_INDICATORS,
            start_year=WB_START_YEAR,
            end_year=WB_END_YEAR
        )

        # 2) Clean and transform the data if necessary
        processed_df = prepare_world_bank_df(raw_df)

        # 3) Save the results to CSV
        logger.info(f"Saving data to {OUTPUT_CSV}")
        processed_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(
            f"Pipeline completed successfully. Saved {len(processed_df)} records to {OUTPUT_CSV}.")

    except Exception as e:
        logger.critical(
            "Pipeline failed with a critical error!", exc_info=True)
        raise


# ----------------------------------------------------------------------------
# 6. Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
