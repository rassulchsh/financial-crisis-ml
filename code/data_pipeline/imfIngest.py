import logging
import os
import requests
import pandas as pd

LOG_FILE = "logs/imf_ingest.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)


IMF_WEO_URL = "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2024/April/WEOApr2024all.ashx"
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

RAW_COUNTRY_CODE_COL = "WEO Country Code"
RAW_SUBJECT_CODE_COL = "WEO Subject Code"
RAW_SUBJECT_NAME_COL = "Subject Descriptor"


def fetch_imf_data(url: str, local_filename: str) -> None:
    if os.path.exists(local_filename):
        logger.info(
            f"File {local_filename} already exists. Skipping download.")
        return

    logger.info(f"Downloading IMF WEO file from {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded file saved as {local_filename}.")
    except Exception as e:
        logger.exception("Failed to download IMF WEO data.")
        raise


def parse_weo_file(local_filename: str) -> pd.DataFrame:
    ext = os.path.splitext(local_filename)[1].lower()
    df_raw = None
    if ext == ".xlsx":
        try:
            with open(local_filename, "rb") as f:
                header = f.read(2)
            if header != b"PK":
                logger.error(
                    "Downloaded file is not a valid XLSX file (missing PK signature).")
                raise ValueError(
                    "Invalid XLSX file: not a ZIP archive. Check URL output.")
        except Exception as e:
            logger.exception("Error reading XLSX file header.")
            raise
        engine = "openpyxl"
        df_raw = pd.read_excel(local_filename, engine=engine)
    elif ext == ".xls":
        engine = "xlrd"
        try:
            df_raw = pd.read_excel(local_filename, engine=engine)
            logger.info("File read successfully using xlrd.")
        except Exception as e:
            logger.warning(
                "Failed to read file as Excel using xlrd. Attempting to read as tab-delimited text.")
            try:
                df_raw = pd.read_csv(
                    local_filename, sep="\t", encoding="utf-16")
            except UnicodeError as ue:
                logger.warning(
                    "UTF-16 read failed, trying with 'utf-16-le' encoding.")
                try:
                    df_raw = pd.read_csv(
                        local_filename, sep="\t", encoding="utf-16-le")
                    logger.info(
                        "File read successfully using 'utf-16-le' encoding.")
                except Exception as e2:
                    logger.exception(
                        "Failed to read file as tab-delimited text with utf-16-le encoding.")
                    raise
            except Exception as e:
                logger.exception("Failed to read file as tab-delimited text.")
                raise
    else:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.info(f"Raw data shape: {df_raw.shape}")
    print(df_raw.columns.tolist())

    if RAW_COUNTRY_CODE_COL in df_raw.columns:
        df_raw.rename(columns={RAW_COUNTRY_CODE_COL: "iso"}, inplace=True)
    else:
        logger.error(
            f"Expected country code column '{RAW_COUNTRY_CODE_COL}' not found.")
        raise ValueError("Country code column missing.")

    year_cols = [col for col in df_raw.columns if str(col).strip().isdigit()]
    if not year_cols:
        logger.error("No year columns identified in the dataset.")
        raise ValueError("Year columns missing.")

    df_long = df_raw.melt(
        id_vars=["iso", RAW_SUBJECT_CODE_COL, RAW_SUBJECT_NAME_COL],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    df_long["year"] = df_long["year"].astype(int)
    logger.info(f"Data after melting: {df_long.shape}")
    return df_long


def process_weo_data(df_long: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filtering rows for selected IMF indicators...")
    valid_codes = set(IMF_INDICATORS.keys())
    mask = df_long[RAW_SUBJECT_CODE_COL].isin(valid_codes)
    df_filtered = df_long.loc[mask].copy()
    logger.info(f"Filtered data shape (by indicators): {df_filtered.shape}")

    df_filtered["indicator"] = df_filtered[RAW_SUBJECT_CODE_COL].map(
        IMF_INDICATORS)
    if df_filtered["indicator"].isnull().any():
        logger.warning(
            "Some rows did not map to a user-friendly indicator name.")

    logger.info("Pivoting data to wide format (one row per country-year)...")
    df_pivot = df_filtered.pivot_table(
        index=["iso", "year"],
        columns="indicator",
        values="value",
        aggfunc="first"
    ).reset_index()
    df_pivot.columns.name = None
    df_pivot["iso"] = df_pivot["iso"].astype(str).str.upper()

    logger.info("Filtering data for target countries...")
    df_final = df_pivot[df_pivot["iso"].isin(JST_COUNTRIES)].copy()
    logger.info(
        f"Data shape after filtering for target countries: {df_final.shape}")
    return df_final


def main():
    try:
        logger.info(
            "Starting IMF WEO ingestion process for crisis forecasting data...")
        fetch_imf_data(IMF_WEO_URL, RAW_FILE)
        df_long = parse_weo_file(RAW_FILE)
        df_processed = process_weo_data(df_long)
        df_processed.to_csv(OUTPUT_FILE, index=False)
        logger.info(
            f"IMF WEO data ingestion complete. Saved file: {OUTPUT_FILE}")
    except Exception as e:
        logger.exception("IMF WEO ingestion process failed!")
        raise


if __name__ == "__main__":
    main()
