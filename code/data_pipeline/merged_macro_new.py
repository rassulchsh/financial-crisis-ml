import os
import logging
import pandas as pd

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/merge_datasets.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)

IMF_FILE = "imf_clean_transformed.csv"
WB_FILE = "wb_clean_transformed.csv"
OUTPUT_FILE = "merged_macro_data.csv"



def load_datasets():
    logger.info("Loading IMF dataset from %s", IMF_FILE)
    df_imf = pd.read_csv(IMF_FILE)
    logger.info("IMF dataset shape: %s", df_imf.shape)

    logger.info("Loading World Bank dataset from %s", WB_FILE)
    df_wb = pd.read_csv(WB_FILE)
    logger.info("World Bank dataset shape: %s", df_wb.shape)

    return df_imf, df_wb



def merge_datasets(df_imf, df_wb):
    logger.info("Merging datasets on keys: ['iso', 'year'] (outer join)")
    merged_df = pd.merge(df_wb, df_imf, on=[
                         "iso", "year"], how="outer", suffixes=('_wb', '_imf'))
    logger.info("Merged dataset shape: %s", merged_df.shape)
    return merged_df

# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------


def main():
    try:
        logger.info("Starting merge of World Bank and IMF datasets...")

        # Load datasets
        df_imf, df_wb = load_datasets()

        # Merge datasets
        merged_df = merge_datasets(df_imf, df_wb)

        # Save merged dataset
        logger.info("Saving merged dataset to %s", OUTPUT_FILE)
        merged_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(
            "Merged dataset saved successfully. Final shape: %s", merged_df.shape)

    except Exception as e:
        logger.exception("Merging process failed!")
        raise


# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
