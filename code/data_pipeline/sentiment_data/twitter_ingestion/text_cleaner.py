# text_cleaner.py
import re
from twitter_loader import _extract_tickers


def clean_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^\w\s\.,?!']", "", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tickers = _extract_tickers(text)
    return text
