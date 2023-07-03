import numpy as np
from pandas import DataFrame


def preprocess(df: DataFrame):
    df['verse_end'] = (
        (df.groupby(['chapter_number', 'verse_number']).token_number.transform(max) == df.token_number)
        .astype(int))

    df['punctuation'] = df['translation'].apply(_punctuation)


PUNCTUATION = [',', '.', '\'', '\"', '!', '?']


def _punctuation(text: str) -> str:
    n = len(text)
    for i in range(n - 1, -1, -1):
        if text[i] not in PUNCTUATION:
            return text[i+1:] if i < n - 1 else ''
    return text
