from pandas import DataFrame


def preprocess(df: DataFrame):
    df['verse_end'] = (
        (df.groupby(['chapter_number', 'verse_number']).token_number.transform(max) == df.token_number)
        .astype(int))
