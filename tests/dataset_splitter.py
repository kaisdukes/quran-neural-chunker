from pandas import DataFrame
from sklearn.model_selection import GroupShuffleSplit


def split_dataset(df: DataFrame, fold: int):

    df['verse_id'] = df['chapter_number'].astype(str) + ':' + df['verse_number'].astype(str)

    train_idx, test_idx = next(
        GroupShuffleSplit(test_size=.10, n_splits=2, random_state=fold)
        .split(df, groups=df['verse_id'])
    )

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_df = train_df.drop(columns=['verse_id'])
    test_df = test_df.drop(columns=['verse_id'])

    return train_df, test_df
