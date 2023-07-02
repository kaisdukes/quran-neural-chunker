from pandas import DataFrame


class SimpleChunker:
    def __init__(self):
        pass

    def predict(self, df: DataFrame):
        predictions = df.copy()

        predictions['chunk_end'] = df.apply(
            lambda row: int(row['verse_end'] or row['pause_mark'] != 0),
            axis=1)

        return predictions
