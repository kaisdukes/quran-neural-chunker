from pandas import DataFrame


class SimpleChunker:
    def __init__(self):
        pass

    def predict(self, df: DataFrame):
        predications = df.copy()
        predications['chunk_end'] = df['verse_end'].astype(int)
        return predications
