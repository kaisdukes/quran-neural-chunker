from pandas import DataFrame
from sklearn import preprocessing
from xgboost import XGBClassifier


class XGBoostChunker:

    def __init__(self):
        self._punctuation_encoder = preprocessing.LabelEncoder()

    def build_mappings(self, entire_dataset: DataFrame):
        self._punctuation_encoder.fit(entire_dataset['punctuation'])

    def train(self, df: DataFrame):
        x, y = self._preprocess(df)
        self.clf = XGBClassifier()
        self.clf.fit(x, y)

    def predict(self, df: DataFrame):
        x = self._preprocess(df)[0]
        out = df.copy()
        out['chunk_end'] = self.clf.predict(x)
        return out

    def _preprocess(self, df: DataFrame):
        x = df[['token_number', 'pause_mark', 'irab_end', 'verse_end', 'punctuation']].copy()
        x['punctuation'] = self._punctuation_encoder.transform(x['punctuation'])

        y = df['chunk_end']
        return x, y
