from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


class CartChunker:

    def __init__(self):
        self._punctuation_encoder = preprocessing.LabelEncoder()

    def build_mappings(self, df: DataFrame):
        self._punctuation_encoder.fit(df['punctuation'])

    def train(self, df: DataFrame):
        x, y = self._preprocess(df)
        self.clf = DecisionTreeClassifier(random_state=42)
        self.clf.fit(x, y)

    def predict(self, df: DataFrame):
        x = self._preprocess(df)[0]
        out = df.copy()
        out['chunk_end'] = self.clf.predict(x)
        return out

    def _preprocess(self, df: DataFrame):
        x = df.drop([
            'chunk_end',
            'chapter_number',
            'verse_number',
            'arabic',
            'pos_tag',
            'translation'
        ], axis=1)

        x['punctuation'] = self._punctuation_encoder.transform(x['punctuation'])

        y = df['chunk_end']
        return x, y
