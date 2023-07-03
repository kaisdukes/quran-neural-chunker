import unittest

import pandas as pd


from dataset_splitter import split_dataset
from src.data import load_data
from src.chunks.chunks import get_chunks
from src.chunks.preprocessor import preprocess
from src.models.xgboost_chunker import XGBoostChunker
from src.models.evaluator import Evaluator


class ChunkTest(unittest.TestCase):

    def setUp(self):
        self.df = load_data()
        self.evaluator = Evaluator()
        preprocess(self.df)

    def test_ten_fold_cross_validation(self):
        for fold in range(10):
            self._train_and_test(fold)

    def _train_and_test(self, fold: int):

        print(f'Fold {fold}')
        train_df, test_df = split_dataset(self.df, fold)

        # train
        model = XGBoostChunker()
        model.build_mappings(self.df)
        model.train(train_df)

        # test
        predictions = model.predict(test_df)
        expected_chunks = get_chunks(test_df)
        output_chunks = get_chunks(predictions)
        self.evaluator.compare(expected_chunks, output_chunks)

        print(f'Running precision: {self.evaluator.precision}')
        print(f'Running recall: {self.evaluator.recall}')
        print(f'Running F1 score: {self.evaluator.f1_score}')


if __name__ == '__main__':
    unittest.main()
