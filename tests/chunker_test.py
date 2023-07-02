import unittest

import pandas as pd


from dataset_splitter import split_dataset
from evaluator import Evaluator
from src.data import load_data
from src.chunks.chunks import get_chunks
from src.chunks.preprocessor import preprocess
from src.models.cart_chunker import CartChunker


class ChunkTest(unittest.TestCase):

    def setUp(self):
        self.df = load_data()
        preprocess(self.df)
        self.train_df, self.test_df = split_dataset(self.df)

    def test_cart_chunker(self):

        model = CartChunker()
        model.build_mappings(self.df)
        model.train(self.train_df)

        predictions = model.predict(self.test_df)
        expected_chunks = get_chunks(self.test_df)
        output_chunks = get_chunks(predictions)

        print(f'Expected: {len(expected_chunks)} chunks')
        print(f'Output: {len(output_chunks)} chunks')

        evaluator = Evaluator()
        evaluator.compare(expected_chunks, output_chunks)

        print(f'Precision: {evaluator.precision}')
        print(f'Recall: {evaluator.recall}')
        print(f'F1 score: {evaluator.f1_score}')


if __name__ == '__main__':
    unittest.main()
