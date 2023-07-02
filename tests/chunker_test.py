import unittest


from evaluator import Evaluator
from src.data import load_data
from src.chunks.chunks import get_chunks
from src.chunks.preprocessor import preprocess
from src.models.simple_chunker import SimpleChunker


class ChunkTest(unittest.TestCase):

    def test_simple_chunker(self):
        df = load_data()
        preprocess(df)

        model = SimpleChunker()
        predictions = model.predict(df)

        expected_chunks = get_chunks(df)
        output_chunks = get_chunks(predictions)

        print(len(expected_chunks))
        print(len(output_chunks))

        evaluator = Evaluator()
        evaluator.compare(expected_chunks, output_chunks)

        print(f'Precision: {evaluator.precision}')
        print(f'Recall: {evaluator.recall}')
        print(f'F1 score: {evaluator.f1_score}')


if __name__ == '__main__':
    unittest.main()
