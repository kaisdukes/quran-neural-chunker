import unittest


from evaluate import evaluate
from src.data import load_data
from src.chunks.chunks import get_chunks
from src.chunks.preprocessor import preprocess
from src.models.simple_chunker import SimpleChunker


class ModelTest(unittest.TestCase):

    def test_simple_verse_chunker(self):
        df = load_data()
        preprocess(df)

        model = SimpleChunker()
        predictions = model.predict(df)

        expected_chunks = get_chunks(df)
        output_chunks = get_chunks(predictions)
        evaluate(expected_chunks, output_chunks)


if __name__ == '__main__':
    unittest.main()
