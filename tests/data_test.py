import unittest

from src.chunks import load_chunks


class DataTest(unittest.TestCase):

    def test_chunks(self):
        df = load_chunks()

        expected_columns = [
            'chapter_number',
            'verse_number',
            'token_number',
            'arabic',
            'pos_tag',
            'translation',
            'pause_mark',
            'irab_end',
            'chunk_end'
        ]

        self.assertListEqual(list(df.columns), expected_columns)
        self.assertFalse(df.isnull().values.any())
        self.assertEqual(len(df), 32617)


if __name__ == '__main__':
    unittest.main()
