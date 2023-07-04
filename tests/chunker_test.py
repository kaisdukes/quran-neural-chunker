import unittest

from models.bilstm_chunker import train_and_test


class ChunkerTest(unittest.TestCase):

    def test_chunker(self):
        train_and_test()


if __name__ == '__main__':
    unittest.main()
