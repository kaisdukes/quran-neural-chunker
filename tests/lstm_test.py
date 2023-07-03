import unittest

from src.models.lstm_chunker import train_and_test


class LstmTest(unittest.TestCase):

    def test_lstm(self):
        train_and_test()


if __name__ == '__main__':
    unittest.main()
