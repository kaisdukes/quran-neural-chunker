from typing import List

from src.chunks.chunk import Chunk


class Evaluator:

    def __init__(self):
        self._expected_chunks = 0
        self._output_chunks = 0
        self._equivalent_chunks = 0

    def compare(self, expected_chunks: List[Chunk], output_chunks: List[Chunk]):
        expected_set = set(expected_chunks)
        output_set = set(output_chunks)

        self._expected_chunks += len(expected_set)
        self._output_chunks += len(output_set)
        self._equivalent_chunks += len(expected_set & output_set)

    @property
    def precision(self):
        return self._equivalent_chunks / self._output_chunks

    @property
    def recall(self):
        return self._equivalent_chunks / self._expected_chunks

    @property
    def f1_score(self):
        precision = self.precision
        recall = self.recall
        return 2 * (precision * recall) / (precision + recall)
