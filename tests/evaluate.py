from typing import List

from src.chunks.chunk import Chunk


def evaluate(expected_chunks: List[Chunk], output_chunks: List[Chunk]):
    expected_set = set(expected_chunks)
    output_set = set(output_chunks)

    tp = len(expected_set & output_set)
    fp = len(output_set - expected_set)
    fn = len(expected_set - output_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
