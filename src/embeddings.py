from typing import Dict

import numpy as np


class Embeddings:

    def __init__(self):
        self._embeddings: Dict[int, np.ndarray] = {}
        self._default_vector: np.ndarray = np.zeros(256)
        self._load_embeddings()

    def get_vector(self, embeddingId: int):
        return self._embeddings.get(embeddingId, self._default_vector)

    def _load_embeddings(self):
        VECTOR_FILE = 'data/vectors.txt'
        with open(VECTOR_FILE, 'r') as file:
            for line in file:
                line = line.strip().split()
                embeddingId = int(line[0][:-1])
                vector = np.array(list(map(float, line[2:])))
                self._embeddings[embeddingId] = vector
