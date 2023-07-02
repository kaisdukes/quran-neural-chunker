from typing import List

from pandas import DataFrame

from .chunk import Chunk
from .location import Location


def get_chunks(df: DataFrame):
    chunks: List[Chunk] = []
    start: Location = None

    for _, row in df.iterrows():
        loc = Location(row['chapter_number'], row['verse_number'], row['token_number'])
        if start is None:
            start = loc
        if row['chunk_end'] == 1:
            end = loc
            chunk = Chunk(start, end)
            chunks.append(chunk)
            start = None

    return chunks
