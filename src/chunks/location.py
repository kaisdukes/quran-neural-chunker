from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    chapter_number: int
    verse_number: int
    token_number: int
