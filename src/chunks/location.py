from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    chapter_number: int
    verse_number: int
    token_number: int

    def __str__(self):
        parts = [str(self.chapter_number), str(self.verse_number)]
        if self.token_number > 0:
            parts.append(str(self.token_number))
        return ':'.join(parts)
