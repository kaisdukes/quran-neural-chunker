from dataclasses import dataclass

from .location import Location


@dataclass(frozen=True)
class Chunk:
    start: Location
    end: Location
