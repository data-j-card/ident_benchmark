import re
from dataclasses import dataclass

INPUT_VIEWS = ("E1", "E2", "E4", "A2", "A6")
REFERENCE_ORDER = ("E1", "A2", "A6", "E2", "E4")
ROW = 1

_PHOTO = re.compile(r"TD_RGB_(?:E_(?P<expression>\d)|A_(?P<row>\d)_(?P<column>\d))\.jpg$")


@dataclass(frozen=True)
class View:
    tag: str
    row: int

    @property
    def is_input(self) -> bool:
        return self.tag in INPUT_VIEWS and self.row in (0, ROW)

    @property
    def in_gallery_column(self) -> bool:
        return self.tag.startswith("A") and self.tag not in INPUT_VIEWS

    @property
    def is_gallery(self) -> bool:
        return self.in_gallery_column and self.row == ROW


def view(photo: str) -> View:
    m = _PHOTO.search(photo)
    if m is None:
        raise ValueError(f"not a Tufts RGB photo: {photo}")
    if m["expression"]:
        return View(f"E{m['expression']}", 0)
    return View(f"A{m['column']}", int(m["row"]))


@dataclass(frozen=True)
class Split:
    identity: str
    inputs: tuple[str, ...]
    gallery: tuple[str, ...]


def split_by_angle(identity: str, photos: list[str]) -> Split:
    by_tag = {v.tag: p for p in photos if (v := view(p)).row in (0, ROW)}
    missing = [t for t in INPUT_VIEWS if t not in by_tag]
    if missing:
        raise ValueError(f"{identity}: missing input photos {missing}")
    gallery = tuple(p for _, p in sorted(by_tag.items()) if view(p).is_gallery)
    if not gallery:
        raise ValueError(f"{identity}: no gallery left")
    return Split(identity, tuple(by_tag[t] for t in INPUT_VIEWS), gallery)


def references(split: Split, r: int) -> tuple[str, ...]:
    if not 0 < r <= len(split.inputs):
        raise ValueError(f"{split.identity}: need 0 < r <= {len(split.inputs)}")
    by_tag = {view(p).tag: p for p in split.inputs}
    return tuple(by_tag[t] for t in REFERENCE_ORDER[:r])
