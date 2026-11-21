from typing import Literal

from benchid.splits import view

Variant = Literal["A", "B"]

EXPRESSION = {"E1": "neutral expression", "E2": "smiling", "E4": "surprised expression"}
SESSION = ("plain blue background", "studio lighting")


def details(photo: str, clothing: str | None) -> tuple[str, ...]:
    tag = view(photo).tag
    pose = ("frontal view", EXPRESSION[tag]) if tag.startswith("E") else ("three-quarter view", "neutral expression")
    worn = (f"wearing {clothing}",) if clothing else ()
    return pose + worn + SESSION


def training_text(subject: str, photo: str, variant: Variant, clothing: str | None = None) -> str:
    return subject if variant == "A" else ", ".join((subject, *details(photo, clothing)))
