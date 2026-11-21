import pytest

from benchid.splits import references, split_by_angle, view

E = [f"10/TD_RGB_E_{k}.jpg" for k in range(1, 6)]
A = [f"10/TD_RGB_A_{r}_{c}.jpg" for r in range(1, 5) for c in range(9)]
PHOTOS = E + A


def test_inputs_are_three_expressions_and_two_three_quarter_views():
    s = split_by_angle("10", PHOTOS)
    assert s.inputs == (
        "10/TD_RGB_E_1.jpg",
        "10/TD_RGB_E_2.jpg",
        "10/TD_RGB_E_4.jpg",
        "10/TD_RGB_A_1_2.jpg",
        "10/TD_RGB_A_1_6.jpg",
    )


def test_gallery_is_one_photo_per_unseen_column():
    s = split_by_angle("10", PHOTOS)
    assert s.gallery == tuple(f"10/TD_RGB_A_1_{c}.jpg" for c in (0, 1, 3, 4, 5, 7, 8))


def test_no_column_is_split_between_input_and_gallery():
    s = split_by_angle("10", PHOTOS)
    column = lambda p: p[-5]
    input_columns = {column(p) for p in s.inputs if "_A_" in p}
    assert not input_columns & {column(p) for p in s.gallery}


def test_sunglasses_and_eyes_closed_never_enter():
    s = split_by_angle("10", PHOTOS)
    assert not {"10/TD_RGB_E_3.jpg", "10/TD_RGB_E_5.jpg"} & set(s.inputs + s.gallery)


def test_split_does_not_depend_on_photo_order():
    assert split_by_angle("10", PHOTOS) == split_by_angle("10", list(reversed(PHOTOS)))


def test_missing_profile_only_shrinks_the_gallery():
    s = split_by_angle("11", [p for p in PHOTOS if not p.endswith("_8.jpg")])
    assert len(s.inputs) == 5 and len(s.gallery) == 6


def test_missing_input_photo_is_an_error():
    with pytest.raises(ValueError, match="A2"):
        split_by_angle("10", [p for p in PHOTOS if p != "10/TD_RGB_A_1_2.jpg"])


def test_references_are_a_subset_of_inputs_most_informative_first():
    s = split_by_angle("10", PHOTOS)
    assert references(s, 3) == ("10/TD_RGB_E_1.jpg", "10/TD_RGB_A_1_2.jpg", "10/TD_RGB_A_1_6.jpg")
    assert set(references(s, 5)) == set(s.inputs)
    with pytest.raises(ValueError):
        references(s, 6)


def test_calibration_roles_match_the_split():
    s = split_by_angle("10", PHOTOS)
    assert {p for p in PHOTOS if view(p).is_input} == set(s.inputs)
    assert {p for p in PHOTOS if view(p).is_gallery} == set(s.gallery)


def test_other_rows_of_a_gallery_column_are_never_genuine_probes():
    assert all(view(f"10/TD_RGB_A_{r}_0.jpg").in_gallery_column for r in range(1, 5))
    assert not any(view(f"10/TD_RGB_A_{r}_2.jpg").in_gallery_column for r in range(1, 5))
