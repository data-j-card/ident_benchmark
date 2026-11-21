from benchid.arms import ARMS, arm_references, qwen_training_pairs
from benchid.splits import split_by_angle

E = [f"10/TD_RGB_E_{k}.jpg" for k in range(1, 6)]
A = [f"10/TD_RGB_A_{r}_{c}.jpg" for r in range(1, 5) for c in range(9)]
SPLIT = split_by_angle("10", E + A)
INFERENCE_REFS = ("10/TD_RGB_E_1.jpg", "10/TD_RGB_A_1_2.jpg", "10/TD_RGB_A_1_6.jpg")


def test_every_input_photo_is_a_training_target_exactly_once():
    targets = [p.target for p in qwen_training_pairs(SPLIT)]
    assert sorted(targets) == sorted(SPLIT.inputs)


def test_a_target_is_never_among_its_own_controls():
    for pair in qwen_training_pairs(SPLIT):
        assert pair.target not in pair.controls


def test_controls_are_three_input_photos_and_never_gallery():
    for pair in qwen_training_pairs(SPLIT):
        assert len(set(pair.controls)) == 3
        assert set(pair.controls) <= set(SPLIT.inputs)
        assert not set(pair.controls) & set(SPLIT.gallery)


def test_targets_outside_the_references_train_with_the_inference_references():
    pairs = {p.target: p.controls for p in qwen_training_pairs(SPLIT)}
    assert pairs["10/TD_RGB_E_2.jpg"] == INFERENCE_REFS
    assert pairs["10/TD_RGB_E_4.jpg"] == INFERENCE_REFS


def test_both_qwen_arms_get_the_same_three_references():
    for name in ("qwen_referencias", "qwen_lora"):
        assert arm_references(ARMS[name], SPLIT) == INFERENCE_REFS


def test_hidream_reference_arm_gets_only_the_neutral_frontal():
    assert arm_references(ARMS["hidream_referencias"], SPLIT) == ("10/TD_RGB_E_1.jpg",)


def test_hidream_lora_generates_from_text_only():
    assert arm_references(ARMS["hidream_lora"], SPLIT) == ()
