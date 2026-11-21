import json

import pytest

from benchid.backends import FakeBackend
from benchid.plan import Pilot
from benchid.runner import BudgetExhausted, Runner
from benchid.splits import split_by_angle


def names(identity):
    return [f"{identity}/TD_RGB_E_{k}.jpg" for k in range(1, 6)] + [
        f"{identity}/TD_RGB_A_{r}_{c}.jpg" for r in range(1, 5) for c in range(9)
    ]


@pytest.fixture
def pilot():
    return Pilot.load()


def runner(pilot, tmp_path):
    return Runner(pilot, FakeBackend(), tmp_path, lambda i: split_by_angle(i, names(i)), tmp_path / "fotos")


def test_native_phase_covers_both_reference_arms_for_every_development_identity_and_scenario(pilot):
    phase = pilot.phase("nativos")
    cells = {(j.arm, j.identity, j.scenario) for j in phase.gens}
    assert len(cells) == 2 * 3 * 10
    assert not phase.trains


def test_captions_phase_trains_both_variants_on_both_models(pilot):
    trains = pilot.phase("captions").trains
    assert {(t.model, t.caption) for t in trains} == {("qwen", "A"), ("qwen", "B"), ("hidream", "A"), ("hidream", "B")}
    assert len(trains) == 8


def test_hyperparameter_phase_refuses_to_run_before_a_caption_is_chosen(pilot):
    with pytest.raises(ValueError, match="caption"):
        pilot.phase("hiper")


def test_a_second_run_resumes_without_redoing_or_paying_again(pilot, tmp_path):
    r = runner(pilot, tmp_path)
    r.run(pilot.phase("sonda_hidream"))
    images = sorted((tmp_path / "imagenes").glob("*.png"))
    spent = r.spent_usd()
    stamps = [p.stat().st_mtime_ns for p in images]
    r.run(pilot.phase("sonda_hidream"))
    assert sorted((tmp_path / "imagenes").glob("*.png")) == images
    assert [p.stat().st_mtime_ns for p in images] == stamps
    assert r.spent_usd() == spent
    assert len(images) == 10


def test_a_lora_image_records_the_checkpoint_it_was_generated_from(pilot, tmp_path):
    runner(pilot, tmp_path).run(pilot.phase("sonda_hidream"))
    record = json.loads(next((tmp_path / "imagenes").glob("*.json")).read_text())
    assert record["job"]["arm"] == "hidream_lora"
    assert record["job"]["epoch"] == 16
    assert record["refs"] == []


def test_runs_stop_once_the_spend_reaches_the_cap(pilot, tmp_path):
    capped = Pilot(**{**pilot.__dict__, "raw": {**pilot.raw, "tope_usd": 0.05}})
    r = runner(capped, tmp_path)
    with pytest.raises(BudgetExhausted):
        r.run(capped.phase("nativos"))
    assert 0 < len(list((tmp_path / "imagenes").glob("*.png"))) < len(capped.phase("nativos").gens)
