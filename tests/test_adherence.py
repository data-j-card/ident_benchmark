import hashlib
import json
from dataclasses import asdict

import pytest

from benchid.adherence import JudgeConfig, Snapshot, score_run, verify
from benchid.plan import Pilot


@pytest.fixture
def cfg():
    return JudgeConfig.load()


def test_the_question_follows_the_vqascore_template_exactly(cfg):
    prompt = cfg.prompt("a red car")
    assert 'USER: <image>\nDoes this figure show "a red car"? Please answer yes or no. ASSISTANT: ' in prompt
    assert prompt.endswith("ASSISTANT: ")
    assert cfg.answer == "Yes"


def test_every_snapshot_is_pinned_to_a_full_commit(cfg):
    assert set(cfg.revisions) == {"zhiqiulin/clip-flant5-xl", "google/flan-t5-xl", "openai/clip-vit-large-patch14-336"}


def test_a_branch_or_short_sha_is_rejected_as_a_snapshot(tmp_path):
    for revision in ("main", "3b4a6b1b", "sin-fijar"):
        with pytest.raises(ValueError, match="40-hex"):
            Snapshot("zhiqiulin/clip-flant5-xl", revision, tmp_path, {})


def snapshot_of(tmp_path, files):
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    sums = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    return Snapshot("org/model", "0" * 40, tmp_path, sums)


def test_verify_accepts_the_pinned_files(tmp_path):
    verify(snapshot_of(tmp_path, {"config.json": b"{}", "pytorch_model.bin": b"weights"}))


def test_verify_rejects_a_tampered_file(tmp_path):
    snapshot = snapshot_of(tmp_path, {"config.json": b"{}", "pytorch_model.bin": b"weights"})
    (tmp_path / "pytorch_model.bin").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="pytorch_model.bin"):
        verify(snapshot)


def test_verify_rejects_a_missing_file(tmp_path):
    snapshot = snapshot_of(tmp_path, {"config.json": b"{}"})
    (tmp_path / "config.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(snapshot)


class FakeJudge:
    def __init__(self):
        self.calls = []

    def __call__(self, image, text):
        self.calls.append((image.name, text))
        return 0.5


def fake_run(tmp_path, jobs):
    images = tmp_path / "imagenes"
    images.mkdir()
    for job in jobs:
        (images / f"{job.id}.png").write_bytes(b"")
        (images / f"{job.id}.json").write_text(json.dumps({"job": asdict(job)}), encoding="utf-8")


def test_all_arms_of_a_scenario_are_judged_on_the_neutral_scenario_text(cfg, tmp_path):
    pilot = Pilot.load()
    jobs = [j for j in pilot.phase("nativos").gens if j.scenario == "P04"]
    jobs += [j for j in pilot.phase("sonda_hidream").gens if j.scenario == "P04"]
    assert {j.arm for j in jobs} == {"qwen_referencias", "hidream_referencias", "hidream_lora"}
    fake_run(tmp_path, jobs)
    judge = FakeJudge()
    score_run(tmp_path, judge, cfg, pilot.scenarios)
    texts = {text for _, text in judge.calls}
    assert texts == {"A photograph of a person" + pilot.scenarios["P04"]}
    assert pilot.raw["token"] not in texts.pop()


def test_a_second_run_skips_images_already_judged(cfg, tmp_path):
    pilot = Pilot.load()
    jobs = pilot.phase("nativos").gens[:3]
    fake_run(tmp_path, jobs)
    assert score_run(tmp_path, FakeJudge(), cfg, pilot.scenarios) == 3
    record = json.loads((tmp_path / "adherencia" / f"{jobs[0].id}.json").read_text(encoding="utf-8"))
    assert record["vqascore"] == 0.5 and record["judge"] == cfg.revisions
    judge = FakeJudge()
    assert score_run(tmp_path, judge, cfg, pilot.scenarios) == 0
    assert judge.calls == []
