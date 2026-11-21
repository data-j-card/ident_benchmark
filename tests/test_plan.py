import pytest

from benchid.plan import Pilot, Weight


def test_every_model_weight_is_pinned_to_a_full_commit():
    pilot = Pilot.load()
    assert set(pilot.weights) == {"qwen", "hidream"}
    assert set(pilot.weights["qwen"]) == {"dit", "text_encoder", "vae"}
    assert set(pilot.weights["hidream"]) == {"dit"}


def test_a_branch_or_short_sha_is_rejected_as_a_pin():
    for revision in ("main", "f68ace85", "sin-fijar"):
        with pytest.raises(ValueError, match="40-hex"):
            Weight("Comfy-Org/Qwen-Image", revision, "vae.safetensors")


def test_jobs_carry_the_pinned_revisions_so_a_new_pin_yields_new_ids():
    pilot = Pilot.load()
    job = pilot.phase("nativos").gens[0]
    model = "qwen" if job.arm.startswith("qwen") else "hidream"
    assert all(w.revision in job.weights for w in pilot.weights[model].values())
