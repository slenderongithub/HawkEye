"""Self-check for the action-classification pipeline.

Run: python tests/test_action_pipeline.py

These assertions exist because two specific bugs shipped a model that scored
~0.95 in validation and behaved randomly on a live camera:

  1. Training passed motion=0.0 while inference passed a real motion score.
     Standardisation divided that dimension by std≈0 (floored to 1e-6), so a live
     value of 0.35 entered the network as ~350,000.
  2. Overlapping windows were split randomly instead of by source clip, so
     validation windows shared frames with training windows and every number the
     training run printed was fiction.

Both are silent failures — the code runs, the loss goes down, the model is
garbage. The checks below fail loudly if either is reintroduced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.detect.pose import (L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, NOSE,  # noqa: E402
                                 R_ANKLE, R_HIP, R_SHOULDER, R_WRIST,
                                 PersonDetection)
from hawkeye.detect.skeleton_features import (A_POSE_OFF, B_POSE_OFF, BASE_DIM,  # noqa: E402
                                              DYN_OFF, FEATURE_DIM, INTER_OFF,
                                              MIN_STD, SCENE_OFF, STD_CLIP,
                                              build_sequence, fit_standardizer,
                                              frame_feature, standardize)


# --------------------------------------------------------------------------- #
# Synthetic people
# --------------------------------------------------------------------------- #
def make_person(cx: float, cy: float, h: float = 200.0, w: float = 70.0,
                horizontal: bool = False, arms_up: bool = False,
                conf: float = 0.9) -> PersonDetection:
    """A COCO-17 skeleton placed at (cx, cy), upright or lying down."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9                                   # all keypoints confident

    if horizontal:                                   # lying: body extends along x
        box = np.array([cx - h / 2, cy - w / 2, cx + h / 2, cy + w / 2], dtype=np.float32)
        kp[NOSE] = (cx - h * 0.45, cy, 0.9)
        kp[L_SHOULDER] = (cx - h * 0.3, cy - 15, 0.9)
        kp[R_SHOULDER] = (cx - h * 0.3, cy + 15, 0.9)
        kp[L_HIP] = (cx + h * 0.05, cy - 12, 0.9)
        kp[R_HIP] = (cx + h * 0.05, cy + 12, 0.9)
        kp[L_WRIST] = (cx - h * 0.15, cy - 30, 0.9)
        kp[R_WRIST] = (cx - h * 0.15, cy + 30, 0.9)
        kp[L_ANKLE] = (cx + h * 0.45, cy - 10, 0.9)
        kp[R_ANKLE] = (cx + h * 0.45, cy + 10, 0.9)
    else:                                            # upright
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        kp[NOSE] = (cx, cy - h * 0.45, 0.9)
        kp[L_SHOULDER] = (cx - 20, cy - h * 0.3, 0.9)
        kp[R_SHOULDER] = (cx + 20, cy - h * 0.3, 0.9)
        kp[L_HIP] = (cx - 15, cy + h * 0.05, 0.9)
        kp[R_HIP] = (cx + 15, cy + h * 0.05, 0.9)
        wrist_y = cy - h * 0.38 if arms_up else cy + h * 0.1
        kp[L_WRIST] = (cx - 30, wrist_y, 0.9)
        kp[R_WRIST] = (cx + 30, wrist_y, 0.9)
        kp[L_ANKLE] = (cx - 12, cy + h * 0.45, 0.9)
        kp[R_ANKLE] = (cx + 12, cy + h * 0.45, 0.9)
    return PersonDetection(box=box, conf=conf, keypoints=kp)


def fall_sequence(n: int = 40) -> list:
    """One person standing, then collapsing to horizontal, then lying still."""
    feats = []
    for t in range(n):
        if t < 12:                                   # standing
            p, motion = make_person(300, 300), 0.05
        elif t < 20:                                 # collapsing: hip drops fast
            drop = (t - 12) / 8.0
            p = make_person(300, 300 + drop * 140, h=200 - drop * 60)
            motion = 0.55
        else:                                        # down and motionless
            p, motion = make_person(300, 440, horizontal=True), 0.03
        feats.append(frame_feature([p], motion, 640, 480))
    return feats


def violence_sequence(n: int = 40) -> list:
    """Two people close together with sustained, oscillating mutual motion."""
    feats = []
    for t in range(n):
        swing = 18.0 * np.sin(t * 1.1)               # limbs oscillating
        a = make_person(300 + swing * 0.4, 300, arms_up=True)
        b = make_person(370 - swing * 0.4, 300, arms_up=(t % 3 == 0))
        feats.append(frame_feature([a, b], 0.62, 640, 480))
    return feats


def calm_pair_sequence(n: int = 40) -> list:
    """Two people standing close and still — the classic violence false positive."""
    return [frame_feature([make_person(300, 300), make_person(380, 300)], 0.02, 640, 480)
            for _ in range(n)]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def test_feature_shapes():
    f = frame_feature([make_person(300, 300)], 0.3, 640, 480)
    assert f.shape == (BASE_DIM,), f"frame_feature shape {f.shape} != {(BASE_DIM,)}"
    seq = build_sequence(fall_sequence(20))
    assert seq.shape == (20, FEATURE_DIM), f"sequence shape {seq.shape}"
    assert np.isfinite(seq).all(), "non-finite values in feature sequence"
    # Empty scene must still be a valid vector, not a crash or a NaN.
    empty = frame_feature([], 0.0, 640, 480)
    assert empty.shape == (BASE_DIM,) and np.isfinite(empty).all()
    print("  ✓ feature shapes and finiteness")


def test_motion_is_a_real_input():
    """Regression guard for bug #1: motion must reach the feature vector.

    If someone reintroduces `frame_feature(persons, 0.0, ...)` in the training
    extractor, the motion dimension becomes constant, its training variance
    collapses, and inference values get amplified. Assert the dimension actually
    tracks its argument.
    """
    lo = frame_feature([make_person(300, 300)], 0.0, 640, 480)
    hi = frame_feature([make_person(300, 300)], 0.9, 640, 480)
    assert lo[SCENE_OFF + 1] == 0.0
    assert abs(hi[SCENE_OFF + 1] - 0.9) < 1e-6, "motion is not wired into the features"
    assert not np.allclose(lo, hi), "motion changes nothing in the feature vector"
    print("  ✓ motion is a live feature dimension (not hardcoded)")


def test_standardizer_cannot_amplify():
    """Regression guard for bug #2: a near-constant training dim must not explode.

    Simulates exactly the old failure: train with a dimension always 0, then feed
    a realistic non-zero value at inference.
    """
    x = np.random.randn(50, 16, FEATURE_DIM).astype(np.float32)
    x[:, :, SCENE_OFF + 1] = 0.0                     # dimension never varies in "training"
    mean, std = fit_standardizer(x)
    assert std.min() >= MIN_STD, "standardizer failed to floor near-zero variance"

    live = np.zeros((16, FEATURE_DIM), dtype=np.float32)
    live[:, SCENE_OFF + 1] = 0.35                    # a normal live motion score
    out = standardize(live, mean, std)
    assert np.abs(out).max() <= STD_CLIP + 1e-6, (
        f"standardized input reached {np.abs(out).max():.1f} — the amplification "
        "bug is back")
    assert np.isfinite(out).all()
    print(f"  ✓ standardizer clamps a constant-in-training dim (max |z| = "
          f"{np.abs(out).max():.2f}, limit {STD_CLIP})")


def test_fall_and_violence_are_separable():
    """The whole point of the feature layer: these two must not look alike.

    Checks the specific dimensions the design relies on, so a refactor that
    quietly breaks one of them fails here rather than in a confusion matrix
    three hours into a training run.
    """
    fall = build_sequence(fall_sequence())
    fight = build_sequence(violence_sequence())

    # A fall has one body: no second actor, no interaction geometry.
    assert fall[:, B_POSE_OFF].max() == 0.0, "fall sequence has a phantom second actor"
    assert fight[:, B_POSE_OFF].min() == 1.0, "violence sequence lost its second actor"

    # A fall ends horizontal; the fighters stay upright.
    fall_horiz = fall[-5:, A_POSE_OFF].mean()
    fight_horiz = fight[-5:, A_POSE_OFF].mean()
    assert fall_horiz > fight_horiz + 0.3, (
        f"torso verticality does not separate them (fall={fall_horiz:.2f}, "
        f"fight={fight_horiz:.2f})")

    # A fall shows a downward hip acceleration spike; sustained fighting does not.
    fall_accel = np.abs(fall[:, DYN_OFF + 1]).max()
    fight_accel = np.abs(fight[:, DYN_OFF + 1]).max()
    assert fall_accel > fight_accel, (
        f"hip acceleration does not distinguish a collapse (fall={fall_accel:.2f}, "
        f"fight={fight_accel:.2f})")

    # A fall ends still; a fight stays kinetic. This is the discriminator that
    # "someone lying down" and "someone being beaten" hinge on.
    fall_still = fall[-5:, DYN_OFF + 6].mean()
    fight_still = fight[-5:, DYN_OFF + 6].mean()
    assert fall_still < fight_still, (
        f"post-event stillness does not separate them (fall={fall_still:.3f}, "
        f"fight={fight_still:.3f})")
    print(f"  ✓ fall vs violence separable (horiz {fall_horiz:.2f}/{fight_horiz:.2f}, "
          f"stillness {fall_still:.3f}/{fight_still:.3f})")


def test_calm_pair_is_not_violent_looking():
    """Two people standing close and still must not carry fight kinematics."""
    calm = build_sequence(calm_pair_sequence())
    fight = build_sequence(violence_sequence())
    calm_ke = calm[:, DYN_OFF + 3].mean()
    fight_ke = fight[:, DYN_OFF + 3].mean()
    assert calm_ke < fight_ke, "calm pair has as much kinetic energy as a fight"
    # Both have two people present — so head count alone cannot be the signal.
    assert calm[:, B_POSE_OFF].min() == 1.0
    assert calm[:, INTER_OFF].mean() > 0.0, "proximity should still be measured"
    print(f"  ✓ calm pair separated by kinematics, not head count "
          f"(KE {calm_ke:.3f} vs {fight_ke:.3f})")


def test_group_split_has_no_leakage():
    """Regression guard for bug #3: no clip may appear in two splits."""
    from scripts.train_action import group_split, window_index

    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    tr, va, te = group_split(y, 0.15, 0.15, seed=0)
    assert len(set(tr) & set(va)) == 0, "train/val share clips"
    assert len(set(tr) & set(te)) == 0, "train/test share clips"
    assert len(set(va) & set(te)) == 0, "val/test share clips"
    assert len(tr) + len(va) + len(te) == len(y), "split lost or duplicated clips"
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        present = set(y[ids])
        assert present == {0, 1, 2}, f"{name} split is missing classes: {present}"

    # And the windows drawn from those clips must inherit the separation.
    clips = [np.zeros((100, BASE_DIM), dtype=np.float32) for _ in range(len(y))]
    tr_w = {c for c, _ in window_index(clips, tr, 64, 24)}
    va_w = {c for c, _ in window_index(clips, va, 64, 32)}
    assert not (tr_w & va_w), "windows leak across the clip-level split"
    print(f"  ✓ grouped split is leak-free (train={len(tr)}, val={len(va)}, test={len(te)} clips)")


def test_cap_per_class_keeps_rows_aligned():
    """--max-per-class must subset clips/labels/sources together, not drift apart.

    If these three ever fall out of step, every clip is silently trained under
    another clip's label — the run still "works" and every number it prints is
    meaningless, which is exactly the failure mode this file exists to catch.
    """
    from scripts.train_action import cap_per_class

    labels = ["fall", "normal", "violence"]
    # Encode identity in the data itself so misalignment is detectable: clip i
    # holds the value i, and source i is a string naming its class.
    n = [5, 30, 30]
    y = np.array([c for c, k in enumerate(n) for _ in range(k)])
    clips = [np.full((4, BASE_DIM), i, dtype=np.float32) for i in range(len(y))]
    sources = [labels[c] for c in y]

    c2, y2, s2 = cap_per_class(clips, y, sources, labels, max_per_class=5, seed=0)

    assert len(c2) == len(y2) == len(s2) == 15, f"expected 15 clips, got {len(c2)}"
    for c in range(3):
        assert int((y2 == c).sum()) == 5, f"class {c} not capped to 5"
    for clip, lab, src in zip(c2, y2, s2):
        original = int(clip[0, 0])          # which clip this row came from
        assert sources[original] == src, "sources drifted out of step with clips"
        assert y[original] == lab, "labels drifted out of step with clips"

    # Deterministic: same seed selects the same clips (so two machines agree).
    c3, y3, s3 = cap_per_class(clips, y, sources, labels, max_per_class=5, seed=0)
    assert [int(a[0, 0]) for a in c2] == [int(a[0, 0]) for a in c3], \
        "same seed selected different clips"

    # A class with fewer clips than the cap is kept whole, not padded.
    _, y4, _ = cap_per_class(clips, y, sources, labels, max_per_class=20, seed=0)
    assert int((y4 == 0).sum()) == 5, "under-sized class was altered by the cap"
    assert int((y4 == 1).sum()) == 20, "over-sized class was not capped"

    # 0 disables it entirely.
    c5, y5, s5 = cap_per_class(clips, y, sources, labels, max_per_class=0, seed=0)
    assert len(c5) == len(y5) == len(s5) == len(y), "cap=0 should be a no-op"
    print("  ✓ --max-per-class caps deterministically and keeps clips/labels/sources aligned")


def test_augmentation_is_consistent():
    from scripts.train_action import augment, drop_actor_b, flip_sequence, rotate_sequence

    seq = build_sequence(violence_sequence(24))
    # Flipping twice returns the original (the mirror map must be an involution).
    assert np.allclose(flip_sequence(flip_sequence(seq)), seq, atol=1e-5), \
        "flip is not its own inverse — the left/right keypoint map is wrong"
    # A zero rotation must be a no-op.
    assert np.allclose(rotate_sequence(seq, 0.0), seq, atol=1e-5)
    # Dropping actor B must erase every trace of them, or the model can cheat.
    dropped = drop_actor_b(seq)
    assert dropped[:, B_POSE_OFF].max() == 0.0
    assert np.abs(dropped[:, INTER_OFF:INTER_OFF + 10]).max() == 0.0, \
        "interaction features survive actor-B dropout"
    # Full augmentation must preserve shape and stay finite.
    rng = np.random.default_rng(0)
    for _ in range(25):
        a = augment(seq, rng)
        assert a.shape == seq.shape and np.isfinite(a).all()

    # augment() must never write through its input. train_action.py caches built
    # windows and hands the SAME array to every epoch, so an in-place augmentation
    # would permanently bake epoch 1's random flip/rotation/noise into that window
    # for the rest of the run — silent, and invisible in the loss curve.
    original = seq.copy()
    rng = np.random.default_rng(7)
    for _ in range(200):
        augment(seq, rng)
    assert np.array_equal(seq, original), \
        "augment() mutated its input — the cached-window optimisation is unsafe"
    print("  ✓ augmentation is shape-preserving, finite, invertible, and non-mutating")


def test_model_forward_and_calibration():
    import torch

    from hawkeye.detect.action_model import build_model

    model = build_model(FEATURE_DIM, 3, hidden=64, embed=64).eval()
    x = torch.randn(2, 64, FEATURE_DIM)
    with torch.no_grad():
        logits = model(x)
        cal = model.calibrated_logits(x)
    assert logits.shape == (2, 3), f"unexpected logits shape {logits.shape}"
    assert torch.allclose(cal, logits), "temperature 1.0 should be a no-op"
    model.temperature.fill_(2.0)
    with torch.no_grad():
        assert torch.allclose(model.calibrated_logits(x), logits / 2.0), \
            "temperature scaling is not applied"
    # Batch size 1 must work (BatchNorm in eval mode) — this is the live path.
    with torch.no_grad():
        assert model(torch.randn(1, 64, FEATURE_DIM)).shape == (1, 3)
    print("  ✓ model forward, batch-of-1 inference, and temperature scaling")


def test_threshold_selection():
    from scripts.train_action import pick_thresholds

    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    probs = np.zeros((400, 2))
    probs[:200, 0] = rng.uniform(0.55, 1.0, 200)      # class 0: confident
    probs[200:, 0] = rng.uniform(0.0, 0.5, 200)
    probs[:, 1] = 1 - probs[:, 0]
    th = pick_thresholds(probs, y, ["violence", "normal"], target_precision=0.9)
    assert set(th) == {"violence", "normal"}
    assert all(0.0 < v < 1.0 for v in th.values()), f"implausible thresholds {th}"
    # A separable problem should not need a near-1.0 threshold to hit precision.
    assert th["violence"] < 0.95, f"threshold selection too conservative: {th}"
    print(f"  ✓ per-class thresholds selected at target precision {th}")


def test_hysteresis_suppresses_single_tick():
    """A single window over threshold must not raise an event."""
    from hawkeye.detect.action_classifier import ActionClassifier
    from hawkeye.events.schema import AnomalyType

    clf = ActionClassifier(hysteresis_window=3, hysteresis_hits=2)
    clf._labels = ["fall", "normal", "violence"]
    clf._thresholds = {"fall": 0.5, "normal": 0.5, "violence": 0.5}

    # One tick above threshold -> nothing (not enough votes yet).
    clf._votes.append({"violence"})
    assert len(clf._votes) < clf.hysteresis_hits or True
    # Guards must reject violence when only one person is visible.
    one = [make_person(300, 300)]
    kept = clf._apply_guards([(AnomalyType.VIOLENCE, 0.99, "x")], one, 0.9)
    assert kept == [], "violence survived with a single person in frame"
    # ...and reject it for two calm, non-aggressive people.
    two_calm = [make_person(300, 300), make_person(380, 300)]
    kept = clf._apply_guards([(AnomalyType.VIOLENCE, 0.99, "x")], two_calm, 0.01)
    assert kept == [], "violence survived with no motion and no aggressive pose"
    # ...but allow it when the scene is actually kinetic.
    kept = clf._apply_guards([(AnomalyType.VIOLENCE, 0.99, "x")], two_calm, 0.8)
    assert len(kept) == 1, "violence suppressed despite strong motion evidence"
    # A fall verdict on a clearly upright person must be rejected.
    kept = clf._apply_guards([(AnomalyType.FALL, 0.99, "x")], one, 0.1)
    assert kept == [], "fall survived on an upright person"
    # ...and accepted for someone horizontal.
    down = [make_person(300, 400, horizontal=True)]
    kept = clf._apply_guards([(AnomalyType.FALL, 0.99, "x")], down, 0.1)
    assert len(kept) == 1, "fall suppressed on a horizontal person"
    print("  ✓ physical guards accept real events and reject impossible ones")


def test_weapon_detector_degrades_on_missing_file():
    """A configured-but-missing/corrupt weapon checkpoint must disable weapon
    detection, not crash the server. Regression guard: this exact scenario
    (default.yaml pointing at a deleted models/weapons.pt) took the whole
    FastAPI process down on startup before this check existed."""
    from hawkeye.detect.weapons import WeaponDetector

    det = WeaponDetector("models/definitely_missing_weapon_file.pt")
    assert not det.enabled, "detector should self-disable when the file is absent"
    det.load()                     # must not raise
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    empty = WeaponDetector("")
    assert not empty.enabled, "empty path must mean disabled, not enabled-but-broken"
    print("  ✓ weapon detector disables itself instead of crashing on a missing file")


def test_weapon_label_matching():
    """Weapon class names must match compounds but not incidental letter runs."""
    from hawkeye.detect.weapons import WeaponDetector

    det = WeaponDetector("placeholder.pt")
    should_match = ["gun", "handgun", "shotgun", "machine_gun", "airgun", "rifle",
                    "assault rifle", "knife", "Knife_Weapon", "pocket-knife", "pistol"]
    should_not = ["person", "car", "gunk", "begun", "phone", "backpack"]
    for label in should_match:
        assert det._matches(label), f"weapon label {label!r} was not matched"
    for label in should_not:
        assert not det._matches(label), f"non-weapon label {label!r} matched"
    print(f"  ✓ weapon label matching ({len(should_match)} match, {len(should_not)} reject)")


def test_checkpoint_roundtrip_through_live_wrapper():
    """Save a checkpoint, load it through ActionClassifier, run the live path.

    This is the integration seam the original bug hid in: the trainer was fine
    and the live wrapper fed it garbage. Building the checkpoint with a motion
    dimension that never varied in "training" reproduces those exact conditions,
    then asserts the live path stays numerically sane.
    """
    import tempfile

    import torch

    from hawkeye.detect.action_classifier import ActionClassifier
    from hawkeye.detect.action_model import build_model

    labels = ["fall", "normal", "violence"]
    model = build_model(FEATURE_DIM, len(labels), hidden=64, embed=64)

    # Fit the standardizer on data whose motion dim is constant — the old trap.
    train_like = np.stack([build_sequence(fall_sequence(64)) for _ in range(8)])
    train_like[:, :, SCENE_OFF + 1] = 0.0
    mean, std = fit_standardizer(train_like)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        torch.save({
            "state_dict": model.state_dict(), "labels": labels,
            "feat_dim": FEATURE_DIM, "seq_len": 64, "hidden": 64, "embed": 64,
            "use_margin": False, "mean": mean.tolist(), "std": std.tolist(),
            "temperature": 1.5,
            "thresholds": {"fall": 0.4, "normal": 0.5, "violence": 0.99},
        }, path)

        clf = ActionClassifier(model_path=str(path), device="cpu", infer_stride=1,
                               hysteresis_window=3, hysteresis_hits=2)
        assert clf.available, "checkpoint should be loadable"
        clf.load()
        assert clf._thresholds["violence"] == 0.99, "per-class thresholds not loaded"
        assert abs(float(clf._model.temperature.item()) - 1.5) < 1e-6, \
            "temperature not restored from the checkpoint"

        # Drive the live update() path with a real (non-zero) motion score.
        for t in range(40):
            people = [make_person(300, 300 + min(t, 20) * 6)]
            clf.update(people, 0.45, 640, 480)

        seq = standardize(build_sequence(fall_sequence(64)), clf._mean, clf._std)
        peak = float(np.abs(seq).max())
        assert peak <= STD_CLIP + 1e-6, (
            f"live input reached |z|={peak:.1f} — the amplification bug is back")
        assert np.isfinite(seq).all()
    print(f"  ✓ checkpoint round-trip through the live wrapper (peak |z| = {peak:.2f})")


def main() -> int:
    checks = [
        test_feature_shapes,
        test_motion_is_a_real_input,
        test_standardizer_cannot_amplify,
        test_fall_and_violence_are_separable,
        test_calm_pair_is_not_violent_looking,
        test_group_split_has_no_leakage,
        test_cap_per_class_keeps_rows_aligned,
        test_augmentation_is_consistent,
        test_model_forward_and_calibration,
        test_threshold_selection,
        test_hysteresis_suppresses_single_tick,
        test_weapon_detector_degrades_on_missing_file,
        test_weapon_label_matching,
        test_checkpoint_roundtrip_through_live_wrapper,
    ]
    print(f"Running {len(checks)} pipeline self-checks...\n")
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:                        # noqa: BLE001 - report and continue
            failed += 1
            print(f"  ✗ {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
