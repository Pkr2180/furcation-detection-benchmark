import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from furcadino import evalap as E


def make(n=60, seed=0):
    rng = np.random.default_rng(seed)
    gts, preds = {}, {}
    for i in range(n):
        k = rng.integers(0, 3)
        b = rng.uniform(0, 400, (k, 2)); wh = rng.uniform(30, 60, (k, 2))
        gb = np.c_[b, b + wh]
        gts[i] = dict(boxes=gb, labels=np.full(k, 10))
        pb = list(gb + rng.normal(0, 3, gb.shape)); ps = list(rng.uniform(0.5, 1, k))
        for _ in range(rng.integers(0, 3)):
            c = rng.uniform(0, 400, 2); pb.append(np.r_[c, c + 40]); ps.append(rng.uniform(0.0, 0.5))
        preds[i] = dict(boxes=np.array(pb).reshape(-1, 4), scores=np.array(ps), labels=np.full(len(ps), 10))
    return preds, gts


def test_perfect_predictions_ap_one():
    _, gts = make()
    perfect = {i: dict(boxes=g["boxes"], scores=np.ones(len(g["boxes"])), labels=g["labels"]) for i, g in gts.items()}
    assert abs(E.ap_class(perfect, gts, 10) - 1.0) < 1e-9


def test_no_predictions_ap_zero_and_nan_without_gt():
    _, gts = make()
    assert E.ap_class({}, gts, 10) == 0.0
    assert np.isnan(E.ap_class({}, gts, 4))


def test_ap_matches_hand_example():
    gts = {0: dict(boxes=np.array([[0, 0, 10, 10]]), labels=np.array([1])),
           1: dict(boxes=np.array([[0, 0, 10, 10]]), labels=np.array([1]))}
    preds = {0: dict(boxes=np.array([[0, 0, 10, 10]]), scores=np.array([0.9]), labels=np.array([1])),
             1: dict(boxes=np.array([[50, 50, 60, 60], [0, 0, 10, 10]]), scores=np.array([0.8, 0.7]), labels=np.array([1, 1]))}
    # ranked: TP(.9), FP(.8), TP(.7) -> precision at recall 0.5 = 1, at recall 1.0 = 2/3
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert abs(E.ap_class(preds, gts, 1) - expected) < 1e-9


def test_conformal_recall_guarantee_holds_empirically():
    rng = np.random.default_rng(1)
    alpha, covered = 0.2, []
    for rep in range(400):
        cal = rng.beta(4, 1.5, 40); test = rng.beta(4, 1.5, 200)
        tau = E.conformal_threshold(cal, alpha)
        covered.append((test >= tau).mean())
    assert np.mean(covered) >= 1 - alpha - 0.01


def test_conformal_none_when_calibration_too_small():
    assert E.conformal_threshold(np.array([0.3, 0.4]), 0.1) is None


def test_group_bootstrap_ci_contains_point():
    preds, gts = make()
    groups = {i: i // 3 for i in gts}
    p, lo, hi, _ = E.group_bootstrap(lambda ids: E.ap_class(preds, gts, 10, ids=ids), list(gts), groups, n=200)
    assert lo <= p <= hi


def test_recall_at_fppi_monotone():
    preds, gts = make()
    r = [E.recall_at_fppi(preds, gts, 10, f) for f in (0.01, 0.1, 0.5, 2.0)]
    assert all(b >= a for a, b in zip(r, r[1:]))
