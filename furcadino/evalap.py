"""Detection evaluation: AP (COCO 101-point), FROC, grouped bootstrap, paired comparison, conformal recall control.
Predictions / ground truth are dicts: image_id -> dict(boxes=(N,4) xyxy, scores=(N,), labels=(N,)).
"""
import numpy as np


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def match_class(preds, gts, cls, thr=0.5, ids=None):
    """Greedy score-ordered matching. Returns (scores, tp flags, n_gt, per-image tp/fp counts)."""
    ids = list(gts) if ids is None else ids
    S, T, n_gt = [], [], 0
    for i in ids:
        g = gts[i]; p = preds.get(i, dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, int)))
        gb = np.asarray(g["boxes"]).reshape(-1, 4)[np.asarray(g["labels"]) == cls]
        pm = np.asarray(p["labels"]) == cls
        pb = np.asarray(p["boxes"]).reshape(-1, 4)[pm]; ps = np.asarray(p["scores"])[pm]
        order = np.argsort(-ps); pb, ps = pb[order], ps[order]
        n_gt += len(gb)
        iou = iou_matrix(pb, gb)
        used = np.zeros(len(gb), bool)
        for k in range(len(pb)):
            tp = 0
            if len(gb):
                j = np.argmax(np.where(used, -1, iou[k]))
                if iou[k, j] >= thr and not used[j]:
                    used[j] = True; tp = 1
            S.append(ps[k]); T.append(tp)
    return np.array(S), np.array(T), n_gt


def ap_from_matches(S, T, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(S) == 0:
        return 0.0
    o = np.argsort(-S, kind="stable"); T = T[o]
    tp = np.cumsum(T); fp = np.cumsum(1 - T)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101); ap = 0.0
    for r in rs:
        idx = np.searchsorted(rec, r, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


def ap_class(preds, gts, cls, thr=0.5, ids=None):
    return ap_from_matches(*match_class(preds, gts, cls, thr, ids))


def ap_multi_iou(preds, gts, cls, ids=None, thrs=np.arange(0.5, 0.96, 0.05)):
    return float(np.nanmean([ap_class(preds, gts, cls, t, ids) for t in thrs]))


def mean_ap50(preds, gts, classes, ids=None):
    aps = [ap_class(preds, gts, c, 0.5, ids) for c in classes]
    return float(np.nanmean(aps))


def recall_at_fppi(preds, gts, cls, fppi, thr=0.5, ids=None):
    """Lesion recall at the score threshold that yields `fppi` false positives per image."""
    ids = list(gts) if ids is None else ids
    S, T, n_gt = match_class(preds, gts, cls, thr, ids)
    if n_gt == 0 or len(S) == 0:
        return 0.0
    o = np.argsort(-S, kind="stable"); T = T[o]
    fp = np.cumsum(1 - T) / len(ids); tp = np.cumsum(T) / n_gt
    ok = np.where(fp <= fppi)[0]
    return float(tp[ok[-1]]) if len(ok) else 0.0


def group_bootstrap(metric_fn, ids, groups, n=2000, seed=0):
    """Resample groups with replacement; metric_fn(list_of_image_ids) -> float. Returns (point, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    byg = {}
    for i in ids:
        byg.setdefault(groups[i], []).append(i)
    keys = list(byg)
    point = metric_fn(list(ids))
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sample = [i for k in pick for i in byg[keys[k]]]
        vals.append(metric_fn(sample))
    vals = np.array(vals)
    return point, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), vals


def paired_bootstrap(metric_a, metric_b, ids, groups, n=2000, seed=0):
    """P(A > B) and 95% CI of A-B with the same resampled groups. metric_*(ids)->float."""
    rng = np.random.default_rng(seed)
    byg = {}
    for i in ids:
        byg.setdefault(groups[i], []).append(i)
    keys = list(byg)
    d = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = [i for k in pick for i in byg[keys[k]]]
        d.append(metric_a(s) - metric_b(s))
    d = np.array(d)
    return float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), float(np.nanmean(d > 0))


def lesion_scores(preds, gts, cls, thr=0.3, ids=None):
    """For each GT lesion of class cls: the highest score among predictions of that class with IoU >= thr (0 if none)."""
    ids = list(gts) if ids is None else ids
    out = []
    for i in ids:
        g = gts[i]; gb = np.asarray(g["boxes"]).reshape(-1, 4)[np.asarray(g["labels"]) == cls]
        if not len(gb):
            continue
        p = preds.get(i, dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, int)))
        pm = np.asarray(p["labels"]) == cls
        iou = iou_matrix(np.asarray(p["boxes"]).reshape(-1, 4)[pm], gb)
        sc = np.asarray(p["scores"])[pm]
        for j in range(len(gb)):
            m = iou[:, j] >= thr if len(sc) else np.zeros(0, bool)
            out.append(float(sc[m].max()) if m.any() else 0.0)
    return np.array(out)


def conformal_threshold(cal_scores, alpha):
    """Split-conformal lower threshold so that a new lesion has score >= tau with prob >= 1 - alpha
    (exchangeable lesions). Returns None if the calibration set is too small for this alpha."""
    n = len(cal_scores)
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return None
    tau = float(np.sort(cal_scores)[k - 1])
    # a lesion with no overlapping prediction has score 0 and can never be flagged: if the quantile is 0 the detector's
    # proposal recall is below 1 - alpha and the target is infeasible at this alpha
    return tau if tau > 0 else None


def apply_threshold(preds, gts, cls, tau, thr=0.3, ids=None):
    """Lesion recall and false-positive boxes per image when only predictions with score >= tau are kept."""
    ids = list(gts) if ids is None else ids
    n_les, hit, fp = 0, 0, 0
    for i in ids:
        g = gts[i]; gb = np.asarray(g["boxes"]).reshape(-1, 4)[np.asarray(g["labels"]) == cls]
        p = preds.get(i, dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, int)))
        keep = (np.asarray(p["labels"]) == cls) & (np.asarray(p["scores"]) >= tau)
        pb = np.asarray(p["boxes"]).reshape(-1, 4)[keep]
        iou = iou_matrix(pb, gb)
        n_les += len(gb)
        hit += int((iou.max(0) >= thr).sum()) if len(gb) and len(pb) else 0
        fp += int((iou.max(1) < thr).sum()) if len(pb) and len(gb) else len(pb)
    return dict(recall=hit / max(n_les, 1), fp_per_image=fp / max(len(ids), 1), n_lesions=n_les)
