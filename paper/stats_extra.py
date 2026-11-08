"""Extra statistics for the manuscript, computed only from the saved prediction files in res_all/.
  python paper/stats_extra.py res_all manifest.json paper/stats_extra.json

Adds what analyze.py does not store: paired group-bootstrap differences for the pre-registered comparisons,
pooled out-of-fold per-class AP@50, furcation precision-recall curves, external-set summaries and training curves.
Per-image matches are cached once, so a bootstrap replicate is a concatenation; the AP is identical to evalap.ap_class
because greedy matching is done independently within each image.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from furcadino import evalap as E          # noqa: E402
from analyze import load_gt, to_np        # noqa: E402

FUR = 9
CLASSES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT", "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]
N_BOOT = 2000


def pool(files):
    d, g, ext, logs = {}, {}, [], []
    for f in sorted(files):
        r = json.load(open(f))
        if r["seed"] != 0:
            continue
        d.update(to_np(r["test"])); g.update(r["test_groups"])
        ext.append(to_np(r["external"])); logs.append(dict(fold=r["fold"], log=r.get("log") or []))
    return d, g, ext, logs


def per_image_matches(preds, gt, ids, cls=FUR, thr=0.5):
    out = {}
    for i in ids:
        S, T, n = E.match_class(preds, {i: gt[i]}, cls, thr, [i])
        out[i] = (S, T, n)
    return out


def ap_from_cache(cache, ids):
    S = np.concatenate([cache[i][0] for i in ids]) if ids else np.zeros(0)
    T = np.concatenate([cache[i][1] for i in ids]) if ids else np.zeros(0)
    n = int(sum(cache[i][2] for i in ids))
    return E.ap_from_matches(S, T.astype(float), n)


def pr_curve(cache, ids):
    S = np.concatenate([cache[i][0] for i in ids]); T = np.concatenate([cache[i][1] for i in ids]).astype(float)
    n = sum(cache[i][2] for i in ids)
    o = np.argsort(-S, kind="stable"); T = T[o]
    tp = np.cumsum(T); fp = np.cumsum(1 - T)
    rec = tp / n; prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    keep = np.unique(np.searchsorted(rec, np.linspace(0, rec[-1] if len(rec) else 0, 200)).clip(0, len(rec) - 1))
    return dict(recall=rec[keep].round(4).tolist(), precision=prec[keep].round(4).tolist())


def conformal(files, gt, iou):
    """Pre-specified sensitivity analysis: split-conformal lesion-recall threshold with lesion matching at `iou`
    (the primary analysis in analyze.py uses 0.3). Same per-fold calibration logic as analyze.py."""
    out = {}
    runs = [json.load(open(f)) for f in sorted(files)]
    for alpha in (0.1, 0.2):
        hit = n = fp = imgs = 0; infeasible = 0
        for r in runs:
            cal_g = {i: gt[i] for i in r["cal_groups"]}
            tau = E.conformal_threshold(E.lesion_scores(to_np(r["cal"]), cal_g, FUR, thr=iou), alpha)
            if tau is None:
                infeasible += 1
                continue
            tg = {i: gt[i] for i in r["test_groups"]}
            a = E.apply_threshold(to_np(r["test"]), tg, FUR, tau, thr=iou)
            hit += a["recall"] * a["n_lesions"]; n += a["n_lesions"]; fp += a["fp_per_image"] * len(tg); imgs += len(tg)
        out[str(alpha)] = dict(target_recall=1 - alpha, folds_feasible=len(runs) - infeasible,
                               achieved_recall=hit / n if n else None, fp_per_image=fp / imgs if imgs else None)
    return out


def main(rd, man, outp):
    gt = load_gt(man)
    V, FILES = {}, {}
    for d in sorted(glob.glob(os.path.join(rd, "*"))):
        v = os.path.basename(d)
        fs = glob.glob(os.path.join(d, "fold*_seed0.json"))
        if len(fs) == 5 and not v.endswith("overfit"):
            V[v] = pool(fs); FILES[v] = fs
    print("variants:", list(V))
    ids = sorted(set.intersection(*[set(V[v][1]) for v in V]))
    groups = V[next(iter(V))][1]
    gsub = {i: gt[i] for i in ids}
    cache = {v: per_image_matches(V[v][0], gt, ids) for v in V}

    out = dict(n_images=len(ids), n_groups=len({groups[i] for i in ids}), n_boot=N_BOOT, variants={}, paired={})
    for v, (p, g, ext, logs) in V.items():
        pc = {c: E.ap_class(p, gsub, k, 0.5, ids) for k, c in enumerate(CLASSES)}
        eids = sorted(ext[0]); eg = {i: gt[i] for i in eids}
        ef = [E.ap_class(e, eg, FUR, 0.5) for e in ext]
        efc = [{c: E.ap_class(e, eg, k, 0.5) for k, c in enumerate(CLASSES)} for e in ext]
        out["variants"][v] = dict(
            fur_ap50_check=ap_from_cache(cache[v], ids),
            per_class_ap50={c: (None if np.isnan(x) else x) for c, x in pc.items()},
            n_gt_per_class={c: int(sum((gsub[i]["labels"] == k).sum() for i in ids)) for k, c in enumerate(CLASSES)},
            pr_fur=pr_curve(cache[v], ids),
            external=dict(n_images=len(eids), n_fur=int(sum((eg[i]["labels"] == FUR).sum() for i in eids)),
                          fur_ap50_mean=float(np.mean(ef)), fur_ap50_min=float(np.min(ef)), fur_ap50_max=float(np.max(ef)),
                          per_class_ap50_mean={c: (None if all(np.isnan(f[c]) for f in efc) else float(np.nanmean([f[c] for f in efc])))
                                               for c in CLASSES}),
            train_logs=logs,
            conformal_iou05=conformal(FILES[v], gt, 0.5))
        print(v, "FUR AP50 (cache check)", round(out["variants"][v]["fur_ap50_check"], 4),
              "| conformal IoU0.5 feasible folds (90%, 80%):", out["variants"][v]["conformal_iou05"]["0.1"]["folds_feasible"],
              out["variants"][v]["conformal_iou05"]["0.2"]["folds_feasible"])

    best_bench = max([b for b in ("rtdetr-l", "yolo11m", "yolov8m") if b in V],
                     key=lambda b: out["variants"][b]["fur_ap50_check"], default=None)
    pairs = [("full", "plain"), ("full", best_bench), ("plain", best_bench)]
    pairs += [("full", a) for a in ("no_mirror", "no_coords", "no_context")]
    pairs += [(best_bench, b) for b in ("rtdetr-l", "yolo11m", "yolov8m") if b != best_bench]
    for a, b in pairs:
        if a not in V or b not in V:
            continue
        m, lo, hi, p = E.paired_bootstrap(lambda s: ap_from_cache(cache[a], s), lambda s: ap_from_cache(cache[b], s),
                                          ids, groups, n=N_BOOT, seed=0)
        # folds where A beats B (pre-registered criterion 3 uses >= 4 of 5)
        fa = per_fold(V[a], gt); fb = per_fold(V[b], gt)
        out["paired"][f"{a}__vs__{b}"] = dict(a=a, b=b, point=out["variants"][a]["fur_ap50_check"] - out["variants"][b]["fur_ap50_check"],
                                             boot_mean=m, ci=[lo, hi], p_a_gt_b=p,
                                             folds_a_better=int(sum(x > y for x, y in zip(fa, fb))),
                                             ext_diff=out["variants"][a]["external"]["fur_ap50_mean"] - out["variants"][b]["external"]["fur_ap50_mean"])
        print(f"{a} vs {b}: diff {out['paired'][f'{a}__vs__{b}']['point']:+.3f} CI {lo:+.3f}..{hi:+.3f} P(a>b)={p:.3f}")
    out["best_benchmark"] = best_bench
    json.dump(out, open(outp, "w"), indent=1)


def per_fold(vdata, gt):
    p, g, _, _ = vdata
    folds = {}
    # rebuild per-fold membership from the files' test groups is not stored per image; recompute from folds.json
    fj = json.load(open(os.path.join(os.path.dirname(HERE), "folds.json")))
    fold_of = {}
    for r in fj["pool"]:
        fold_of[r["path"]] = r["fold"]
    for i in g:
        folds.setdefault(fold_of[i], []).append(i)
    return [E.ap_class(p, {i: gt[i] for i in folds[k]}, FUR, 0.5) for k in sorted(folds)]


if __name__ == "__main__":
    main(*sys.argv[1:4])
