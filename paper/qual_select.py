"""Pre-specified selection of images for the qualitative figures (run once, BEFORE any qualitative output is viewed).
Rules (seed 0, numpy default_rng):
  main      4 pooled test images that contain >= 1 furcation box
  failures  for the best benchmark: the 6 highest-scoring false-positive furcation boxes and 6 random missed lesions
            at the pre-defined operating point (0.25 false-positive furcation boxes per image)
  midline   12 random pooled images
  external  4 random external images that contain >= 1 furcation box
  gt_example 1 random pooled image with >= 1 furcation box and >= 6 annotated classes
  duplicate  1 random near-duplicate pair whose members come from different published splits
Writes paper/qual_cases.json.   python paper/qual_select.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from furcadino import evalap as E      # noqa: E402
from analyze import load_gt, to_np    # noqa: E402

FUR = 9
rng = np.random.default_rng(0)
gt = load_gt(os.path.join(ROOT, "manifest.json"))
fj = json.load(open(os.path.join(ROOT, "folds.json")))
pool = {r["path"]: r for r in fj["pool"]}
ext = [e["path"] for e in fj["external"]]
has_fur = lambda p: bool((gt[p]["labels"] == FUR).any())

main = sorted(rng.choice(sorted(p for p in pool if has_fur(p)), 4, replace=False).tolist())
midline = sorted(rng.choice(sorted(pool), 12, replace=False).tolist())
external = sorted(rng.choice(sorted(p for p in ext if has_fur(p)), 4, replace=False).tolist())
rich = sorted(p for p in pool if has_fur(p) and len(set(gt[p]["labels"].tolist())) >= 6)
gt_example = str(rng.choice(rich))
groups = {}
for p, r in pool.items():
    groups.setdefault(r["group"], []).append(p)
cross = sorted(tuple(sorted(g)) for g in groups.values()
               if len(g) >= 2 and len({pool[p]["orig_split"] for p in g}) >= 2)
dup = list(cross[int(rng.integers(len(cross)))][:2])


# operating point and failure cases for the best benchmark
def operating_threshold(preds, ids, fppi=0.25):
    S, T, _ = E.match_class(preds, {i: gt[i] for i in ids}, FUR, 0.5, ids)
    o = np.argsort(-S, kind="stable"); S, T = S[o], T[o]
    fp = np.cumsum(1 - T) / len(ids)
    ok = np.where(fp <= fppi)[0]
    return float(S[ok[-1]]) if len(ok) else float(S[0])


stats = json.load(open(os.path.join(HERE, "stats_extra.json")))
bb = stats["best_benchmark"]
MODELS = ["rtdetr-l", "yolo11m", "yolov8m", "plain", "full", "no_mirror", "no_coords", "no_context"]
preds, thr = {}, {}
for v in MODELS:
    d = {}
    for k in range(5):
        r = json.load(open(os.path.join(ROOT, "res_all", v, f"fold{k}_seed0.json")))
        d.update(to_np(r["test"]))
    preds[v] = d
    thr[v] = operating_threshold(d, sorted(pool))
fp_boxes, fn_boxes = [], []
for i in sorted(pool):
    g = gt[i]; gb = g["boxes"][g["labels"] == FUR]
    p = preds[bb][i]; m = (p["labels"] == FUR) & (p["scores"] >= thr[bb])
    pb, ps = p["boxes"][m], p["scores"][m]
    iou = E.iou_matrix(pb, gb)
    for k in range(len(pb)):
        if not len(gb) or iou[k].max() < 0.5:
            fp_boxes.append((float(ps[k]), i, pb[k].tolist()))
    for j in range(len(gb)):
        if not len(pb) or iou[:, j].max() < 0.5:
            fn_boxes.append((i, gb[j].tolist()))
fp_top = sorted(fp_boxes, key=lambda x: -x[0])[:6]
fn_pick = [fn_boxes[k] for k in sorted(rng.choice(len(fn_boxes), 6, replace=False).tolist())]

out = dict(seed=0, rules=__doc__, main=main, midline=midline, external=external, gt_example=gt_example,
           duplicate_pair=dup, operating_fppi=0.25, thresholds=thr, best_benchmark=bb,
           failures=dict(false_positives=fp_top, missed=fn_pick, n_fp_total=len(fp_boxes), n_fn_total=len(fn_boxes)),
           fold_of={p: pool[p]["fold"] for p in set(main + midline + [gt_example] + dup + [x[1] for x in fp_top] + [x[0] for x in fn_pick])})
json.dump(out, open(os.path.join(HERE, "qual_cases.json"), "w"), indent=1)
print("main", main, "\nexternal", external, "\ngt_example", gt_example, "\ndup", dup, "\nthresholds", {k: round(v, 3) for k, v in thr.items()},
      "\nFP total", len(fp_boxes), "FN total", len(fn_boxes), "\nfolds needed", sorted({pool[p]['fold'] for p in main + midline}))
