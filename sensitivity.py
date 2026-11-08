"""Sensitivity analyses on saved predictions. Usage: python sensitivity.py results_dir manifest.json folds.json out.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from furcadino import evalap as E
from analyze import load_gt, to_np

FUR = 9


def pooled(files, seed=0):
    d, g = {}, {}
    for f in files:
        r = json.load(open(f))
        if r["seed"] != seed:
            continue
        d.update(to_np(r["test"])); g.update(r["test_groups"])
    return d, g


if __name__ == "__main__":
    rd, man, folds, outp = sys.argv[1:5]
    gt = load_gt(man)
    fj = json.load(open(folds))
    gsize = {}
    for r in fj["pool"]:
        gsize[r["group"]] = gsize.get(r["group"], 0) + 1
    dup_img = {r["path"] for r in fj["pool"] if gsize[r["group"]] > 1}
    area = {i: (g["boxes"][g["labels"] == FUR][:, 2] - g["boxes"][g["labels"] == FUR][:, 0]) *
               (g["boxes"][g["labels"] == FUR][:, 3] - g["boxes"][g["labels"] == FUR][:, 1]) for i, g in gt.items()}
    allfur = np.concatenate([a for a in area.values() if len(a)])
    t1, t2 = np.percentile(allfur, [33.3, 66.7])
    out = {}
    for d in sorted(glob.glob(os.path.join(rd, "*"))):
        v = os.path.basename(d)
        fs = glob.glob(os.path.join(d, "fold*_seed0.json"))
        if len(fs) < 5 or v.endswith("overfit"):
            continue
        p, grp = pooled(fs)
        ids = sorted(grp)
        g = {i: gt[i] for i in ids}
        res = {}
        for thr in (0.3, 0.5, 0.75):
            pt, lo, hi, _ = E.group_bootstrap(lambda s: E.ap_class(p, g, FUR, thr, s), ids, grp, n=500, seed=1)
            res[f"fur_ap_iou{thr}"] = [pt, lo, hi]
        keep = [i for i in ids if i not in dup_img]
        res["fur_ap50_excluding_dup_images"] = dict(n_images=len(keep), ap=E.ap_class(p, g, FUR, 0.5, keep))
        # image-level bootstrap (each image its own group)
        pt, lo, hi, _ = E.group_bootstrap(lambda s: E.ap_class(p, g, FUR, 0.5, s), ids, {i: i for i in ids}, n=500, seed=2)
        res["fur_ap50_image_level_ci"] = [pt, lo, hi]
        # box-size tertiles: keep only lesions in the tertile, ignore predictions via GT filtering
        for name, lo_a, hi_a in (("small", 0, t1), ("medium", t1, t2), ("large", t2, 1e12)):
            gg = {}
            for i in ids:
                b = g[i]["boxes"]; l = g[i]["labels"]
                a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                m = (l != FUR) | ((a >= lo_a) & (a < hi_a))
                gg[i] = dict(boxes=b[m], labels=l[m])
            res[f"fur_ap50_{name}_lesions"] = E.ap_class(p, gg, FUR, 0.5, ids)
        out[v] = res
        print(v, {k: (round(x[0], 3) if isinstance(x, list) else x) for k, x in res.items() if k.startswith("fur_ap_iou")})
    json.dump(out, open(outp, "w"), indent=1)
