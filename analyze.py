"""Pooled out-of-fold analysis of all variants. Usage: python analyze.py results_dir manifest.json out.json
Everything (AP, intervals, paired differences, conformal recall) is computed from the saved prediction files."""
import glob, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from furcadino import evalap as E

FUR = 9
H, W = 728, 1316


def load_gt(manifest):
    gt = {}
    for m in json.load(open(manifest)):
        b = np.array([[(c[1] - c[3] / 2) * W, (c[2] - c[4] / 2) * H, (c[1] + c[3] / 2) * W, (c[2] + c[4] / 2) * H]
                      for c in m["boxes"]], float).reshape(-1, 4)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        lab = np.array([c[0] for c in m["boxes"]], int)
        keep = ((b[:, 2] - b[:, 0]) >= 2) & ((b[:, 3] - b[:, 1]) >= 2)
        gt[m["path"]] = dict(boxes=b[keep], labels=lab[keep])
    return gt


def to_np(p):
    return {k: dict(boxes=np.array(v["boxes"], float).reshape(-1, 4), scores=np.array(v["scores"], float),
                    labels=np.array(v["labels"], int)) for k, v in p.items()}


def analyze_variant(files, gt):
    runs = [json.load(open(f)) for f in sorted(files)]
    seeds = sorted({r["seed"] for r in runs})
    out = dict(variant=runs[0]["variant"], n_runs=len(runs), seeds=seeds)
    res = {}
    for s in seeds:
        rs = [r for r in runs if r["seed"] == s]
        pooled, groups = {}, {}
        for r in rs:
            pooled.update(to_np(r["test"])); groups.update(r["test_groups"])
        ids = sorted(groups)
        g = {i: gt[i] for i in ids}
        fur = lambda idx: E.ap_class(pooled, g, FUR, 0.5, idx)
        point, lo, hi, _ = E.group_bootstrap(fur, ids, groups, n=1000, seed=s)
        res[s] = dict(
            fur_ap50=point, fur_ap50_ci=[lo, hi],
            fur_ap5095=E.ap_multi_iou(pooled, g, FUR, ids),
            map50_all=E.mean_ap50(pooled, g, range(14), ids),
            map50_other=E.mean_ap50(pooled, g, [c for c in range(14) if c != FUR], ids),
            fur_recall_fppi={str(f): E.recall_at_fppi(pooled, g, FUR, f, 0.5, ids) for f in (0.1, 0.25, 0.5)},
            per_fold_fur_ap50=[E.ap_class(to_np(r["test"]), {i: gt[i] for i in r["test_groups"]}, FUR, 0.5) for r in rs],
            n_images=len(ids), n_fur=int(sum((g[i]["labels"] == FUR).sum() for i in ids)))
        # external: each fold model scored on the external set, then mean over folds
        ext = [to_np(r["external"]) for r in rs]
        eids = sorted(ext[0])
        eg = {i: gt[i] for i in eids}
        res[s]["external_fur_ap50_per_fold"] = [E.ap_class(p, eg, FUR, 0.5) for p in ext]
        res[s]["external_map50_other_per_fold"] = [E.mean_ap50(p, eg, [c for c in range(14) if c != FUR]) for p in ext]
        # conformal lesion-recall control, per fold calibration, pooled test result
        cf = {}
        for alpha in (0.1, 0.2):
            hit = n = fp = 0; imgs = 0; taus = []; infeasible = 0
            for r in rs:
                cal_g = {i: gt[i] for i in r["cal_groups"]}
                tau = E.conformal_threshold(E.lesion_scores(to_np(r["cal"]), cal_g, FUR), alpha)
                if tau is None:
                    infeasible += 1
                    continue
                taus.append(tau)
                tg = {i: gt[i] for i in r["test_groups"]}
                a = E.apply_threshold(to_np(r["test"]), tg, FUR, tau)
                hit += a["recall"] * a["n_lesions"]; n += a["n_lesions"]; fp += a["fp_per_image"] * len(tg); imgs += len(tg)
            cf[str(alpha)] = dict(target_recall=1 - alpha, achieved_recall=hit / max(n, 1), fp_per_image=fp / max(imgs, 1),
                                  n_lesions=int(n), mean_tau=float(np.mean(taus)) if taus else None, folds_infeasible=infeasible, folds_feasible=len(taus))
        res[s]["conformal"] = cf
    out["by_seed"] = {str(k): v for k, v in res.items()}
    out["_pooled_cache"] = None
    return out


def paired(files_a, files_b, gt, seed=0, n=1000):
    def pool(files):
        d, g = {}, {}
        for f in files:
            r = json.load(open(f))
            if r["seed"] != seed:
                continue
            d.update(to_np(r["test"])); g.update(r["test_groups"])
        return d, g
    pa, ga = pool(files_a); pb, gb = pool(files_b)
    ids = sorted(set(ga) & set(gb)); gg = {i: gt[i] for i in ids}
    return E.paired_bootstrap(lambda s: E.ap_class(pa, gg, FUR, 0.5, s), lambda s: E.ap_class(pb, gg, FUR, 0.5, s),
                              ids, ga, n=n)


if __name__ == "__main__":
    rd, man, outp = sys.argv[1:4]
    gt = load_gt(man)
    allres, files = {}, {}
    for d in sorted(glob.glob(os.path.join(rd, "*"))):
        v = os.path.basename(d)
        fs = glob.glob(os.path.join(d, "fold*_seed*.json"))
        if not fs:
            continue
        files[v] = fs
        allres[v] = analyze_variant(fs, gt)
        b = allres[v]["by_seed"]
        s0 = sorted(b)[0]
        print(f"{v:14s} runs={allres[v]['n_runs']} FUR AP50={b[s0]['fur_ap50']:.3f} CI={b[s0]['fur_ap50_ci'][0]:.3f}-{b[s0]['fur_ap50_ci'][1]:.3f} "
              f"mAP50other={b[s0]['map50_other']:.3f} ext={np.mean(b[s0]['external_fur_ap50_per_fold']):.3f}")
    json.dump(allres, open(outp, "w"), indent=1)
