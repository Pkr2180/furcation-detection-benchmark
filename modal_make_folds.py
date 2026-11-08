"""Stage 1: leakage-safe grouped folds. CPU only.
Pools train+valid+test of YOLO-Dental (the external clinic set stays untouched), clusters near-duplicate images
(perceptual hash Hamming <= 8 of 256 bits) into groups, and builds 5 group-disjoint folds stratified on furcation presence.
Writes /data/audit/folds.json to volume furcation-v4 and prints a summary.
Run: modal run modal_make_folds.py   (profile pradeepaiperio)
"""
import modal

app = modal.App("furcation-v4-folds")
vol = modal.Volume.from_name("furcation-v4", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4", "imagehash==4.3.1", "scikit-learn==1.4.2")

HAM = 8
FUR = 9


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def make():
    import json, collections
    import numpy as np, imagehash
    from sklearn.model_selection import StratifiedGroupKFold

    man = json.load(open("/data/audit/manifest.json"))
    ext = [m for m in man if m["path"].startswith("test_alte_cabinete")]
    pool = [m for m in man if not m["path"].startswith("test_alte_cabinete")]
    hp = [imagehash.hex_to_hash(m["phash"]) for m in pool]
    he = [imagehash.hex_to_hash(m["phash"]) for m in ext]

    parent = list(range(len(pool)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if hp[i] - hp[j] <= HAM:
                parent[find(i)] = find(j)
    groups = [find(i) for i in range(len(pool))]
    gid = {g: k for k, g in enumerate(sorted(set(groups)))}
    groups = [gid[g] for g in groups]
    gsize = collections.Counter(groups)

    # external images that have a near-duplicate in the pool are flagged (kept, reported)
    ext_dup = [bool(any(he[a] - hp[b] <= HAM for b in range(len(pool)))) for a in range(len(ext))]

    y = np.array([int(any(b[0] == FUR for b in m["boxes"])) for m in pool])
    nfur = np.array([sum(1 for b in m["boxes"] if b[0] == FUR) for m in pool])
    # stratify groups on has-furcation; weight is images
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    fold = np.full(len(pool), -1)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(pool)), y, groups)):
        fold[te] = k
    assert (fold >= 0).all()
    # group-disjointness check
    for g in set(groups):
        assert len({int(fold[i]) for i in range(len(pool)) if groups[i] == g}) == 1

    rows = [dict(path=m["path"], group=groups[i], fold=int(fold[i]), n_fur=int(nfur[i]), w=m["w"], h=m["h"],
                 orig_split=m["path"].split("/")[0]) for i, m in enumerate(pool)]
    erows = [dict(path=m["path"], n_fur=sum(1 for b in m["boxes"] if b[0] == FUR), dup_with_pool=ext_dup[i],
                  w=m["w"], h=m["h"]) for i, m in enumerate(ext)]
    summ = dict(n_pool=len(pool), n_groups=len(gsize), largest_groups=sorted(gsize.values())[-8:],
                n_singletons=sum(1 for v in gsize.values() if v == 1),
                per_fold={int(k): dict(images=int((fold == k).sum()), with_fur=int(y[fold == k].sum()),
                                       fur_boxes=int(nfur[fold == k].sum())) for k in range(5)},
                external=dict(images=len(ext), fur_boxes=int(sum(e["n_fur"] for e in erows)),
                              with_near_dup_in_pool=int(sum(ext_dup))))
    json.dump(dict(summary=summ, pool=rows, external=erows), open("/data/audit/folds.json", "w"))
    vol.commit()
    return summ


@app.local_entrypoint()
def main():
    import json
    s = make.remote()
    print(json.dumps(s, indent=1))
    open("folds_summary_local.json", "w").write(json.dumps(s, indent=1))
