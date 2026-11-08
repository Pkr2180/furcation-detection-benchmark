"""Qualitative figures (radiographs, ground truth, predictions, score maps, midline, preprocessing, leakage, failures).
All cases come from paper/qual_cases.json (pre-specified random selection, qual_select.py); every prediction shown is
out-of-fold. Inputs: manifest.json, folds.json, res_all/, paper/qual_images/{resized,raw,explain}.
  python paper/make_qual_figs.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from furcadino import evalap as E      # noqa: E402
from analyze import load_gt, to_np    # noqa: E402

FIG = os.path.join(HERE, "figures"); QI = os.path.join(HERE, "qual_images")
Q = json.load(open(os.path.join(HERE, "qual_cases.json")))
GT = load_gt(os.path.join(ROOT, "manifest.json"))
MAN = {m["path"]: m for m in json.load(open(os.path.join(ROOT, "manifest.json")))}
FJ = json.load(open(os.path.join(ROOT, "folds.json")))
POOL = {r["path"]: r for r in FJ["pool"]}
FUR, H, W = 9, 728, 1316
CLASSES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT", "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]
CLASS_FULL = ["Implant", "Prosthetic restoration", "Obturation", "Endodontic treatment", "Caries", "Bone loss", "Impacted tooth",
              "Apical periodontitis", "Root fragment", "Furcation lesion", "Apical surgery", "Root resorption",
              "Orthodontic appliance", "Surgical device"]
NAME = {"rtdetr-l": "RT-DETR-L", "yolo11m": "YOLO11m", "full": "BiFurca-DINO", "plain": "DINOv2-FCOS (plain)"}
C_TP, C_FP, C_FN, C_GT = "#38a169", "#e53e3e", "#ecc94b", "#00b5d8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def save(fig, name):
    for ext in ("png", "tif"):
        kw = dict(dpi=300, bbox_inches="tight", facecolor="white")
        if ext == "tif":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print("saved", name)


def img(p):
    return np.asarray(Image.open(os.path.join(QI, "resized", p.replace("/", "__").rsplit(".", 1)[0] + ".jpg")).convert("L"))


def heat(v, p):
    return np.load(os.path.join(QI, "explain", v, p.replace("/", "__").rsplit(".", 1)[0] + ".npz"))


_cache = {}


def preds(v, p, external=False, fold=None):
    k = POOL[p]["fold"] if not external else (0 if fold is None else fold)
    key = (v, k)
    if key not in _cache:
        _cache[key] = json.load(open(os.path.join(ROOT, "res_all", v, f"fold{k}_seed0.json")))
    sect = _cache[key]["external" if external else "test"]
    return to_np({p: sect[p]})[p]


def classify(pr, p, thr):
    """TP / FP furcation predictions at score >= thr and missed GT lesions (IoU 0.5, greedy by score)."""
    gb = GT[p]["boxes"][GT[p]["labels"] == FUR]
    m = (pr["labels"] == FUR) & (pr["scores"] >= thr)
    pb, ps = pr["boxes"][m], pr["scores"][m]
    o = np.argsort(-ps); pb, ps = pb[o], ps[o]
    used = np.zeros(len(gb), bool); tp, fp = [], []
    iou = E.iou_matrix(pb, gb)
    for k in range(len(pb)):
        j = int(np.argmax(np.where(used, -1, iou[k]))) if len(gb) else -1
        if j >= 0 and iou[k, j] >= 0.5 and not used[j]:
            used[j] = True; tp.append((pb[k], ps[k]))
        else:
            fp.append((pb[k], ps[k]))
    return tp, fp, [gb[j] for j in range(len(gb)) if not used[j]]


def box(ax, b, color, lw=1.4, ls="-", label=None, fs=6, below=False):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw, ls=ls))
    if label:
        t = ax.text(b[0], b[3] + 3 if below else b[1] - 3, label, color=color, fontsize=fs, va="top" if below else "bottom",
                    fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.55, pad=0.4), clip_on=True)


def crop_for(p, pad=70, aspect=1.5, min_w=420):
    """Crop window: union of all GT furcation boxes, padded, fixed 3:2 aspect (rule-based, not chosen by eye)."""
    gb = GT[p]["boxes"][GT[p]["labels"] == FUR]
    x0, y0, x1, y1 = gb[:, 0].min() - pad, gb[:, 1].min() - pad, gb[:, 2].max() + pad, gb[:, 3].max() + pad
    w = max(x1 - x0, (y1 - y0) * aspect, min_w); h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = np.clip(cx - w / 2, 0, W - w), np.clip(cy - h / 2, 0, H - h)
    return x0, y0, x0 + w, y0 + h


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================ Figure 2: architecture
def fig_architecture():
    fig = plt.figure(figsize=(7.4, 4.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def blk(x, y, w, h, t, fc, ec="#2d3748", fs=6.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, c="#2d3748", ls="-"):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=c, ls=ls, mutation_scale=8))

    p = Q["main"][0]
    iax = fig.add_axes([0.005, 0.56, 0.17, 0.2]); iax.imshow(img(p), cmap="gray"); off(iax)
    iax.set_title("Panoramic radiograph\n728 × 1316 px", fontsize=6.5)
    blk(20, 37, 15, 13, "DINOv2-base\n(frozen, ViT-B/14)\nlast 4 layers,\nlearned mix", "#e2e8f0", bold=False)
    arr(17.5, 43.5, 20, 43.5)
    blk(39, 37, 12, 13, "1×1 projection\n768 → 256\nstride 14", "#edf2f7")
    arr(35.5, 43.5, 39, 43.5)
    # three optional modules
    blk(55, 47, 20, 10, "(a) Midline-mirror module\nlabel-free midline (soft-argmax of\nleft–right similarity) + contralateral\ncross-attention (7×7 window)", "#fed7d7", fs=5.9)
    blk(55, 35.5, 20, 9.5, "(b) Arch-canonical coordinates\nsigned distance to midline,\nheight; Fourier-encoded", "#fefcbf", fs=5.9)
    blk(55, 24, 20, 9.5, "(c) Coarse context transformer\n2 layers over stride-28 map\n(evidence from other findings)", "#c6f6d5", fs=5.9)
    arr(51.5, 44, 55, 51); arr(51.5, 43.5, 55, 40); arr(51.5, 43, 55, 29)
    blk(79, 33, 19, 15, "3-level pyramid\nstrides 7 / 14 / 28\n\nFCOS head\n(softplus box regression,\ncentre-ness, 14 classes)", "#e9d8fd", fs=6.2)
    arr(75.5, 52, 79, 44); arr(75.5, 40, 79, 40.5); arr(75.5, 29, 79, 37)
    blk(79, 14, 19, 12, "Output\nboxes + scores for\n14 findings incl.\nfurcation lesion", "#e2e8f0", fs=6.2)
    arr(88.5, 33, 88.5, 26.5)
    ax.text(20, 31.5, "Frozen backbone: only the projection, modules (a)–(c),\npyramid and head are trained (20 epochs).",
            fontsize=6.2, color="#4a5568")
    ax.text(20, 26.5, "Variants: full = (a)+(b)+(c); plain = none;\nw/o mirror / coords / context = one removed.",
            fontsize=6.2, color="#4a5568")
    # midline example from the model itself
    z = heat("full", p)
    iax2 = fig.add_axes([0.01, 0.02, 0.30, 0.27]); iax2.imshow(img(p), cmap="gray")
    iax2.axvline(float(z["midline_px"]), color="#e53e3e", lw=1.2)
    off(iax2); iax2.set_title("Midline estimated by module (a), no labels\n(out-of-fold model; red line)", fontsize=6.3)
    sax = fig.add_axes([0.37, 0.06, 0.25, 0.17])
    sax.set_title("Soft-argmax over candidate columns (red = estimate)", fontsize=6)
    sax.plot(z["cand_px"], z["sim"], color="#2d3748", lw=1)
    sax.axvline(float(z["midline_px"]), color="#e53e3e", lw=1)
    sax.set_xlabel("Candidate midline column (px)", fontsize=6); sax.set_ylabel("Left–right\nsimilarity", fontsize=6)
    sax.tick_params(labelsize=5.5)
    for s in ("top", "right"):
        sax.spines[s].set_visible(False)
    save(fig, "Figure2_BiFurca_architecture")


# ============================================================ Figure 3: ground truth
def fig_ground_truth():
    p = Q["gt_example"]
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_axes([0.0, 0.42, 0.72, 0.56]); ax.imshow(img(p), cmap="gray"); off(ax)
    cmap = plt.get_cmap("tab20")
    present = sorted(set(GT[p]["labels"].tolist()))
    for b, l in zip(GT[p]["boxes"], GT[p]["labels"]):
        box(ax, b, "#00e5ff" if l == FUR else cmap(l), lw=2.2 if l == FUR else 0.9)
    ax.set_title(f"A  Ground-truth annotation (all findings; furcation lesions in cyan, thick)", fontsize=8, loc="left")
    fb = GT[p]["boxes"][GT[p]["labels"] == FUR][0]
    cx_, cy_ = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
    zax = fig.add_axes([0.745, 0.47, 0.22, 0.17]); zax.imshow(img(p), cmap="gray")
    zax.set_xlim(cx_ - 90, cx_ + 90); zax.set_ylim(cy_ + 60, cy_ - 60); off(zax)
    for b, l in zip(GT[p]["boxes"], GT[p]["labels"]):
        box(zax, b, "#00e5ff" if l == FUR else cmap(l), lw=2 if l == FUR else 0.8)
    zax.set_title("Zoom on the furcation box", fontsize=6.5)
    for s in zax.spines.values():
        s.set_visible(True); s.set_color("#00e5ff")
    lax = fig.add_axes([0.73, 0.45, 0.26, 0.5]); lax.axis("off")
    for k, l in enumerate(present):
        lax.add_patch(Rectangle((0, 1 - (k + 1) * 0.075), 0.08, 0.05, fc="none", ec="#00e5ff" if l == FUR else cmap(l), lw=2 if l == FUR else 1.2,
                                transform=lax.transAxes))
        lax.text(0.12, 1 - (k + 1) * 0.075 + 0.025, f"{CLASS_FULL[l]} ({CLASSES[l]})", va="center", fontsize=6.5, transform=lax.transAxes)
    # density of furcation centres over the pooled set
    cx, cy, ww, hh = [], [], [], []
    for q, g in GT.items():
        if q not in POOL:
            continue
        b = g["boxes"][g["labels"] == FUR]
        cx += ((b[:, 0] + b[:, 2]) / 2).tolist(); cy += ((b[:, 1] + b[:, 3]) / 2).tolist()
        ww += (b[:, 2] - b[:, 0]).tolist(); hh += (b[:, 3] - b[:, 1]).tolist()
    bx = fig.add_axes([0.05, 0.05, 0.5, 0.32])
    hb = bx.hexbin(cx, cy, gridsize=(30, 12), extent=(0, W, 0, H), cmap="magma", mincnt=1)
    bx.set_xlim(0, W); bx.set_ylim(H, 0); bx.set_aspect("equal")
    bx.axvline(W / 2, color="#a0aec0", lw=0.8, ls="--")
    bx.set_title(f"B  Location of all {len(cx)} furcation boxes (pooled set)", fontsize=8, loc="left")
    bx.set_xlabel("x (px, model input)", fontsize=6.5); bx.set_ylabel("y (px)", fontsize=6.5); bx.tick_params(labelsize=6)
    fig.colorbar(hb, ax=bx, fraction=0.025, pad=0.01).set_label("boxes", fontsize=6)
    sx = fig.add_axes([0.69, 0.07, 0.29, 0.28])
    sx.scatter(ww, hh, s=5, alpha=0.5, color="#2b6cb0")
    for s_, lab in ((7, "FCOS stride 7"), (14, "DINOv2 patch 14 px")):
        sx.axvline(s_, color="#e53e3e" if s_ == 14 else "#dd6b20", lw=0.8, ls="--")
        sx.axhline(s_, color="#e53e3e" if s_ == 14 else "#dd6b20", lw=0.8, ls="--")
        sx.text(s_ + 1, max(hh) * 0.95, lab, fontsize=5.5, rotation=90, va="top", color="#e53e3e" if s_ == 14 else "#dd6b20")
    sx.set_xlabel("Box width (px)", fontsize=6.5); sx.set_ylabel("Box height (px)", fontsize=6.5); sx.tick_params(labelsize=6)
    sx.set_title(f"C  Furcation box size (median {np.median(ww):.0f} × {np.median(hh):.0f} px)", fontsize=8, loc="left")
    for s in ("top", "right"):
        sx.spines[s].set_visible(False)
    save(fig, "Figure3_ground_truth")
    return dict(n=len(cx), med_w=float(np.median(ww)), med_h=float(np.median(hh)))


# ============================================================ Figure: qualitative predictions
def fig_predictions():
    models = ["rtdetr-l", "yolo11m", "full"]
    cases = Q["main"]
    fig, axs = plt.subplots(len(cases), 5, figsize=(7.4, 1.08 * len(cases) + 0.45),
                            gridspec_kw=dict(width_ratios=[1.21, 1, 1, 1, 1], wspace=0.03, hspace=0.06))
    counts = {}
    for r, p in enumerate(cases):
        I = img(p); c = crop_for(p)
        a = axs[r, 0]; a.imshow(I, cmap="gray"); off(a)
        a.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fill=False, ec="white", lw=1, ls="--"))
        a.text(20, 90, f"test fold {POOL[p]['fold']}", color="white", fontsize=6, fontweight="bold",
               bbox=dict(fc="black", ec="none", alpha=0.5, pad=0.6))
        cols = [("Ground truth", None)] + [(NAME[m], m) for m in models]
        for j, (title, m) in enumerate(cols, start=1):
            a = axs[r, j]; a.imshow(I, cmap="gray"); a.set_xlim(c[0], c[2]); a.set_ylim(c[3], c[1]); off(a)
            if m is None:
                for b in GT[p]["boxes"][GT[p]["labels"] == FUR]:
                    box(a, b, C_GT, lw=1.6)
            else:
                tp, fp, fn = classify(preds(m, p), p, Q["thresholds"][m])
                for b, s in tp:
                    box(a, b, C_TP, label=f"{s:.2f}")
                for b, s in fp:
                    box(a, b, C_FP, label=f"{s:.2f}", below=True)
                for b in fn:
                    box(a, b, C_FN, ls="--", lw=1.4)
                counts.setdefault(m, [0, 0, 0])
                counts[m][0] += len(tp); counts[m][1] += len(fp); counts[m][2] += len(fn)
            if r == 0:
                a.set_title(title, fontsize=7.5)
        if r == 0:
            axs[r, 0].set_title("Radiograph (crop dashed)", fontsize=7.5)
    h = [Rectangle((0, 0), 1, 1, fc="none", ec=c_, lw=1.5, ls=ls) for c_, ls in ((C_GT, "-"), (C_TP, "-"), (C_FP, "-"), (C_FN, "--"))]
    fig.legend(h, ["ground-truth furcation", "true positive (score)", "false positive (score)", "missed lesion"],
               loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    save(fig, "Figure5_qualitative_predictions")
    return counts


# ============================================================ Figure: score maps
def fig_score_maps(cases, name, external=False):
    fig, axs = plt.subplots(len(cases), 3, figsize=(7.4, 1.45 * len(cases) + 0.6), gridspec_kw=dict(wspace=0.03, hspace=0.1))
    vmax = max(float(heat(v, p)["heat"].max()) for p in cases for v in ("full", "plain"))
    for r, p in enumerate(cases):
        I = img(p); zf, zp = heat("full", p), heat("plain", p)
        gb = GT[p]["boxes"][GT[p]["labels"] == FUR]
        for j, (title, z) in enumerate((("Ground truth + midline (red)", None), ("BiFurca-DINO score map", zf),
                                         ("Plain DINOv2-FCOS score map", zp))):
            a = axs[r, j]; a.imshow(I, cmap="gray"); off(a)
            if z is None:
                a.axvline(float(zf["midline_px"]), color="#e53e3e", lw=1)
            else:
                hm = np.ma.masked_less(z["heat"].astype(np.float32), 0.05)   # transparent where score < 0.05
                im = a.imshow(hm, cmap="inferno", alpha=0.75, vmin=0, vmax=vmax)
            for b in gb:
                box(a, b, C_GT, lw=1.1)
            if r == 0:
                a.set_title(title, fontsize=7.5)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("FCOS furcation score  √(p_class × p_centre-ness)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, name)


# ============================================================ Supplementary: preprocessing pre/post
def fig_preprocessing():
    p = Q["gt_example"]
    raw = np.asarray(Image.open(os.path.join(QI, "raw", p)).convert("L"))
    rh, rw = raw.shape
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 1.9), gridspec_kw=dict(wspace=0.05))
    a = axs[0]; a.imshow(raw, cmap="gray"); off(a)
    for c in MAN[p]["boxes"]:
        if c[0] == FUR:
            b = [(c[1] - c[3] / 2) * rw, (c[2] - c[4] / 2) * rh, (c[1] + c[3] / 2) * rw, (c[2] + c[4] / 2) * rh]
            box(a, b, C_GT, lw=1.2)
    a.set_title(f"A  Original image\n({rw} × {rh} px), YOLO labels", fontsize=7)
    I = img(p)
    a = axs[1]; a.imshow(I, cmap="gray"); off(a)
    for b in GT[p]["boxes"][GT[p]["labels"] == FUR]:
        box(a, b, C_GT, lw=1.2)
    a.set_title(f"B  Model input\n({W} × {H} px), boxes rescaled", fontsize=7)
    x = I.astype(np.float32) / 255
    a = axs[2]; im = a.imshow((x - 0.456) / 0.224, cmap="coolwarm", vmin=-2.5, vmax=2.5); off(a)
    a.set_title("C  ImageNet-normalised input\n(z-score, green channel)", fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.03, pad=0.01).ax.tick_params(labelsize=5.5)
    save(fig, "FigureS5_preprocessing")


# ============================================================ Figure 3: furcation ground truth before / after preprocessing
def fig_gt_prepost():
    """Original radiograph + raw YOLO label line -> model input with rescaled box -> binary furcation mask.
    Verified numerically (raises on failure): label file == manifest; original-pixel boxes rescaled to the model input
    equal the boxes used in the analysis; mask area == union of boxes; stored model input == resize of the original."""
    cases = Q["main"]
    checks = dict(label_vs_manifest_max_abs=0.0, rescaled_vs_analysis_max_px=0.0, mask_vs_box_area_max_px=0, resize_mean_absdiff=[])
    fig, axs = plt.subplots(len(cases), 5, figsize=(7.4, 1.18 * len(cases) + 0.55),
                            gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1, 1.6], wspace=0.04, hspace=0.1))
    for r, p in enumerate(cases):
        rawim = Image.open(os.path.join(QI, "raw", p)).convert("L"); raw = np.asarray(rawim)
        rh, rw = raw.shape
        lab_path = os.path.join(QI, "raw", p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        lines = [l.split() for l in open(lab_path).read().strip().splitlines() if l.strip()]
        fur_lines = [l for l in lines if int(l[0]) == FUR]
        man_fur = [c for c in MAN[p]["boxes"] if c[0] == FUR]
        assert len(fur_lines) == len(man_fur) > 0, (p, len(fur_lines), len(man_fur))
        for l, c in zip(fur_lines, man_fur):
            d = max(abs(float(a) - b) for a, b in zip(l[1:5], c[1:5]))
            checks["label_vs_manifest_max_abs"] = max(checks["label_vs_manifest_max_abs"], d)
            assert d < 1e-6, (p, l, c)
        raw_boxes = np.array([[(float(l[1]) - float(l[3]) / 2) * rw, (float(l[2]) - float(l[4]) / 2) * rh,
                               (float(l[1]) + float(l[3]) / 2) * rw, (float(l[2]) + float(l[4]) / 2) * rh] for l in fur_lines])
        resc = raw_boxes * np.array([W / rw, H / rh, W / rw, H / rh])
        ana = GT[p]["boxes"][GT[p]["labels"] == FUR]
        dpx = float(np.abs(np.sort(resc, 0) - np.sort(ana, 0)).max())
        checks["rescaled_vs_analysis_max_px"] = max(checks["rescaled_vs_analysis_max_px"], dpx)
        assert dpx < 1.0, (p, resc, ana)
        I = img(p)
        mask = np.zeros((H, W), bool)
        for b in ana:
            mask[int(round(b[1])):int(round(b[3])), int(round(b[0])):int(round(b[2]))] = True
        exp = sum((int(round(b[3])) - int(round(b[1]))) * (int(round(b[2])) - int(round(b[0]))) for b in ana)
        checks["mask_vs_box_area_max_px"] = max(checks["mask_vs_box_area_max_px"], abs(int(mask.sum()) - exp) if len(ana) == 1 else 0)
        local = np.asarray(rawim.resize((W, H), Image.BILINEAR)).astype(np.float32)
        checks["resize_mean_absdiff"].append(float(np.abs(local - I.astype(np.float32)).mean()))
        # crops: same anatomical window in both resolutions
        c = crop_for(p, pad=55, min_w=300)
        cr = (c[0] * rw / W, c[1] * rh / H, c[2] * rw / W, c[3] * rh / H)
        a = axs[r, 0]; a.imshow(raw, cmap="gray"); off(a)
        for b in raw_boxes:
            box(a, b, C_GT, lw=1.2)
        a.add_patch(Rectangle((cr[0], cr[1]), cr[2] - cr[0], cr[3] - cr[1], fill=False, ec="white", lw=0.8, ls="--"))
        a.text(25, rh * 0.09, f"{rw} × {rh} px", color="white", fontsize=5.5, bbox=dict(fc="black", ec="none", alpha=0.5, pad=0.5))
        a = axs[r, 1]; a.imshow(raw, cmap="gray"); a.set_xlim(cr[0], cr[2]); a.set_ylim(cr[3], cr[1]); off(a)
        for b in raw_boxes:
            box(a, b, C_GT, lw=1.4)
        l0 = fur_lines[0]
        txt = f"class {l0[0]} (FUR)\ncx {float(l0[1]):.4f}  cy {float(l0[2]):.4f}\nw  {float(l0[3]):.4f}  h  {float(l0[4]):.4f}"
        a.text(cr[0] + (cr[2] - cr[0]) * 0.03, cr[3] - (cr[3] - cr[1]) * 0.04, txt, color="#00e5ff", fontsize=4.4, family="monospace",
               va="bottom", bbox=dict(fc="black", ec="none", alpha=0.7, pad=0.6), clip_on=True)
        a = axs[r, 2]; a.imshow(I, cmap="gray"); a.set_xlim(c[0], c[2]); a.set_ylim(c[3], c[1]); off(a)
        for b in ana:
            box(a, b, C_GT, lw=1.4, label=f"{b[2] - b[0]:.0f}×{b[3] - b[1]:.0f} px", fs=5)
        a = axs[r, 3]; a.imshow(mask, cmap="gray", vmin=0, vmax=1); a.set_xlim(c[0], c[2]); a.set_ylim(c[3], c[1]); off(a)
        for s in a.spines.values():
            s.set_visible(True); s.set_color("#a0aec0"); s.set_linewidth(0.5)
        a = axs[r, 4]; a.imshow(I, cmap="gray"); off(a)
        a.imshow(np.ma.masked_where(~mask, mask.astype(float)), cmap="cool", alpha=0.9, vmin=0, vmax=1)
        a.text(25, H * 0.09, f"{W} × {H} px", color="white", fontsize=5.5, bbox=dict(fc="black", ec="none", alpha=0.5, pad=0.5))
        axs[r, 0].text(-0.02, 0.5, p.split("/")[0] + "/" + p.split("/")[-1], transform=axs[r, 0].transAxes, rotation=90,
                       va="center", ha="right", fontsize=5.5)
    for j, t in enumerate(["A  Original\n(pre-processing)", "B  Original, zoom\n+ YOLO label line",
                           "C  Model input, zoom\n(post-processing)", "D  Furcation mask\n(model input, zoom)",
                           "E  Mask overlay\n(model input)"]):
        axs[0, j].set_title(t, fontsize=6.4)
    assert max(checks["resize_mean_absdiff"]) < 4.0, checks
    save(fig, "Figure3_furcation_gt_pre_post")
    return checks


# ============================================================ Supplementary: leakage pair
def fig_duplicate():
    a_, b_ = Q["duplicate_pair"]
    A = np.asarray(Image.open(os.path.join(QI, "raw", a_)).convert("L")).astype(np.float32)
    B = np.asarray(Image.open(os.path.join(QI, "raw", b_)).convert("L").resize(A.shape[::-1])).astype(np.float32)
    D = np.abs(A - B)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 1.9), gridspec_kw=dict(wspace=0.05))
    for a, X, t in ((axs[0], A, f"A  {a_}\n(published training split)"), (axs[1], B, f"B  {b_}\n(published validation split)")):
        a.imshow(X, cmap="gray"); off(a); a.set_title(t, fontsize=6.8)
    im = axs[2].imshow(D, cmap="magma", vmin=0, vmax=60); off(axs[2])
    axs[2].set_title(f"C  |A − B| pixel difference\n(mean {D.mean():.2f} grey levels; max {D.max():.0f})", fontsize=6.8)
    fig.colorbar(im, ax=axs[2], fraction=0.03, pad=0.01).ax.tick_params(labelsize=5.5)
    save(fig, "FigureS6_cross_split_duplicate")
    return float(D.mean())


# ============================================================ Supplementary: failure gallery
def fig_failures():
    bb = Q["best_benchmark"]
    fps, fns = Q["failures"]["false_positives"], Q["failures"]["missed"]
    fig, axs = plt.subplots(2, 6, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=0.04, hspace=0.25))

    def win(b, s=150):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0, y0 = np.clip(cx - s, 0, W - 2 * s), np.clip(cy - s * 0.66, 0, H - 1.33 * s)
        return x0, y0, x0 + 2 * s, y0 + 1.33 * s

    for k, (score, p, b) in enumerate(fps):
        a = axs[0, k]; a.imshow(img(p), cmap="gray"); c = win(b); a.set_xlim(c[0], c[2]); a.set_ylim(c[3], c[1]); off(a)
        box(a, b, C_FP, label=f"{score:.2f}")
        for g in GT[p]["boxes"][GT[p]["labels"] == FUR]:
            box(a, g, C_GT, lw=1)
    for k, (p, b) in enumerate(fns):
        a = axs[1, k]; a.imshow(img(p), cmap="gray"); c = win(b); a.set_xlim(c[0], c[2]); a.set_ylim(c[3], c[1]); off(a)
        box(a, b, C_FN, ls="--")
    axs[0, 0].set_title(f"A  Six highest-scoring false positives of {NAME.get(bb, bb)} (red; cyan = ground truth)", fontsize=7, loc="left")
    axs[1, 0].set_title(f"B  Six randomly selected missed lesions (yellow, dashed) at 0.25 FP/image", fontsize=7, loc="left")
    save(fig, "FigureS7_failure_gallery")
    # how many of the shown false positives overlap an annotated furcation (IoU > 0 but < 0.5): localisation, not anatomy
    return sum(1 for _, p, b in fps if len(GT[p]["boxes"][GT[p]["labels"] == FUR]) and
               E.iou_matrix([b], GT[p]["boxes"][GT[p]["labels"] == FUR]).max() > 0)


# ============================================================ Supplementary: midline sanity
def fig_midline():
    cases = Q["midline"]
    fig, axs = plt.subplots(3, 4, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.03, hspace=0.06))
    pos = []
    for a, p in zip(axs.ravel(), cases):
        z = heat("full", p); a.imshow(img(p), cmap="gray"); off(a)
        a.axvline(float(z["midline_px"]), color="#e53e3e", lw=0.9); pos.append(float(z["midline_px"]))
    fig.suptitle(f"Label-free midline (red) on 12 random radiographs; estimate {min(pos):.0f}–{max(pos):.0f} px "
                 f"(image centre {W / 2:.0f} px)", fontsize=7.5, y=0.99)
    save(fig, "FigureS9_midline_estimates")
    return dict(min=min(pos), max=max(pos))


def map_stats():
    """Quantify the score maps instead of describing them by eye (all images with saved maps):
    Pearson r between BiFurca and plain maps; share of score mass (score >= 0.05) inside annotated furcation boxes,
    and the area share of those boxes for reference; peak vs top saved furcation score where that box was retained."""
    names = sorted(os.listdir(os.path.join(QI, "explain", "full")))
    r_list, inside, area, peak = [], {"full": [], "plain": []}, [], {"full": [], "plain": []}
    for nm in names:
        p = nm[:-4].replace("__", "/")
        p = next(q for q in list(GT) if q.rsplit(".", 1)[0] == p)
        hf = heat("full", p)["heat"].astype(np.float64); hp = heat("plain", p)["heat"].astype(np.float64)
        r_list.append(float(np.corrcoef(hf.ravel(), hp.ravel())[0, 1]))
        gb = GT[p]["boxes"][GT[p]["labels"] == FUR]
        m = np.zeros((H, W), bool)
        for b in gb:
            m[int(round(b[1])):int(round(b[3])), int(round(b[0])):int(round(b[2]))] = True
        if m.any():
            area.append(float(m.mean()))
            for v, h in (("full", hf), ("plain", hp)):
                hm = np.where(h >= 0.05, h, 0)
                inside[v].append(float(hm[m].sum() / max(hm.sum(), 1e-9)))
        for v, h in (("full", hf), ("plain", hp)):
            ext = p.startswith("test_alte")
            pr = preds(v, p, external=ext, fold=int(heat(v, p)["fold"]))
            s = pr["scores"][pr["labels"] == FUR]
            if len(s):
                peak[v].append(abs(float(h.max()) - float(s.max())))
    return dict(n_images=len(names), pearson_full_plain_median=float(np.median(r_list)), pearson_min=float(min(r_list)),
                pearson_max=float(max(r_list)), mass_in_box_full_median=float(np.median(inside["full"])),
                mass_in_box_plain_median=float(np.median(inside["plain"])), box_area_share_median=float(np.median(area)),
                n_with_fur=len(area), peak_vs_saved_max_absdiff=float(max(peak["full"] + peak["plain"])),
                n_peak_checked=len(peak["full"]) + len(peak["plain"]))


if __name__ == "__main__":
    out = {}
    out["map_stats"] = map_stats()
    fig_architecture()
    out["gt"] = fig_ground_truth()
    out["qual_counts"] = fig_predictions()
    fig_score_maps(Q["main"], "Figure6_score_maps")
    fig_score_maps(Q["external"], "FigureS8_external_score_maps", external=True)
    out["gt_prepost_checks"] = fig_gt_prepost()
    out["dup_mean_absdiff"] = fig_duplicate()
    out["fp_overlapping_gt"] = fig_failures()
    out["midline"] = fig_midline()
    out["thresholds"] = Q["thresholds"]; out["n_fp_total"] = Q["failures"]["n_fp_total"]; out["n_fn_total"] = Q["failures"]["n_fn_total"]
    json.dump(out, open(os.path.join(HERE, "qual_numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(out, default=float)[:600])
