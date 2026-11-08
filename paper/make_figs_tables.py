"""All manuscript figures, tables and the numbers used in the text, generated from saved result files only.
  python paper/make_figs_tables.py
Inputs (v4_frontier/): res_final.json (analyze.py), sens_final.json (sensitivity.py), paper/stats_extra.json,
folds_summary_local.json, data_audit_local.json, and the old FastViT checkpoint history (../fastvit_results/...).
Outputs: paper/figures/*.png|tif, paper/tables/*.csv, paper/numbers.json.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
J = lambda p: json.load(open(os.path.join(ROOT, p)))

RES = J("res_final.json"); SENS = J("sens_final.json"); EX = J("paper/stats_extra.json")
FS = J("folds_summary_local.json"); AUD = J("data_audit_local.json")
OLD = json.load(open(os.path.join(ROOT, "..", "fastvit_results", "outputs_fastvit", "checkpoint_metrics.json")))["history"]

NAME = {"rtdetr-l": "RT-DETR-L", "yolo11m": "YOLO11m", "yolov8m": "YOLOv8m", "plain": "DINOv2-FCOS (plain)",
        "full": "BiFurca-DINO (full)", "no_mirror": "BiFurca w/o mirror", "no_coords": "BiFurca w/o arch coords",
        "no_context": "BiFurca w/o context"}
ORDER = [v for v in ["rtdetr-l", "yolo11m", "yolov8m", "plain", "full", "no_mirror", "no_coords", "no_context"] if v in RES]
FAMILY = {v: ("CNN/DETR benchmark" if v in ("rtdetr-l", "yolo11m", "yolov8m") else "Frozen DINOv2") for v in ORDER}
COL = {"CNN/DETR benchmark": "#2b6cb0", "Frozen DINOv2": "#c05621"}
# one distinct shade per model, within its family hue (blues = fine-tuned benchmarks, oranges = frozen DINOv2)
MCOL = {"rtdetr-l": "#08306b", "yolo11m": "#2171b5", "yolov8m": "#6baed6",
        "plain": "#7f2704", "full": "#d94801", "no_mirror": "#f16913", "no_coords": "#fd8d3c", "no_context": "#fdae6b"}
CLASSES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT", "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]
CLASS_FULL = {"IMP": "Implant", "PRR": "Prosthetic restoration", "OBT": "Obturation", "END": "Endodontic treatment",
              "CAR": "Caries", "BON": "Bone loss", "IMT": "Impacted tooth", "API": "Apical periodontitis",
              "ROT": "Root fragment", "FUR": "Furcation lesion", "APS": "Apical surgery", "ROR": "Root resorption",
              "ORD": "Orthodontic appliance", "SRD": "Surgical device"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def b0(v):
    return RES[v]["by_seed"]["0"]


def save(fig, name):
    for ext in ("png", "tif"):
        kw = dict(dpi=300, bbox_inches="tight")
        if ext == "tif":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), **kw)
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(TAB, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


f3 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
N = {}  # every number quoted in the text

# ---------------------------------------------------------------- data numbers
tr = AUD["by_dir"]
N["n_raw_images"] = AUD["n_images"]; N["near_dup_pairs"] = AUD["near_duplicate_pairs"]; N["cross_split_dups"] = AUD["cross_dir_dups"]
N["n_train_orig"] = tr["train/images"]["n_images"]; N["n_valid_orig"] = tr["valid/images"]["n_images"]; N["n_test_orig"] = tr["test/images"]["n_images"]
N["fur_valid_orig"] = tr["valid/images"]["boxes_per_class"]["FUR"]
N["n_pool"] = FS["n_pool"]; N["n_groups"] = FS["n_groups"]; N["n_singletons"] = FS["n_singletons"]
N["n_fur_pool"] = b0(ORDER[0])["n_fur"]
N["ext_images"] = FS["external"]["images"]; N["ext_fur"] = FS["external"]["fur_boxes"]; N["ext_dup_in_pool"] = FS["external"]["with_near_dup_in_pool"]
N["fur_area_median_pct"] = AUD["fur_box_area_frac"]["median"] * 100
N["fur_boxes_per_fold"] = [FS["per_fold"][str(k)]["fur_boxes"] for k in range(5)]
r0 = json.load(open(os.path.join(ROOT, "res_all", ORDER[0], "fold0_seed0.json")))
N["n_train_fold0"] = r0["n_train"]; N["n_cal_fold0"] = r0["n_cal"]; N["n_test_fold0"] = r0["n_test"]
oldf = [(e, x) for e, x in zip(OLD["epoch"], OLD["fur_ap50"]) if x is not None]
N["old_ckpt_epochs"] = [e for e, _ in oldf]; N["old_ckpt_fur"] = [x for _, x in oldf]
N["old_selected"] = max(x for _, x in oldf); N["old_selected_epoch"] = [e for e, x in oldf if x == N["old_selected"]][0]
N["old_final"] = oldf[-1][1]; N["old_min"] = min(x for _, x in oldf)

# ---------------------------------------------------------------- per-model numbers
for v in ORDER:
    b = b0(v); e = EX["variants"][v]["external"]
    N[v] = dict(fur=b["fur_ap50"], lo=b["fur_ap50_ci"][0], hi=b["fur_ap50_ci"][1], fur5095=b["fur_ap5095"],
                other=b["map50_other"], all=b["map50_all"], r01=b["fur_recall_fppi"]["0.1"], r025=b["fur_recall_fppi"]["0.25"],
                r05=b["fur_recall_fppi"]["0.5"], folds=b["per_fold_fur_ap50"], ext=e["fur_ap50_mean"], ext_min=e["fur_ap50_min"],
                ext_max=e["fur_ap50_max"], ext_other=float(np.mean(b["external_map50_other_per_fold"])),
                conf=b["conformal"], max_recall=EX["variants"][v]["pr_fur"]["recall"][-1],
                per_class=EX["variants"][v]["per_class_ap50"])
N["paired"] = EX["paired"]; N["best_benchmark"] = EX["best_benchmark"]; N["n_boot"] = EX["n_boot"]
N["sens"] = SENS

# ---------------------------------------------------------------- Figure 1: study design
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)


def box(x, y, w, h, t, fc="#edf2f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="#2d3748", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.0)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9, color="#2d3748"))


box(0.2, 8.1, 4.4, 1.6, f"YOLO-Dental (public, 14 classes)\n{N['n_raw_images']:,} panoramic radiographs\n"
    f"train {N['n_train_orig']:,} / valid {N['n_valid_orig']} / test {N['n_test_orig']} + external clinic {N['ext_images']}")
box(5.4, 8.1, 4.4, 1.6, f"Data audit (perceptual hash)\n{N['near_dup_pairs']} near-duplicate pairs,\n{N['cross_split_dups']} across the original splits\n"
    f"original validation set: only {N['fur_valid_orig']} furcation boxes", fc="#fed7d7")
arrow(4.6, 8.9, 5.4, 8.9)
box(0.2, 5.6, 4.4, 1.8, f"Pooled development set\n{N['n_pool']:,} images, {N['n_fur_pool']} furcation boxes\n"
    f"{N['n_groups']:,} duplicate groups → 5 group-disjoint folds\n(no near-duplicate crosses a fold)", fc="#e6fffa")
box(5.4, 5.6, 4.4, 1.8, f"External clinic set (held out)\n{N['ext_images']} images, {N['ext_fur']} furcation boxes\n"
    f"{N['ext_dup_in_pool']} near-duplicates of the pool\nscored by every fold model", fc="#e6fffa")
arrow(2.4, 8.1, 2.4, 7.4); arrow(7.6, 8.1, 7.6, 7.4)
box(0.2, 2.9, 4.4, 2.0, "Each fold: train / 12% calibration groups / test\n\nBenchmarks: YOLOv8m, YOLO11m, RT-DETR-L\n(COCO-pretrained, 60 epochs, 1280 px)\n"
    "Frozen DINOv2-base + FCOS: plain, BiFurca\n(full, − mirror, − coords, − context; 20 epochs)")
box(5.4, 2.9, 4.4, 2.0, "Pre-specified protocol (dated file, before any\nBiFurca result)\nfinal-epoch weights, no checkpoint selection\n"
    "primary: pooled out-of-fold furcation AP@50\n2,000× group bootstrap, paired comparisons\nconformal lesion-recall threshold", fc="#fefcbf")
arrow(2.4, 5.6, 2.4, 4.9); arrow(4.6, 3.9, 5.4, 3.9)
box(2.6, 0.3, 4.8, 1.8, "Outputs\nfurcation AP@50 (95% CI), AP@50:95, recall at 0.1/0.25/0.5 FP per image,\n"
    "other-class mAP@50, external AP, ablations, sensitivity analyses", fc="#edf2f7")
arrow(2.4, 2.9, 4.0, 2.1); arrow(7.6, 2.9, 6.0, 2.1)
save(fig, "Figure1_study_design")

# ---------------------------------------------------------------- Figure 2: single split instability vs grouped CV
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.35))
a = axs[0]
a.plot(N["old_ckpt_epochs"], N["old_ckpt_fur"], "o-", color="#718096")
a.scatter([N["old_selected_epoch"]], [N["old_selected"]], s=60, color="#c53030", zorder=3, label=f"reported ({N['old_selected']:.3f})")
a.scatter([N["old_ckpt_epochs"][-1]], [N["old_final"]], s=60, color="#2b6cb0", zorder=3, label=f"final epoch ({N['old_final']:.3f})")
a.set_xlabel("Training epoch (checkpoint)"); a.set_ylabel("Furcation AP@50"); a.set_ylim(0, 0.45)
a.set_title(f"A  Earlier single split\n({N['fur_valid_orig']} validation furcation boxes)", fontsize=8.5, loc="left")
a.legend(frameon=False, fontsize=7)
a = axs[1]
for k, v in enumerate(ORDER):
    y = N[v]["folds"]
    a.scatter(np.full(5, k) + np.linspace(-0.15, 0.15, 5), y, s=14, color=MCOL[v], alpha=0.8)
    a.plot([k - 0.25, k + 0.25], [N[v]["fur"]] * 2, color="k", lw=1.4)
a.set_xticks(range(len(ORDER))); a.set_xticklabels([NAME[v] for v in ORDER], rotation=40, ha="right", fontsize=7)
a.set_ylabel("Furcation AP@50"); a.set_ylim(0, 0.45)
a.set_title("B  Grouped 5-fold CV: per-fold (dots) and pooled (bar)", fontsize=8.5, loc="left")
save(fig, "Figure2_single_split_vs_grouped_cv")

# ---------------------------------------------------------------- Figure 3: primary endpoint forest
fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(ORDER) + 1.0))
for k, v in enumerate(ORDER[::-1]):
    c = MCOL[v]
    ax.errorbar(N[v]["fur"], k, xerr=[[N[v]["fur"] - N[v]["lo"]], [N[v]["hi"] - N[v]["fur"]]], fmt="o", color=c, capsize=3)
    ax.scatter(N[v]["ext"], k - 0.22, marker="D", s=18, facecolor="white", edgecolor=c)
    ax.text(0.47, k, f"{N[v]['fur']:.3f} ({N[v]['lo']:.3f}–{N[v]['hi']:.3f})", va="center", fontsize=7.3)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([NAME[v] for v in ORDER[::-1]])
ax.set_xlim(0, 0.62); ax.set_xlabel("Furcation AP@50")
ax.scatter([], [], marker="o", color="k", label="pooled out-of-fold (95% group-bootstrap CI)")
ax.scatter([], [], marker="D", facecolor="white", edgecolor="k", label="external clinic set (mean of 5 fold models)")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "Figure3_primary_endpoint")

# ---------------------------------------------------------------- Figure 4: PR curves
fig, ax = plt.subplots(figsize=(4.6, 3.8))
sty = {"rtdetr-l": "-", "yolo11m": "--", "yolov8m": ":", "plain": "-", "full": "--", "no_mirror": ":", "no_coords": "-.", "no_context": (0, (5, 1, 1, 1, 1, 1))}
for v in ORDER:
    pr = EX["variants"][v]["pr_fur"]
    ax.plot(pr["recall"], pr["precision"], linestyle=sty[v], color=MCOL[v], lw=1.3, label=f"{NAME[v]} ({N[v]['fur']:.3f})")
ax.set_xlabel("Recall (furcation lesions)"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=6.5)
save(fig, "Figure4_furcation_PR_curves")

# ---------------------------------------------------------------- Figure 5: paired differences
P = EX["paired"]
PF = lambda p, d=2: ("< 0.001" if p == 0 else "> 0.999" if p == 1 else f"{p:.{d}f}")
fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(P) + 0.9))
for k, (key, d) in enumerate(list(P.items())[::-1]):
    ax.errorbar(d["point"], k, xerr=[[d["point"] - d["ci"][0]], [d["ci"][1] - d["point"]]], fmt="o", color="#2d3748", capsize=3)
    ax.text(0.20, k, f"{d['point']:+.3f}  P(A>B) {PF(d['p_a_gt_b']) if d['p_a_gt_b'] in (0, 1) else '= ' + PF(d['p_a_gt_b'])}", va="center", fontsize=7)
ax.axvline(0, color="#a0aec0", lw=0.8)
ax.set_yticks(range(len(P))); ax.set_yticklabels([f"{NAME[d['a']]} − {NAME[d['b']]}" for d in list(P.values())[::-1]], fontsize=7.5)
ax.set_xlim(-0.25, 0.36); ax.set_xlabel("Difference in furcation AP@50 (A − B), paired group bootstrap 95% CI")
save(fig, "Figure5_paired_differences")

# ---------------------------------------------------------------- Supplementary figures
M = np.array([[EX["variants"][v]["per_class_ap50"][c] if EX["variants"][v]["per_class_ap50"][c] is not None else np.nan for c in CLASSES] for v in ORDER])
fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(ORDER) + 1.4))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 0.6 else "black")
ax.set_xticks(range(14)); ax.set_xticklabels(CLASSES, fontsize=7); ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([NAME[v] for v in ORDER], fontsize=7)
fig.colorbar(im, ax=ax, label="AP@50", fraction=0.03)
save(fig, "FigureS1_per_class_AP50")

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
sv = [v for v in ORDER if v in SENS]
x = np.arange(3); w = 0.8 / len(sv)
for k, v in enumerate(sv):
    axs[0].bar(x + k * w, [SENS[v][f"fur_ap_iou{t}"][0] for t in (0.3, 0.5, 0.75)], w, color=MCOL[v], label=NAME[v])
    axs[1].bar(x + k * w, [SENS[v][f"fur_ap50_{s}_lesions"] for s in ("small", "medium", "large")], w, color=MCOL[v])
axs[0].set_xticks(x + 0.4 - w / 2); axs[0].set_xticklabels(["IoU 0.3", "IoU 0.5", "IoU 0.75"]); axs[0].set_ylabel("Furcation AP")
axs[1].set_xticks(x + 0.4 - w / 2); axs[1].set_xticklabels(["small", "medium", "large"]); axs[1].set_ylabel("Furcation AP@50")
axs[0].set_title("A  Matching IoU threshold", loc="left", fontsize=8.5); axs[1].set_title("B  Lesion size tertile", loc="left", fontsize=8.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.12))
fig.subplots_adjust(wspace=0.3)
save(fig, "FigureS2_sensitivity")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for v in ORDER:
    logs = EX["variants"][v]["train_logs"]
    if not logs or not logs[0]["log"]:
        continue
    L = np.array([[e["loss"] for e in l["log"]] for l in logs])
    ax.plot(range(1, L.shape[1] + 1), L.mean(0), linestyle=sty[v], color=MCOL[v], label=NAME[v])
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (mean of 5 folds)"); ax.legend(frameon=False, fontsize=7)
save(fig, "FigureS3_DINO_training_loss")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
fp = [0.1, 0.25, 0.5]
for v in ORDER:
    ax.plot(fp, [N[v]["r01"], N[v]["r025"], N[v]["r05"]], marker="o", linestyle=sty[v], color=MCOL[v], label=NAME[v])
ax.set_xlabel("False-positive furcation boxes per image"); ax.set_ylabel("Furcation lesion recall"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=6.5)
save(fig, "FigureS4_recall_at_FPPI")

# ---------------------------------------------------------------- Tables
rows = [[f"Fold {k}", FS["per_fold"][str(k)]["images"], FS["per_fold"][str(k)]["with_fur"], FS["per_fold"][str(k)]["fur_boxes"]] for k in range(5)]
rows.append(["Pooled development set", N["n_pool"], sum(FS["per_fold"][str(k)]["with_fur"] for k in range(5)), N["n_fur_pool"]])
rows.append(["External clinic set", N["ext_images"], tr["test_alte_cabinete/Ext-validation/images"]["images_with_class"]["FUR"], N["ext_fur"]])
write_csv("Table1_dataset_folds", ["Partition", "Images", "Images with furcation", "Furcation boxes"], rows)

write_csv("Table2_main_results",
          ["Model", "Furcation AP@50 (95% CI)", "Furcation AP@50:95", "Recall @0.1 FP/img", "Recall @0.25 FP/img", "Recall @0.5 FP/img",
           "Other 13 classes mAP@50", "External furcation AP@50, mean (range)", "External other-class mAP@50"],
          [[NAME[v], f"{N[v]['fur']:.3f} ({N[v]['lo']:.3f}–{N[v]['hi']:.3f})", f3(N[v]["fur5095"]), f3(N[v]["r01"]), f3(N[v]["r025"]), f3(N[v]["r05"]),
            f3(N[v]["other"]), f"{N[v]['ext']:.3f} ({N[v]['ext_min']:.3f}–{N[v]['ext_max']:.3f})", f3(N[v]["ext_other"])] for v in ORDER])

bb = N["best_benchmark"]
pf = P.get("full__vs__plain"); pb = P.get(f"full__vs__{bb}")
crit = [
    ["1. BiFurca beats plain DINOv2, P(diff>0) ≥ 0.95", f"diff {pf['point']:+.3f} ({pf['ci'][0]:+.3f} to {pf['ci'][1]:+.3f}); P {PF(pf['p_a_gt_b']) if pf['p_a_gt_b'] in (0, 1) else '= ' + PF(pf['p_a_gt_b'])}", "No" if pf["p_a_gt_b"] < 0.95 else "Yes"],
    [f"2. BiFurca beats best benchmark ({NAME[bb]}), P ≥ 0.95", f"diff {pb['point']:+.3f} ({pb['ci'][0]:+.3f} to {pb['ci'][1]:+.3f}); P {PF(pb['p_a_gt_b']) if pb['p_a_gt_b'] in (0, 1) else '= ' + PF(pb['p_a_gt_b'])}", "No" if pb["p_a_gt_b"] < 0.95 else "Yes"],
    ["3. Direction holds externally and in ≥ 4/5 folds", f"external diff vs {NAME[bb]} {pb['ext_diff']:+.3f}; better in {pb['folds_a_better']}/5 folds",
     "Yes" if (pb["ext_diff"] > 0 and pb["folds_a_better"] >= 4) else "No"],
    ["4. Other-class mAP@50 within 0.02 of best benchmark", f"{N['full']['other']:.3f} vs {N[bb]['other']:.3f} ({N['full']['other'] - N[bb]['other']:+.3f})",
     "Yes" if N["full"]["other"] - N[bb]["other"] >= -0.02 else "No"]]
write_csv("Table3_preregistered_criteria", ["Pre-specified criterion", "Observed", "Met"], crit)
N["criteria"] = crit

write_csv("Table4_paired_comparisons", ["Comparison (A − B)", "Difference in furcation AP@50", "95% CI", "P(A > B)", "Folds A better", "External difference"],
          [[f"{NAME[d['a']]} − {NAME[d['b']]}", f"{d['point']:+.3f}", f"{d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}", PF(d['p_a_gt_b'], 3),
            f"{d['folds_a_better']}/5", f"{d['ext_diff']:+.3f}"] for d in P.values()])

write_csv("TableS1_per_class_AP50", ["Class", "Pooled boxes"] + [NAME[v] for v in ORDER],
          [[f"{CLASS_FULL[c]} ({c})", EX["variants"][ORDER[0]]["n_gt_per_class"][c]] + [f3(EX["variants"][v]["per_class_ap50"][c]) for v in ORDER] for c in CLASSES])
write_csv("TableS2_sensitivity", ["Model", "AP IoU 0.3", "AP IoU 0.5", "AP IoU 0.75", "AP@50 excl. duplicate-group images (n)",
                                  "AP@50 image-level bootstrap CI", "AP@50 small", "AP@50 medium", "AP@50 large"],
          [[NAME[v]] + [f"{SENS[v][f'fur_ap_iou{t}'][0]:.3f} ({SENS[v][f'fur_ap_iou{t}'][1]:.3f}–{SENS[v][f'fur_ap_iou{t}'][2]:.3f})" for t in (0.3, 0.5, 0.75)]
           + [f"{SENS[v]['fur_ap50_excluding_dup_images']['ap']:.3f} ({SENS[v]['fur_ap50_excluding_dup_images']['n_images']})",
              f"{SENS[v]['fur_ap50_image_level_ci'][1]:.3f}–{SENS[v]['fur_ap50_image_level_ci'][2]:.3f}"]
           + [f3(SENS[v][f"fur_ap50_{s}_lesions"]) for s in ("small", "medium", "large")] for v in ORDER if v in SENS])
write_csv("TableS3_per_fold_AP50", ["Model"] + [f"Fold {k} ({N['fur_boxes_per_fold'][k]} boxes)" for k in range(5)] + ["SD across folds"],
          [[NAME[v]] + [f3(x) for x in N[v]["folds"]] + [f"{np.std(N[v]['folds'], ddof=1):.3f}"] for v in ORDER])
cr = []
for v in ORDER:
    for al in ("0.1", "0.2"):
        c = N[v]["conf"][al]
        c5 = EX["variants"][v]["conformal_iou05"][al]
        cr.append([NAME[v], f"{c['target_recall']:.1f}", f"{c['folds_feasible']}/5", f3(c["achieved_recall"]) if c["folds_feasible"] else "infeasible",
                   f3(c["fp_per_image"]) if c["folds_feasible"] else "—", f3(c["mean_tau"]) if c["mean_tau"] is not None else "—",
                   f"{c5['folds_feasible']}/5"])
write_csv("TableS4_conformal", ["Model", "Target lesion recall", "Folds feasible (IoU 0.3)", "Achieved recall (test, IoU 0.3)",
                                "False-positive boxes per image", "Mean threshold", "Folds feasible (IoU 0.5; sensitivity)"], cr)
N["conf05_feasible"] = {al: sum(EX["variants"][v]["conformal_iou05"][al]["folds_feasible"] for v in ORDER) for al in ("0.1", "0.2")}
write_csv("TableS5_old_single_split_checkpoints", ["Epoch", "Validation furcation AP@50 (FastViT-SA24 + FCOS, earlier study)"],
          [[e, f"{x:.3f}"] for e, x in zip(N["old_ckpt_epochs"], N["old_ckpt_fur"])])
write_csv("TableS6_training_configuration", ["Model", "Pretraining", "Trainable parts", "Input size", "Epochs", "Optimiser / schedule", "Weights evaluated"],
          [["YOLOv8m, YOLO11m, RT-DETR-L", "COCO", "all layers", "1280 px (long side)", "60", "Ultralytics default, cosine LR, AMP, seed 0", "final epoch"],
           ["DINOv2-FCOS plain and BiFurca variants", "DINOv2-base (frozen)", "neck + modules + FCOS head", "728 × 1316 px", "20",
            "AdamW 2e-4, wd 1e-4, 200-step warm-up + cosine, batch 4, seed 0", "final epoch"]])

json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
print("figures:", sorted(os.listdir(FIG))); print("tables:", sorted(os.listdir(TAB)))
