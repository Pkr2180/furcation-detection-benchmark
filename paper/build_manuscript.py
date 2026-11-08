"""Build the manuscript and supplementary .docx from paper/numbers.json and paper/tables/*.csv. Never hand-edit numbers.
  python paper/make_figs_tables.py && python paper/build_manuscript.py && python paper/audit_manuscript.py
"""
import csv, json, os, re
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, Cm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables"); OUT = os.path.join(HERE, "manuscript")
os.makedirs(OUT, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
QN = json.load(open(os.path.join(HERE, "qual_numbers.json")))      # make_qual_figs.py
QC = json.load(open(os.path.join(HERE, "qual_cases.json")))        # qual_select.py (pre-specified cases)
NAME = {"rtdetr-l": "RT-DETR-L", "yolo11m": "YOLO11m", "yolov8m": "YOLOv8m", "plain": "the plain DINOv2-FCOS detector",
        "full": "BiFurca-DINO", "no_mirror": "BiFurca without the mirror module", "no_coords": "BiFurca without arch coordinates",
        "no_context": "BiFurca without the context transformer"}
COMPONENT = {"no_mirror": "the midline-mirror (contralateral) module", "no_coords": "the arch-canonical coordinates",
             "no_context": "the coarse context transformer"}
BENCH = [v for v in ("rtdetr-l", "yolo11m", "yolov8m") if v in N]
ABL = [v for v in ("no_mirror", "no_coords", "no_context") if v in N]
bb = N["best_benchmark"]; P = N["paired"]
f3 = lambda x: f"{x:.3f}"
ci = lambda v: f"{f3(N[v]['fur'])} (95% CI {f3(N[v]['lo'])}–{f3(N[v]['hi'])})"
dci = lambda d: f"{d['point']:+.3f} (95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}; P(A>B) = {d['p_a_gt_b']:.2f})"
pfp = P["full__vs__plain"]; pfb = P[f"full__vs__{bb}"]; ppb = P[f"plain__vs__{bb}"]
others = [b for b in BENCH if b != bb]
ext_best = max([v for v in NAME if v in N], key=lambda v: N[v]["ext"])
spread = {v: (min(N[v]["folds"]), max(N[v]["folds"])) for v in NAME if v in N}
DINO = ["plain", "full"] + ABL
maxrec = max(N[v]["max_recall"] for v in BENCH + DINO)
maxrec_v = max(BENCH + DINO, key=lambda v: N[v]["max_recall"])
feas80 = sum(N[v]["conf"]["0.2"]["folds_feasible"] for v in BENCH + DINO)
ratio = N["full"]["fur"] / N[bb]["fur"]
half_words = "about half" if 0.4 <= ratio <= 0.6 else f"{ratio:.0%} of"
fold_gap = spread[bb][1] - spread[bb][0]; bench_gap = N[bb]["fur"] - min(N[b]["fur"] for b in BENCH)
# sensitivity wording decided from the numbers, not asserted
S_ = N["sens"]
sens_keys = {"IoU 0.3": lambda v: S_[v]["fur_ap_iou0.3"][0], "IoU 0.5": lambda v: S_[v]["fur_ap_iou0.5"][0],
             "IoU 0.75": lambda v: S_[v]["fur_ap_iou0.75"][0],
             "duplicate-group images excluded": lambda v: S_[v]["fur_ap50_excluding_dup_images"]["ap"]}
held = [k for k, f in sens_keys.items() if min(f(b) for b in BENCH) > max(f(d) for d in DINO if d in S_)]
broke = [k for k in sens_keys if k not in held]
top = {k: max(BENCH + [d for d in DINO if d in S_], key=f) for k, f in sens_keys.items()}
small_worst = all(S_[v]["fur_ap50_small_lesions"] < min(S_[v]["fur_ap50_medium_lesions"], S_[v]["fur_ap50_large_lesions"]) for v in S_)

# ------------------------------------------------------------------ ablation wording, chosen by rule from the CIs
abl_sent, abl_null = [], True
for a in ABL:
    d = P.get(f"full__vs__{a}")
    if d is None:
        continue
    abl_sent.append(f"removing {COMPONENT[a]} changed furcation AP@50 by {-d['point']:+.3f} "
                    f"(full minus ablated {d['point']:+.3f}, 95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f})")
    if d["ci"][0] > 0 or d["ci"][1] < 0:
        abl_null = False
if ABL and len(abl_sent) == len(ABL):
    abl_text = ("In the pre-specified ablations, " + "; ".join(abl_sent) + ". " +
                ("All three intervals included zero, so no anatomical component made a detectable contribution."
                 if abl_null else "At least one interval excluded zero; the direction of each effect is given in Table 3."))
    abl_methods_done = True
else:
    abl_text = ("The three single-component ablations were still running when this version was generated and are "
                "reported in a later version.")
    abl_methods_done = False

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.paragraph_format.line_spacing = 2.0; st.paragraph_format.space_after = Pt(0)
for s in doc.sections:
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)


def para(text, bold_lead=None, align=None, size=None, italic=False):
    p = doc.add_paragraph()
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if bold_lead:
        r = p.add_run(bold_lead + " "); r.bold = True
        if size: r.font.size = Pt(size)
    for k, part in enumerate(re.split(r"\^\{([^}]*)\}", text)):
        r = p.add_run(part)
        if k % 2:
            r.font.superscript = True
        if size: r.font.size = Pt(size)
        r.italic = italic
    return p


def head(t, lvl=1):
    h = doc.add_heading(t, level=lvl)
    for r in h.runs:
        r.font.name = "Times New Roman"; r.font.color.rgb = None


def table_csv(name, caption, note=None, d=None):
    d = d or doc
    rows = list(csv.reader(open(os.path.join(TAB, name + ".csv"), encoding="utf-8")))
    p = d.add_paragraph(); r = p.add_run(caption); r.bold = True
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            cell = t.cell(i, j); cell.text = ""
            run = cell.paragraphs[0].add_run(c); run.font.size = Pt(8); run.bold = (i == 0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    if note:
        q = d.add_paragraph(); rr = q.add_run(note); rr.font.size = Pt(9); rr.italic = True
        q.paragraph_format.line_spacing = 1.0
    d.add_paragraph()


def table_rows(header, rows, caption, note=None, d=None):
    path = os.path.join(TAB, "_tmp_rows.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    table_csv("_tmp_rows", caption, note, d)
    os.remove(path)


def figure(name, legend, d=None, width=6.3):
    d = d or doc
    d.add_picture(os.path.join(FIG, name + ".png"), width=Inches(width))
    p = d.add_paragraph(); p.paragraph_format.line_spacing = 1.0
    lead, rest = legend.split(". ", 1)
    r = p.add_run(lead + ". "); r.bold = True
    p.add_run(rest)
    d.add_paragraph()


# ================================================================== TITLE PAGE
TITLE = ("Automated furcation detection on panoramic radiographs is not yet solved: a position paper on leakage-safe "
         "evaluation, supported by a pre-registered benchmark")
REPO = "https://github.com/Pkr2180/furcation-detection-benchmark"
para(TITLE, align="c", size=14).runs[0].bold = True
para("Pradeep Kumar Yadalam, BDS, MDS, PhD", align="c")
para("Professor and Head of Research, Department of Periodontics, Saveetha Dental College and Hospitals, Saveetha "
     "Institute of Medical and Technical Sciences (SIMATS), Chennai 600077, Tamil Nadu, India.", align="c")
para("Corresponding author: Pradeepkumar.sdc@saveetha.com", align="c")

# ================================================================== ABSTRACT
head("Abstract")
para("Deep-learning detectors for furcation involvement on panoramic radiographs are reported with promising "
     "accuracy, usually from one small validation split. We argue that these figures overstate what the technology can "
     "do, that the main obstacle is evaluation rather than architecture, and that the field needs a minimum evaluation "
     "standard before new architectures are claimed. We support this position with a pre-registered, leakage-safe "
     "benchmark of current detectors and of a new anatomy-informed foundation-model detector.", "Position:")
para(f"The public YOLO-Dental dataset (14 radiographic findings) was audited with perceptual hashing, which found "
     f"{N['near_dup_pairs']} near-duplicate image pairs, {N['cross_split_dups']} of them across the published splits. "
     f"The {N['n_pool']:,} development images ({N['n_fur_pool']} furcation boxes) were grouped by duplicate cluster into "
     f"five group-disjoint folds; a separate clinic set ({N['ext_images']} images, {N['ext_fur']} furcation boxes) was "
     f"held out. YOLOv8m, YOLO11m and RT-DETR-L were compared with a frozen DINOv2 detector and with BiFurca-DINO, which "
     f"adds a label-free midline-mirror module, arch-canonical coordinates and a context transformer. Final-epoch weights "
     f"were evaluated. The primary endpoint was pooled out-of-fold furcation average precision at 50% overlap (AP@50) "
     f"with a group-bootstrap 95% confidence interval (CI).", "Evidence (methods):")
para(f"{NAME[bb]} had the highest furcation AP@50, {ci(bb)}, followed by "
     + " and ".join(f"{NAME[b]} {ci(b)}" for b in others) +
     f". BiFurca-DINO reached {ci('full')} and the plain DINOv2 detector {ci('plain')}; the paired difference was "
     f"{dci(pfp)}. All four pre-registered criteria for a new-model claim failed. Per-fold furcation AP@50 ranged from "
     f"{f3(spread[bb][0])} to {f3(spread[bb][1])} for the best model. Pooled lesion recall never exceeded "
     f"{f3(maxrec)} for any model, and a conformal threshold for 80% recall was feasible in {feas80} of "
     f"{5 * len(BENCH + DINO)} model-folds. An earlier single-split analysis of this dataset had reported "
     f"{f3(N['old_selected'])} by choosing the best of six checkpoints on {N['fur_valid_orig']} validation boxes; its "
     f"final epoch scored {f3(N['old_final'])}.", "Evidence (results):")
para("We propose a ten-item minimum evaluation standard: duplicate audit and group-disjoint splits; no model "
     "selection on the reported data; pooled out-of-fold or multi-site estimates with interval estimates; lesion-level "
     "recall at stated false-positive rates; an untouched external set; strong fine-tuned baselines; paired "
     "comparisons; pre-registered decision rules with ablations; reporting of negative results; and release of "
     "splits, predictions and code. New architectures should be judged only against this standard.",
     "Recommendations:")
para("Current panoramic furcation detectors miss too many lesions to rule out furcation involvement; clinical probing "
     "remains necessary, and published accuracy figures should be read with the evaluation protocol in mind.",
     "Clinical significance:")
para("furcation defects; panoramic radiography; deep learning; object detection; data leakage; evaluation standards; "
     "position paper; foundation models; periodontitis", "Keywords:")

# ================================================================== INTRODUCTION
head("1. Introduction")
para("Furcation involvement is the destruction of periodontal support between the roots of multi-rooted teeth. It "
     "reduces the long-term survival of affected molars and changes the treatment plan.^{1–3} Diagnosis relies on "
     "horizontal probing, graded with Hamp's classification, supported by radiographs.^{2,4} Panoramic radiographs are "
     "taken routinely and could flag furcations for closer examination, but projection distortion, superimposition and "
     "observer variation make radiographic assessment unreliable.^{5,6}")
para("Deep learning is now widely applied to dental radiographs for caries, periapical lesions, bone loss and "
     "landmarks.^{7–14} For furcation involvement, two recent systematic reviews found high pooled accuracy but "
     "marked heterogeneity and immature evidence, and both asked for standardised protocols and external "
     "validation.^{15,16} Most studies classify pre-cropped teeth, which is an easier task than locating a lesion on a "
     "full radiograph.^{17–20} Detection studies are few and use periapical radiographs, deciduous molars or paediatric "
     "panoramic cohorts,^{21–23} and one study used segmentation.^{24} Vision transformers and foundation models remain "
     "largely untested in this area.^{25}")
para("Two problems limit this evidence. First, medical-imaging datasets often contain near-duplicate images; if copies "
     "fall on both sides of a train-test split, accuracy is overstated.^{26,27} Second, when the best checkpoint or "
     "configuration is chosen on the same small validation set that is reported, the estimate is biased upwards. A "
     "rare finding such as furcation lesions makes both problems worse because a handful of boxes decides the score. Our "
     f"own earlier single-split analysis of a public panoramic dataset illustrates this. The reported furcation AP@50 of "
     f"{f3(N['old_selected'])} was the best of six checkpoints evaluated on {N['fur_valid_orig']} validation boxes, "
     f"whereas the checkpoints ranged from {f3(N['old_min'])} to {f3(N['old_selected'])} and the final epoch scored "
     f"{f3(N['old_final'])} (Figure 1A; Supplementary Table S1).")
para("Self-supervised foundation models such as DINOv2 produce general-purpose image features without labels.^{28} "
     "A furcation lesion also has anatomical regularities a detector could exploit: it occurs in the molar region, "
     "and a panoramic radiograph is roughly mirror-symmetric, so the contralateral molar is a natural within-patient "
     "reference. We designed BiFurca-DINO to encode these priors on top of frozen DINOv2 features and fixed in advance "
     "the criteria it had to meet.")
para("This is a position paper. Our position has three parts. (1) Reported accuracy for radiographic furcation "
     "detection is inflated by evaluation choices, chiefly near-duplicate leakage and model selection on small "
     "validation sets. (2) Under leakage-safe evaluation the task is far from solved, and an architectural novelty "
     "that is not tested against strong fine-tuned baselines, with ablations and pre-registered criteria, should not "
     "be accepted as progress. (3) The field needs a shared minimum evaluation standard. Sections 2 and 3 give the "
     "evidence (study design in Figure 2): we audited a public dataset, built a leakage-safe grouped cross-validation with an untouched external "
     "set, benchmarked current detectors, and tested our own anatomy-informed foundation-model detector against "
     "criteria fixed in advance. Section 4 explains why single-split figures mislead, Section 5 proposes the "
     "standard, Section 6 sets out the open problems that define the real frontier, and Section 7 considers "
     "counter-arguments.")

# ================================================================== METHODS
head("2. Evidence: a pre-registered, leakage-safe benchmark")
head("2.1 Dataset and audit", 2)
para(f"The public YOLO-Dental dataset^{{29}} (CC BY 4.0) contains {N['n_raw_images']:,} de-identified panoramic "
     f"radiographs annotated with bounding boxes for 14 findings, published as training ({N['n_train_orig']:,}), "
     f"validation ({N['n_valid_orig']}) and test ({N['n_test_orig']}) splits plus an external set of {N['ext_images']} "
     f"radiographs from another clinic. All images were hashed perceptually and compared pairwise. The audit found "
     f"{N['near_dup_pairs']} near-duplicate pairs, {N['cross_split_dups']} of which crossed the published splits, and "
     f"only {N['fur_valid_orig']} furcation boxes in the published validation split. Furcation boxes were small (median "
     f"{N['fur_area_median_pct']:.2f}% of the image area; median {QN['gt']['med_w']:.0f} × {QN['gt']['med_h']:.0f} pixels "
     f"at model input, about two DINOv2 patches) and clustered in the posterior mandible and maxilla on both sides "
     f"(Figure 3). Images were resized from their original resolution to 1,316 × 728 pixels and the normalised YOLO "
     f"labels were converted to pixel boxes and binary furcation masks at that resolution (Figure 4). This chain was "
     f"checked on the {len(QC['main'])} radiographs of Figure 4: the label files matched the analysed annotations "
     f"exactly, the rescaled boxes matched the boxes used for evaluation to within "
     f"{max(QN['gt_prepost_checks']['rescaled_vs_analysis_max_px'], 0.001):.3f} pixels, and the stored model inputs "
     f"differed from a fresh resize of the originals by at most "
     f"{max(QN['gt_prepost_checks']['resize_mean_absdiff']):.1f} grey levels (JPEG compression). Boxes narrower or "
     f"shorter than 2 pixels after resizing were discarded. "
     f"One near-duplicate pair crossing the published training and validation splits, drawn at random, was "
     f"pixel-identical (mean absolute difference {QN['dup_mean_absdiff']:.2f} grey levels; Supplementary Figure S1).")
head("2.2 Grouped cross-validation and external set", 2)
para(f"The training, validation and test splits were pooled ({N['n_pool']:,} images, {N['n_fur_pool']} furcation "
     f"boxes). Near-duplicates were linked into {N['n_groups']:,} groups ({N['n_singletons']:,} singletons), and groups "
     f"were assigned to five folds stratified by furcation content, so no near-duplicate pair crossed a fold (Table 1; "
     f"Figure 2). In each fold, 12% of the non-test groups were set aside as a calibration set and the rest were used "
     f"for training (fold 0: {N['n_train_fold0']:,} training, {N['n_cal_fold0']} calibration, {N['n_test_fold0']} test "
     f"images). The external clinic set had no near-duplicates in the pool ({N['ext_dup_in_pool']}) and was scored by "
     f"all five fold models without being used for any decision.")
head("2.3 Detectors", 2)
para("Three widely used detectors served as benchmarks: YOLOv8m,^{30} YOLO11m^{31} and the real-time detection "
     "transformer RT-DETR-L.^{32} Each started from COCO weights and was fine-tuned end to end for 60 epochs at 1,280 "
     "pixels with the Ultralytics defaults, cosine learning rate and seed 0.")
para("The foundation-model detectors used a frozen DINOv2-base backbone^{28} (last four layers mixed by learned "
     "weights) and a fully convolutional one-stage (FCOS) head^{33}"
     " at 728 × 1,316 pixels. The plain model (DINOv2-FCOS) adds nothing else. BiFurca-DINO adds three modules, each of "
     "which can be switched off: (a) a midline-mirror module that locates the facial midline by maximising left-right "
     "feature similarity, without labels, and lets every location attend to its contralateral counterpart; (b) "
     "arch-canonical coordinates, Fourier-encoded distance from the midline and normalised height, so the detector can "
     "learn where in the arch furcations occur; and (c) a two-layer transformer over a coarse feature map so that "
     "evidence from the other 13 findings on the same radiograph is available. The FCOS regression head used a softplus "
     "output because the default ReLU output collapsed to zero-size boxes on frozen features in pilot runs. These "
     "models were trained for 20 epochs with AdamW (learning rate 2 × 10⁻⁴, weight decay 10⁻⁴, warm-up and cosine "
     "decay, batch 4, seed 0). The architecture is shown in Figure 5. Training settings are listed in Supplementary "
     "Table S2; training curves are in Supplementary Figure S2. Every detector kept at most 100 detections per image "
     "across all 14 classes.")
head("2.4 Pre-registered protocol and outcomes", 2)
para("The protocol and decision rules were written down before any BiFurca-DINO result was seen. Final-epoch weights "
     "were evaluated for every model; no checkpoint, threshold or configuration was chosen on test data. The primary "
     "endpoint was pooled out-of-fold furcation AP@50 (COCO 101-point interpolation) over all development images, each "
     "predicted by the model that had not seen it. Secondary outcomes were furcation AP@50:95, lesion recall at 0.1, "
     "0.25 and 0.5 false-positive furcation boxes per image, mean AP@50 over the other 13 findings, and external "
     "furcation AP@50 (mean of the five fold models).")
para(f"Confidence intervals came from {N['n_boot']:,} bootstrap resamples of duplicate groups. Models were compared "
     "with a paired group bootstrap on the same resamples, reporting the difference, its 95% CI and the proportion of "
     "resamples in which model A beat model B. BiFurca-DINO counted as an advance only if it (1) beat the plain DINOv2 "
     "detector with P ≥ 0.95, (2) beat the best benchmark with P ≥ 0.95, (3) kept that direction on the external set "
     "and in at least four of five folds, and (4) lost no more than 0.02 other-class mAP@50 to the best benchmark. "
     "Ablations removing each module were part of the main grid. A split-conformal threshold for 90% and 80% lesion "
     "recall was computed on each fold's calibration set.^{34} Sensitivity analyses on the saved predictions varied the "
     "matching threshold (IoU 0.3, 0.5, 0.75), split lesions into size tertiles, excluded every image belonging to a "
     "duplicate group, and replaced the group bootstrap by an image-level bootstrap. Reporting follows TRIPOD+AI.^{35}")
head("2.5 Ethics", 2)
para("Only a de-identified public dataset was used; no new radiographs were taken and no ethics approval was required.")

# ================================================================== RESULTS
head("3. What the benchmark shows")
head("3.1 Primary endpoint", 2)
para(f"{NAME[bb]} had the highest pooled furcation AP@50, {ci(bb)} (Table 2; Figure 6). "
     + " ".join(f"{NAME[b]} reached {ci(b)}." for b in others) +
     f" The two frozen-DINOv2 detectors scored {half_words} as much: BiFurca-DINO {ci('full')} and the plain model "
     f"{ci('plain')}. Relative to {NAME[bb]}, BiFurca-DINO differed by {dci(pfb)} and the plain model by {dci(ppb)} "
     f"(Table 3; Figure 7). Among the benchmarks, "
     + "; ".join(f"{NAME[bb]} minus {NAME[b]} was {dci(P[f'{bb}__vs__{b}'])}" for b in others if f"{bb}__vs__{b}" in P)
     + ". Precision-recall curves are shown in Supplementary Figure S3.")
para(f"Furcation AP@50:95 was low for every model ({f3(min(N[v]['fur5095'] for v in BENCH + ['plain', 'full']))}–"
     f"{f3(max(N[v]['fur5095'] for v in BENCH + ['plain', 'full']))}), so boxes were rarely tight. At 0.25 false-positive "
     f"boxes per image, lesion recall was {f3(N[bb]['r025'])} for {NAME[bb]} and {f3(N['full']['r025'])} for "
     f"BiFurca-DINO (Supplementary Figure S4).")
head("3.2 Pre-registered criteria and ablations", 2)
para(f"BiFurca-DINO and the plain DINOv2 detector were indistinguishable: {dci(pfp)}. Criterion 1 therefore failed, "
     f"and so did criteria 2–4 (Table 4). Under the pre-registered rule, the new-architecture claim is withdrawn and "
     f"this study is reported as a benchmark. " + abl_text)
head("3.3 Fold-to-fold variation and external set", 2)
para(f"Furcation AP@50 varied widely between folds of 46–72 boxes, from {f3(spread[bb][0])} to {f3(spread[bb][1])} for "
     f"{NAME[bb]} and from {f3(spread['full'][0])} to {f3(spread['full'][1])} for BiFurca-DINO (Figure 1B; "
     f"Supplementary Table S3). For {NAME[bb]} the fold-to-fold spread ({f3(fold_gap)}) was "
     + ("larger than" if fold_gap > bench_gap else "smaller than") +
     f" the pooled gap between the best and the worst benchmark ({f3(bench_gap)}), so a single small split cannot "
     f"reliably rank detectors. On the external clinic set ({N['ext_fur']} furcation boxes), mean furcation "
     f"AP@50 was highest for {NAME[ext_best]} ({f3(N[ext_best]['ext'])}, range over fold models "
     f"{f3(N[ext_best]['ext_min'])}–{f3(N[ext_best]['ext_max'])}); "
     + ", ".join(f"{NAME[v]} {f3(N[v]['ext'])}" for v in BENCH + ['plain', 'full'] if v != ext_best) +
     f". With so few external lesions these estimates are imprecise, and the external ranking "
     + ("differed from" if ext_best != bb else "matched") + " the development ranking.")
head("3.4 Recall control, other findings and sensitivity analyses", 2)
feas = [(v, a) for v in BENCH + ["plain", "full"] + ABL for a in ("0.1", "0.2") if N[v]["conf"][a]["folds_feasible"] > 0]
para(f"The conformal threshold for 90% lesion recall was infeasible in every fold for every model, and the 80% target "
     f"was feasible in {sum(N[v]['conf']['0.2']['folds_feasible'] for v in BENCH + ['plain', 'full'] + ABL)} of "
     f"{5 * len(BENCH + ['plain', 'full'] + ABL)} model-folds (Supplementary Table S4). The reason is that a share of "
     f"furcation lesions received no overlapping prediction at any score. In the pooled out-of-fold predictions "
     f"(IoU 0.5), the highest recall reached at the lowest score threshold was {f3(maxrec)} ({NAME[maxrec_v]}); it "
     f"was {f3(N[bb]['max_recall'])} for {NAME[bb]} and {f3(N['full']['max_recall'])} for BiFurca-DINO. Mean AP@50 for "
     f"the other 13 findings was {f3(N[bb]['other'])} for {NAME[bb]} and {f3(N['full']['other'])} for BiFurca-DINO "
     f"(per-finding values in Supplementary Table S5 and Figure S5).")
S = N["sens"]
para("Every benchmark stayed above both frozen-DINOv2 detectors " + ("in " + ", ".join(held) if held else "in none of the analyses")
     + (f"; at {', '.join(broke)} this no longer held" if broke else "") +
     f" (Supplementary Table S6; Figure S6). At IoU 0.3, furcation AP was {f3(S[bb]['fur_ap_iou0.3'][0])} for {NAME[bb]} "
     f"and {f3(S['full']['fur_ap_iou0.3'][0])} for BiFurca-DINO; at IoU 0.75 every model was at or below "
     f"{f3(max(S[v]['fur_ap_iou0.75'][0] for v in S))}, so no detector localised furcations tightly. Excluding all "
     f"{N['n_pool'] - S[bb]['fur_ap50_excluding_dup_images']['n_images']} images that belonged to a duplicate group gave "
     + ", ".join(f"{f3(S[v]['fur_ap50_excluding_dup_images']['ap'])} for {NAME[v]}" for v in BENCH + ['full'])
     + (f", so the order among the benchmarks changed ({NAME[top['duplicate-group images excluded']]} first)"
        if top["duplicate-group images excluded"] != bb else "") +
     f". The image-level bootstrap interval for {NAME[bb]} was {f3(S[bb]['fur_ap50_image_level_ci'][1])}–"
     f"{f3(S[bb]['fur_ap50_image_level_ci'][2])}, against {f3(N[bb]['lo'])}–{f3(N[bb]['hi'])} for the group bootstrap."
     + (" Small lesions were detected worst by every model." if small_worst else ""))

head("3.5 What the detectors see", 2)
qc = QN["qual_counts"]
para(f"Figure 8 shows four development radiographs with furcation lesions, drawn at random before any output was "
     f"viewed, each scored by the fold model that had not seen it, at the score threshold giving 0.25 false-positive "
     f"furcation boxes per image. Across the {sum(qc['rtdetr-l'][k] for k in (0, 2))} annotated lesions in these "
     f"images, {NAME[bb]} found {qc['rtdetr-l'][0]}, YOLO11m {qc['yolo11m'][0]} and BiFurca-DINO {qc['full'][0]}; "
     f"BiFurca-DINO also produced {qc['full'][1]} false positive" + ("s" if qc['full'][1] != 1 else "") +
     f". The score maps of the two DINOv2 detectors (Figure 9; external examples in Supplementary Figure S7) were "
     f"broad and similar to each other, covering several neighbouring teeth rather than the inter-radicular area, "
     f"in keeping with the null ablations. The label-free midline estimate fell between "
     f"{QN['midline']['min']:.0f} and {QN['midline']['max']:.0f} pixels on 12 random radiographs (image centre "
     f"{1316 / 2:.0f}; Supplementary Figure S8), so the mirror module received a plausible axis. Of the six "
     f"highest-scoring false positives of {NAME[bb]}, {QN['fp_overlapping_gt']} overlapped an annotated furcation but "
     f"did not reach 50% overlap (Supplementary Figure S9), so part of the measured error reflects box geometry "
     f"rather than missed anatomy; across the pooled set, {NAME[bb]} produced {QN['n_fp_total']} false-positive and "
     f"{QN['n_fn_total']} missed furcation boxes at this operating point.")

# ================================================================== DISCUSSION
head("4. Why reported accuracy misleads")
head("4.1 What the evidence says", 2)
para(f"When near-duplicates were kept within folds, weights were fixed in advance and every image was predicted out "
     f"of fold, the best detector found furcation lesions on panoramic radiographs with an AP@50 of "
     f"{f3(N[bb]['fur'])}. This is well below the {f3(N['old_selected'])} reported from the same dataset with a single "
     f"split and checkpoint selection, and well below the near-perfect accuracy reported by tooth-level classification "
     f"studies.^{{15–20}} No model's pooled lesion recall exceeded {f3(maxrec)}, even at its lowest score threshold. Our new anatomy-informed foundation-model "
     f"detector failed every pre-registered criterion, and its added modules changed nothing measurable compared with "
     f"the plain DINOv2 detector.")
head("4.2 Why single-split estimates mislead", 2)
para(f"Two mechanisms explain most of the gap. First, the published validation split held {N['fur_valid_orig']} "
     f"furcation boxes. With so few lesions, the score swings between checkpoints (here {f3(N['old_min'])} to "
     f"{f3(N['old_selected'])}), and picking the maximum reports noise as signal. Second, {N['cross_split_dups']} "
     f"near-duplicate pairs crossed the published splits, so part of any split-based score measures memory rather than "
     f"generalisation.^{{26,27}} Even under grouped cross-validation, folds of 46–72 furcation boxes gave scores that "
     f"differed by up to {f3(spread[bb][1] - spread[bb][0])} for the same detector. Furcation-detection studies should "
     f"therefore report pooled out-of-fold or multi-site estimates with confidence intervals, describe how duplicates "
     f"were handled, and state which weights were evaluated.")
head("4.3 Why the foundation-model detector failed", 2)
para("The DINOv2 backbone was frozen and its features have a stride of 14 pixels, while the median furcation box "
     f"covered {N['fur_area_median_pct']:.2f}% of the image. Fine-tuned benchmarks trained at higher resolution can "
     "adapt their early layers to the faint inter-radicular radiolucency; a frozen general-purpose backbone cannot. The "
     "anatomical modules act on top of these features, so they cannot recover detail the features lack. This is an "
     "explanation, not a tested result: the study did not fine-tune the backbone or vary its resolution, and the "
     "foundation-model detectors also trained for fewer epochs than the benchmarks. The comparison is therefore between "
     "two practical recipes, not between architectures at equal compute.")
head("5. A minimum evaluation standard for radiographic furcation detection")
para("We propose that a furcation-detection study should not claim a new or better detector unless it meets the ten "
     "items in Table 5. None is technically demanding; each removes a specific source of optimism that the evidence "
     "above shows to be large. The items extend general reporting guidance^{27,35} to the particular problems of a "
     "rare, small radiographic finding. Table 5 also shows how the present benchmark meets each item, so that the "
     "standard is demonstrably achievable with a public dataset and modest compute.")
STANDARD = [
    ["1. Duplicate audit", "Hash every image; report near-duplicate pairs and how many cross splits.",
     f"{N['near_dup_pairs']} pairs found, {N['cross_split_dups']} across published splits"],
    ["2. Group-disjoint splits", "Keep all copies of a patient/image in one split or fold.",
     f"{N['n_groups']:,} groups, five group-disjoint folds"],
    ["3. No selection on reported data", "Fix weights, thresholds and configurations without the reported test data.",
     "Final-epoch weights; thresholds from calibration groups only"],
    ["4. Pooled or multi-site estimates with intervals", "Report pooled out-of-fold or multi-site AP with a resampling CI.",
     f"Pooled AP over {N['n_pool']:,} images, {N['n_boot']:,}-resample group bootstrap"],
    ["5. Lesion-level recall at stated false-positive rates", "Report recall at fixed false positives per image and the maximum attainable recall.",
     "Recall at 0.1/0.25/0.5 FP per image; conformal feasibility"],
    ["6. Untouched external set", "Score an external set that played no part in any decision.",
     f"{N['ext_images']} clinic images, {N['ext_dup_in_pool']} duplicates of the pool"],
    ["7. Strong fine-tuned baselines", "Compare with current fine-tuned CNN and transformer detectors, not only older ones.",
     "YOLOv8m, YOLO11m, RT-DETR-L"],
    ["8. Paired comparisons", "Compare models on the same resamples and report the difference with its interval.",
     "Paired group bootstrap, P(A>B), folds won, external difference"],
    ["9. Pre-registered rules and ablations", "Fix in advance what counts as an improvement; ablate every new component.",
     "Four criteria fixed before results; three single-module ablations"],
    ["10. Open splits, predictions and code; report negatives", "Release fold assignment, per-image predictions and code; publish failed claims.",
     "All released (Data and Code Availability); negative result reported"],
]
head("6. Where the frontier actually is", 1)
para(f"If the standard is applied, the open problems look different from those suggested by architecture-centred "
     f"papers. First, recall: no model's pooled lesion recall exceeded {f3(maxrec)}, so the binding constraint for a "
     f"screening use is proposal coverage, not ranking. Methods that guarantee coverage, for example tooth-first "
     f"pipelines that examine every molar furcation region, or conformal recall control,^{{34}} are more relevant than a "
     f"further gain in AP. Second, localisation: AP at IoU 0.75 was near zero for every model, so box geometry on "
     f"panoramic radiographs is poorly defined, and region-level or tooth-level outputs may be a better target. Third, "
     f"reference standard: labels without probing or Hamp grade cannot show clinical accuracy; paired panoramic, "
     f"periapical or cone-beam images with clinical charting are needed.^{{2,6,15,16}} Fourth, data: the external set "
     f"held {N['ext_fur']} furcation boxes, too few to rank detectors; multi-site sets with hundreds of lesions are "
     f"required. Fifth, foundation models: a frozen DINOv2 backbone was clearly weaker than fine-tuned detectors here, "
     f"so the open question is whether fine-tuned or higher-resolution foundation backbones help, tested under the "
     f"same standard.")
head("7. Alternative views", 1)
para("One objection is that tooth-level classification studies already report high accuracy, so detection is the "
     "wrong benchmark.^{17–20} Those studies start from a pre-cropped tooth and answer an easier question; a clinical "
     "tool on a full panoramic radiograph must first find the right region, which is exactly where our evidence shows "
     "the losses. A second objection is that our foundation-model result reflects a frozen backbone and a shorter "
     "schedule rather than foundation models as such. We agree, and we draw no conclusion about foundation models in "
     "general; our point is that an architectural addition must beat both its own plain backbone and strong fine-tuned "
     "baselines under the same protocol before it is called progress. A third objection is that pre-registration is too "
     "rigid for engineering research. Our own case suggests the opposite: without fixed rules, the earlier best "
     "checkpoint and our new modules would both have been reported as improvements.")
head("8. Clinical implications", 1)
para("The best detectors find some furcation lesions but miss many, and most boxes are loose. They could at most "
     "prompt a closer look at a molar; they cannot rule furcation involvement out, and they do not grade it. Probing, "
     "and three-dimensional imaging where needed, remain the reference. Published accuracy figures for radiographic "
     "furcation detection should be interpreted alongside the evaluation protocol that produced them.^{15,16,36}")
head("9. Limitations", 1)
para("The evidence comes from a single public dataset with labels of unknown "
     "inter-observer agreement and no Hamp grade or probing reference; an external set with only "
     f"{N['ext_fur']} furcation boxes; one training seed per configuration; unequal input size and training length "
     "between the benchmark and foundation-model families; and no fine-tuned or larger foundation backbone. Additional "
     "seeds, a larger backbone and other schedules were pre-specified only if BiFurca-DINO passed its criteria and were "
     "therefore not run. Every detector kept at most 100 detections per image across all 14 classes; on busy "
     "radiographs this cap can crowd out low-scoring furcation candidates and lower the attainable recall, equally "
     "for all models.")

head("10. Conclusions")
para(f"Automated furcation detection on panoramic radiographs is not yet solved. Under leakage-safe evaluation the best "
     f"current detector reached a furcation AP@50 of {f3(N[bb]['fur'])}, {N[bb]['fur'] / N['old_selected']:.0%} of the "
     f"figure obtained from a single split with checkpoint selection, and no detector's pooled lesion recall exceeded "
     f"{f3(maxrec)}. Our own anatomy-informed detector gave no measurable benefit once tested against pre-registered "
     "criteria. We therefore argue that progress in this area should be judged against a minimum evaluation standard "
     "(Table 5), that strong fine-tuned baselines and negative results belong in every report, and that the real "
     "frontier is lesion coverage, clinical reference standards and multi-site data rather than new architectures.")

# ================================================================== DECLARATIONS
head("Declarations")
para("This research received no external funding.", "Funding:")
para("The author declares no conflict of interest.", "Conflict of interest:")
para("Only a de-identified, publicly available radiograph dataset was used; no ethics approval was required.", "Ethics statement:")
para("The radiographs and annotations analysed in this study are from the public YOLO-Dental dataset (Zenodo, "
     "DOI 10.5281/zenodo.15487429) and are not redistributed here. Derived data needed to reproduce every number, "
     "table and figure are openly available at " + REPO + ": the duplicate-group and fold assignment of every image "
     "(folds.json), the dataset audit summary, the per-image predictions of every model in every fold (calibration, "
     "test and external sets), and the aggregated result files from which the manuscript is generated. The pre-registered "
     "decision rules (PREREGISTERED_DECISIONS.md) are in the same repository.", "Data availability:")
para("All code is openly available at " + REPO + " under the MIT licence: dataset audit and fold construction, "
     "training of all detectors (Modal cloud GPU scripts), the BiFurca-DINO model, evaluation, statistical analysis, "
     "and the scripts that generate the figures, tables and manuscript text from the saved results. Software versions "
     "are pinned in the scripts (PyTorch 2.4.1, torchvision 0.19.1, transformers 4.44.2, Ultralytics 8.3.40). The released "
     "version will be archived on Zenodo with a DOI at acceptance.", "Code availability:")
para("P.K.Y. conceived the study, designed the protocol, ran the analyses and wrote the manuscript.", "Author contributions:")
para("An AI assistant (Claude, Anthropic) was used to help write analysis code and draft text. The author checked the "
     "code, the outputs and the text and takes full responsibility for the content.", "Use of AI tools:")
para("Reported according to TRIPOD+AI; the checklist is provided as a separate file.", "Reporting guideline:")

# ================================================================== REFERENCES
REFS = [
    "Papapanou PN, Sanz M, Buduneli N, et al. Periodontitis: consensus report of workgroup 2 of the 2017 World Workshop on the Classification of Periodontal and Peri-Implant Diseases and Conditions. J Periodontol. 2018;89(Suppl 1):S173–S182.",
    "Hamp SE, Nyman S, Lindhe J. Periodontal treatment of multirooted teeth. Results after 5 years. J Clin Periodontol. 1975;2(3):126–135.",
    "McGuire MK, Nunn ME. Prognosis versus actual outcome. III. The effectiveness of clinical parameters in accurately predicting tooth survival. J Periodontol. 1996;67(7):666–674.",
    "Tonetti MS, Greenwell H, Kornman KS. Staging and grading of periodontitis: framework and proposal of a new classification and case definition. J Periodontol. 2018;89(Suppl 1):S159–S172.",
    "Mol A, van der Stelt PF. Imaging methods in periodontology. Periodontol 2000. 2004;34:34–48.",
    "Graetz C, Plaumann A, Wiebe JF, Springer C, Sälzer S, Dörfer CE. Periodontal probing versus radiographs for the diagnosis of furcation involvement. J Periodontol. 2014;85(10):1371–1379.",
    "Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. Med Image Anal. 2017;42:60–88.",
    "Lee JH, Kim DH, Jeong SN, Choi SH. Detection and diagnosis of dental caries using a deep learning-based convolutional neural network algorithm. J Dent. 2018;77:106–111.",
    "Kunz F, Stellzig-Eisenhauer A, Zeman F, Boldt J. Artificial intelligence in orthodontics: evaluation of a fully automated cephalometric analysis using a customized convolutional neural network. J Orofac Orthop. 2020;81(1):52–68.",
    "Bayraktar Y, Ayan E. Diagnosis of interproximal caries lesions with deep convolutional neural network in digital bitewing radiographs. Clin Oral Investig. 2022;26(1):623–632.",
    "Schwendicke F, Golla T, Dreher M, Krois J. Convolutional neural networks for dental image diagnostics: a scoping review. J Dent. 2019;91:103226.",
    "Krois J, Ekert T, Meinhold L, et al. Deep learning for the radiographic detection of periodontal bone loss. Sci Rep. 2019;9(1):8495.",
    "Orhan K, Bayrakdar IS, Ezhov M, Kravtsov A, Özyürek T. Evaluation of artificial intelligence for detecting periapical pathosis on cone-beam computed tomography scans. Int Endod J. 2020;53(5):680–689.",
    "Endres MG, Hillen F, Salloumis M, et al. Development of a deep learning algorithm for periapical disease detection in dental radiographs. Diagnostics. 2020;10(6):430.",
    "Chatzopoulos GS, Koidou VP, Tsalikis L, Kaklamanos EG. Artificial intelligence for detection and classification of furcation defects using radiographic imaging: a systematic review. Imaging Sci Dent. 2025;55(4):322–334.",
    "Atieh MA, Shah M, Hakam A, et al. Diagnostic accuracy of artificial intelligence-based deep learning models in detecting furcation involvement: a systematic review and meta-analysis. J Periodontol. 2026;97(7):1559–1574.",
    "Mao YC, Huang YC, Chen TY, et al. Deep learning for dental diagnosis: a novel approach to furcation involvement detection on periapical radiographs. Bioengineering (Basel). 2023;10(7):802.",
    "Vilkomir K, Phen C, Baldwin F, Cole J, Herndon N, Zhang W. Classification of mandibular molar furcation involvement in periapical radiographs by deep learning. Imaging Sci Dent. 2024;54(3):257–263.",
    "Kurumlu M, Karacaoglu F, Kalkan M, Ulku I, Akagunduz E, Orhan K. Accuracy of deep learning models in detecting mandibular furcation defects on panoramic radiographs. Diagnostics (Basel). 2026;16(10):1500.",
    "Zhang X, Guo E, Liu X, et al. Enhancing furcation involvement classification on panoramic radiographs with vision transformers. BMC Oral Health. 2025;25:153.",
    "Valente NA, Americo LM, Ciancetta F, Mari S. Deep learning-based detection of periodontal infrabony and furcation defects on periapical radiographs: a feasibility study. Int Dent J. 2026;76(2):109380.",
    "Priyanka, Pathak S, Juneja M, Aggarwal N, Jaiswal M, Rana P. FD-CAD: an AI-based computer-aided diagnosis system for accurate furcation detection in dental radiographs. Biomed Signal Process Control. 2026;125:110735.",
    "Karamüftüoğlu N, Bulut A, Akın M, Sağıroğlu Ş. Panoramic radiograph-based deep learning models for diagnosis and clinical decision support of furcation lesions in primary molars. Children (Basel). 2025;12(11):1517.",
    "Kurt-Bayrakdar S, Bayrakdar İŞ, Yavuz MB, et al. Detection of periodontal bone loss patterns and furcation defects from panoramic radiographs using deep learning algorithm: a retrospective study. BMC Oral Health. 2024;24:155.",
    "Meng K, Lv S, Liu J. Beyond 2D and CNNs: future directions for deep learning in periodontal diagnostics. Int Dent J. 2026;76(4):109672.",
    "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804.",
    "Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. NPJ Digit Med. 2022;5:48.",
    "Oquab M, Darcet T, Moutakanni T, et al. DINOv2: learning robust visual features without supervision. Trans Mach Learn Res. 2024.",
    "Mureșanu S, Hedeșiu M, Iacob LM. Dataset for automating dental condition detection on panoramic radiographs [dataset]. Zenodo; 2025. doi:10.5281/zenodo.15487430. Licence CC BY 4.0.",
    "Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8. GitHub; 2023. https://github.com/ultralytics/ultralytics",
    "Jocher G, Qiu J. Ultralytics YOLO11. GitHub; 2024. https://github.com/ultralytics/ultralytics",
    "Zhao Y, Lv W, Xu S, et al. DETRs beat YOLOs on real-time object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024:16965–16974.",
    "Tian Z, Shen C, Chen H, He T. FCOS: fully convolutional one-stage object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2019:9627–9636.",
    "Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. Found Trends Mach Learn. 2023;16(4):494–591.",
    "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.",
    "Ekert T, Krois J, Meinhold L, et al. Deep learning for the radiographic detection of apical lesions. J Endod. 2019;45(7):917–922.e5.",
]
head("References")
for k, r in enumerate(REFS, 1):
    p = para(f"{k}. {r}"); p.paragraph_format.line_spacing = 1.15

# ================================================================== TABLES + FIGURES
doc.add_page_break(); head("Tables")
table_csv("Table1_dataset_folds", "Table 1. Development folds and external set.",
          "Folds are group-disjoint: all near-duplicates of an image are in the same fold.")
table_csv("Table2_main_results", "Table 2. Furcation detection and other findings, pooled out-of-fold over five group-disjoint folds.",
          f"AP, average precision; CI, {N['n_boot']:,}-resample group bootstrap. Recall at k FP/img: furcation lesion recall "
          "when k false-positive furcation boxes per image are allowed. External: mean of the five fold models on the "
          f"held-out clinic set ({N['ext_images']} images, {N['ext_fur']} furcation boxes); range in brackets.")
table_csv("Table4_paired_comparisons", "Table 3. Paired comparisons of furcation AP@50.",
          "Paired group bootstrap on identical resamples. P(A > B): share of resamples in which A scored higher. "
          "External difference: difference of the external means.")
table_csv("Table3_preregistered_criteria", "Table 4. Pre-registered criteria for claiming a new detector (all four required).")
table_rows(["Item", "Requirement", "How this study met it"], STANDARD,
           "Table 5. Proposed minimum evaluation standard for radiographic furcation detection.",
           "A study that does not meet an item should state so and explain why.")
with open(os.path.join(TAB, "Table5_minimum_evaluation_standard.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["Item", "Requirement", "How this study met it"]] + STANDARD)
doc.add_page_break(); head("Figures")
gc = QN["gt_prepost_checks"]
FIGS = [
    ("Figure1_study_design", "Figure 2. Study design. Duplicate audit, group-disjoint five-fold cross-validation, held-out external clinic set, detectors and the pre-registered evaluation protocol."),
    ("Figure3_ground_truth", f"Figure 3. Ground truth. (A) A randomly selected development radiograph with all annotated findings; furcation lesions in cyan, with a zoom on the furcation box. (B) Location of all {QN['gt']['n']} furcation boxes of the pooled development set at model-input resolution (dashed line: image centre). (C) Furcation box width and height in pixels at model input (median {QN['gt']['med_w']:.0f} × {QN['gt']['med_h']:.0f} px) against the FCOS stride (7 px) and the DINOv2 patch size (14 px). Images: YOLO-Dental (CC BY 4.0)."),
    ("Figure3_furcation_gt_pre_post", f"Figure 4. Furcation ground truth before and after preprocessing, for the four pre-specified random development radiographs used in Figures 8 and 9. (A) Original radiograph at native resolution with the annotated furcation box. (B) Zoom on the furcation with the raw YOLO label line (class 9 = furcation; normalised centre x, centre y, width, height). (C) The same region of the 1,316 × 728-pixel model input with the rescaled box and its size in pixels. (D) Binary furcation mask at model input. (E) Mask overlaid on the full model input. Checks: label files equal the analysed annotations (maximum difference {gc['label_vs_manifest_max_abs']:.1f}); rescaled boxes equal the evaluation boxes (maximum difference < 0.001 px); mask area equals box area; stored inputs differ from a fresh resize of the originals by ≤ {max(gc['resize_mean_absdiff']):.1f} grey levels (JPEG)."),
    ("Figure2_BiFurca_architecture", "Figure 5. BiFurca-DINO. Frozen DINOv2-base features are projected to 256 channels and passed through three switchable modules: (a) a label-free midline-mirror module, (b) arch-canonical coordinates and (c) a coarse context transformer, followed by a three-level pyramid and an FCOS head. Bottom: the midline estimated without labels on a development radiograph by its out-of-fold model, and the left-right similarity profile over candidate columns."),
    ("Figure2_single_split_vs_grouped_cv", f"Figure 1. Instability of single-split estimates. (A) Validation furcation AP@50 of the six saved checkpoints of the earlier FastViT-SA24 + FCOS model on the published validation split ({N['fur_valid_orig']} furcation boxes); the reported value was the maximum, the final epoch is in blue. (B) Furcation AP@50 per fold (dots) and pooled out-of-fold (bar) under grouped cross-validation."),
    ("Figure3_primary_endpoint", "Figure 6. Primary endpoint. Pooled out-of-fold furcation AP@50 with 95% group-bootstrap CI (filled circles) and mean external-set AP@50 of the five fold models (open diamonds). Blue: fine-tuned CNN/DETR benchmarks; orange: frozen-DINOv2 detectors."),
    ("Figure5_paired_differences", "Figure 7. Paired differences in furcation AP@50 (A − B) with 95% paired group-bootstrap CI and the share of resamples in which A scored higher."),
    ("Figure5_qualitative_predictions", f"Figure 8. Out-of-fold predictions on four development radiographs selected at random (seed 0) among those with a furcation lesion, before any output was viewed. Each image is scored by the fold model that never saw it, at the threshold giving 0.25 false-positive furcation boxes per image (RT-DETR-L {QC['thresholds']['rtdetr-l']:.3f}, YOLO11m {QC['thresholds']['yolo11m']:.3f}, BiFurca-DINO {QC['thresholds']['full']:.3f}). Green: true positive (IoU ≥ 0.5) with score; red: false positive; yellow dashed: missed lesion; cyan: ground truth. Crop windows are set by rule around the annotated lesions."),
    ("Figure6_score_maps", "Figure 9. Furcation score maps of the frozen-DINOv2 detectors on the radiographs of Figure 8 (out-of-fold models). Left: ground truth and the label-free midline. Middle and right: the FCOS furcation score √(class probability × centre-ness), the quantity by which boxes are ranked, for BiFurca-DINO and the plain DINOv2 detector; values below 0.05 are transparent. The peak of each map matched the top saved furcation score of the same model within 0.01 wherever that box was retained."),
]
SUPP_FIGS = [
    ("FigureS1_per_class_AP50", "Figure S5. Pooled out-of-fold AP@50 per finding and model."),
    ("FigureS2_sensitivity", "Figure S6. Sensitivity of furcation AP to (A) the IoU matching threshold and (B) lesion size tertile."),
    ("FigureS3_DINO_training_loss", "Figure S2. Training loss of the frozen-DINOv2 detectors (mean of five folds)."),
    ("FigureS4_recall_at_FPPI", "Figure S4. Furcation lesion recall at 0.1, 0.25 and 0.5 false-positive boxes per image."),
    ("FigureS6_cross_split_duplicate", f"Figure S1. A randomly selected near-duplicate pair crossing the published splits: (A) training image, (B) validation image, (C) absolute pixel difference (mean {QN['dup_mean_absdiff']:.2f} grey levels): the two files are identical."),
    ("FigureS7_failure_gallery", f"Figure S9. Failure cases of {NAME[bb]} at 0.25 false-positive boxes per image. (A) The six highest-scoring false positives ({QN['fp_overlapping_gt']} overlap an annotated furcation but fall below 50% overlap); (B) six randomly selected missed lesions. Pooled totals at this threshold: {QN['n_fp_total']} false positives, {QN['n_fn_total']} missed lesions."),
    ("FigureS8_external_score_maps", "Figure S7. Score maps of the frozen-DINOv2 detectors on four random external-clinic radiographs with furcation lesions (fold-0 models), as in Figure 9."),
    ("FigureS9_midline_estimates", f"Figure S8. Label-free midline estimates (red) of BiFurca-DINO on 12 random development radiographs (out-of-fold models); range {QN['midline']['min']:.0f}–{QN['midline']['max']:.0f} px, image centre 658 px."),
    ("Figure4_furcation_PR_curves", "Figure S3. Precision-recall curves for furcation lesions, pooled out-of-fold, IoU 0.5. Pooled AP@50 in brackets."),
]
_num = lambda l: int(re.match(r"Figure S?(\d+)\.", l).group(1))
FIGS.sort(key=lambda x: _num(x[1])); SUPP_FIGS.sort(key=lambda x: _num(x[1]))
assert [_num(l) for _, l in FIGS] == list(range(1, len(FIGS) + 1)) and [_num(l) for _, l in SUPP_FIGS] == list(range(1, len(SUPP_FIGS) + 1))
json.dump(dict(main=[(n, l.split(".")[0]) for n, l in FIGS], supp=[(n, l.split(".")[0]) for n, l in SUPP_FIGS]),
          open(os.path.join(HERE, "figure_map.json"), "w"), indent=1)
for n, l in FIGS:
    figure(n, l)
doc.save(os.path.join(OUT, "Furcation_Benchmark_Manuscript.docx"))

# ================================================================== SUPPLEMENT
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(0.8)
p = doc.add_paragraph(); r = p.add_run("Supplementary Material\n" + TITLE); r.bold = True
SUPP_TABLES = [
    ("TableS1_per_class_AP50", "Table S5. Pooled out-of-fold AP@50 for each of the 14 findings.", "—: no boxes of this class in the pooled set."),
    ("TableS2_sensitivity", "Table S6. Sensitivity analyses of furcation AP (computed from saved predictions; no retraining).",
     "IoU, intersection-over-union threshold for a match. Duplicate-group images excluded: every image that has a near-duplicate is removed. "
     "Size tertiles: furcation boxes split into thirds by area; AP computed against one tertile at a time."),
    ("TableS3_per_fold_AP50", "Table S3. Furcation AP@50 in each fold.", None),
    ("TableS4_conformal", "Table S4. Split-conformal threshold for lesion recall, calibrated on each fold's calibration groups.",
     "Infeasible: in that fold, more than the allowed share of calibration lesions had no overlapping prediction at any score, so no threshold can reach the target recall."),
    ("TableS5_old_single_split_checkpoints", "Table S1. Earlier single-split analysis: validation furcation AP@50 at each saved checkpoint.",
     f"Source: checkpoint history of the earlier FastViT-SA24 + FCOS run on the published validation split ({N['fur_valid_orig']} furcation boxes)."),
    ("TableS6_training_configuration", "Table S2. Training configuration.", None),
]
SUPP_TABLES.sort(key=lambda x: int(re.match(r"Table S(\d+)\.", x[1]).group(1)))
for n, cap, note in SUPP_TABLES:
    table_csv(n, cap, note, doc)
json.dump({"supp_tables": [(n, c.split(".")[0]) for n, c, _ in SUPP_TABLES],
           "main_tables": [("Table1_dataset_folds", "Table 1"), ("Table2_main_results", "Table 2"), ("Table4_paired_comparisons", "Table 3"),
                           ("Table3_preregistered_criteria", "Table 4"), ("Table5_minimum_evaluation_standard", "Table 5")]},
          open(os.path.join(HERE, "table_map.json"), "w"), indent=1)
for n, l in SUPP_FIGS:
    figure(n, l, doc, width=6.0)
doc.save(os.path.join(OUT, "Furcation_Benchmark_Supplementary.docx"))
print("built", os.listdir(OUT), "| ablations in text:", abl_methods_done)
