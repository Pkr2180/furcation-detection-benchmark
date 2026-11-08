# Furcation detection benchmark (leakage-safe, pre-registered)

Code, fold assignment, per-image predictions and result files for the position paper

> **Automated furcation detection on panoramic radiographs is not yet solved: a position paper on leakage-safe evaluation, supported by a pre-registered benchmark**
> Pradeep Kumar Yadalam, Saveetha Dental College and Hospitals, SIMATS, Chennai, India.

Every number, table and figure in the paper is generated from the files in this repository. None is typed by hand.

## What is here

| Path | Contents |
|---|---|
| `PREREGISTERED_DECISIONS.md` | Endpoint, protocol and decision rules, written before any BiFurca-DINO result was seen |
| `modal_data_audit.py` | Downloads YOLO-Dental from Zenodo, audits labels, perceptual-hash near-duplicates |
| `modal_make_folds.py` | Duplicate groups → five group-disjoint folds + calibration groups; held-out external set |
| `folds.json` | Fold, duplicate group and original split of every image (no labels) |
| `modal_yolo.py` | YOLOv8m / YOLO11m / RT-DETR-L training and prediction (Ultralytics 8.3.40) |
| `furcadino/model.py` | BiFurca-DINO: frozen DINOv2 + midline-mirror attention + arch coordinates + context + FCOS |
| `modal_train.py`, `spawn_dino.py` | DINOv2 detector training (variants `full`, `plain`, `no_mirror`, `no_coords`, `no_context`) |
| `furcadino/evalap.py` | AP (COCO 101-point), recall at FP/image, group bootstrap, paired bootstrap, split-conformal recall |
| `analyze.py`, `sensitivity.py` | Primary analysis and pre-registered sensitivity analyses |
| `predictions/<model>/fold*_seed0.json.gz` | Per-image boxes, scores and labels for test, calibration and external images of every fold |
| `res_final.json`, `sens_final.json`, `paper/stats_extra.json` | Aggregated results |
| `paper/` | Scripts that build every figure, table and manuscript number (`numbers.json`), plus the number audit |
| `tests/` | Unit tests for the metrics and the model |

## Data

Radiographs and annotations come from the public **YOLO-Dental** dataset (Zenodo, DOI [10.5281/zenodo.15487429](https://doi.org/10.5281/zenodo.15487429)). They are not redistributed here. `modal_data_audit.py` downloads them and rebuilds `manifest.json` (image paths and boxes), which the analysis scripts need.

## Reproduce the analysis from the saved predictions (CPU only)

```bash
pip install -r requirements.txt
python unpack_predictions.py                     # predictions/*.json.gz -> res_all/
# manifest.json: run `modal run modal_data_audit.py`, then `modal volume get furcation-v4 /audit/manifest.json .`
python analyze.py res_all manifest.json res_final.json
python sensitivity.py res_all manifest.json folds.json sens_final.json
python paper/stats_extra.py res_all manifest.json paper/stats_extra.json
python paper/make_figs_tables.py                 # figures, tables, numbers.json
python -m pytest tests -q
```

## Re-train (Modal GPU, A100-40GB)

```bash
modal run modal_data_audit.py && modal run modal_make_folds.py && modal run modal_train.py::prep
modal run modal_yolo.py::all_baselines           # 3 benchmarks x 5 folds, 60 epochs
modal deploy modal_train.py && python spawn_dino.py --epochs 20   # 5 DINOv2 variants x 5 folds
```

All models are evaluated with final-epoch weights; nothing is selected on test data.

## Headline result

Pooled out-of-fold furcation AP@50 (1,628 images, 309 furcation boxes, 2,000-resample group bootstrap):
RT-DETR-L 0.220, YOLO11m 0.207, YOLOv8m 0.160, plain DINOv2-FCOS 0.100, BiFurca-DINO 0.100.
BiFurca-DINO failed all four pre-registered criteria, so the new-architecture claim is withdrawn. Ablations (full minus ablated):
mirror module +0.016 (95% CI −0.019 to +0.050), arch coordinates +0.014 (−0.014 to +0.045), context transformer +0.007 (−0.034 to +0.048);
no component made a detectable contribution. Full tables are in `paper/tables/`; `paper/tables/Table5_minimum_evaluation_standard.csv` is the proposed ten-item evaluation standard.

## Licence

Code: MIT (see `LICENSE`). Derived data files (folds, predictions, results): CC BY 4.0. The YOLO-Dental dataset keeps its own licence.
