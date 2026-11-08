# Decisions fixed before any BiFurca-DINO result was seen (2026-11-05, written 13:05 IST)

Primary endpoint: pooled out-of-fold furcation AP@50 (5 group-disjoint folds, 1,628 images, 309 boxes), group-bootstrap 95% CI.
Final-epoch weights, no checkpoint selection. Seed 0 for the main grid.

"Really good" (launch the extended sensitivity runs) requires ALL of:
1. BiFurca (full) beats the plain DINOv2 detector on the primary endpoint with paired group-bootstrap P(diff > 0) >= 0.95.
2. BiFurca beats the best YOLO/RT-DETR benchmark on the same endpoint with P >= 0.95.
3. The direction holds on the external clinic set (mean over fold models) and in at least 4 of 5 folds.
4. Other-class mAP@50 of BiFurca is not more than 0.02 below the best benchmark.

If 1 holds but 2 does not: report as "a better DINOv2 head, not better than the best CNN" (benchmark paper).
If 1 does not hold: the architecture claim is withdrawn; the paper becomes a leak-safe benchmark of furcation detectors.
Ablations (no_mirror, no_coords, no_context vs full) are already part of the main grid and are reported either way.

Sensitivity analyses (run on saved predictions, no new training): IoU 0.3/0.5/0.75; furcation box-size tertiles;
exclude every image that has a near-duplicate; image-level instead of group-level bootstrap; conformal matching IoU 0.3 vs 0.5.
Sensitivity needing new training (only if "really good"): seeds 1 and 2 for full and plain; DINOv2-large backbone; 10/30 epochs.
