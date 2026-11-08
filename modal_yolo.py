"""Benchmark detectors on the SAME grouped folds, calibration sets and external set as BiFurca-DINO.
  modal run modal_yolo.py --model yolov8m --fold 0 --seed 0 --epochs 60
Models: yolov8m | yolo11m | rtdetr-l  (COCO-pretrained, full fine-tuning, final-epoch weights, no checkpoint selection).
Predictions are written in the 1316x728 frame used for ground truth in the analysis.
"""
import modal

app = modal.App("furcation-v4-yolo")
vol = modal.Volume.from_name("furcation-v4", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0", "wget")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("ultralytics==8.3.40", "numpy==1.26.4", "opencv-python-headless==4.10.0.84")
         .run_commands("cd /root && for m in yolov8m yolo11m rtdetr-l; do wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/$m.pt || true; done; ls -la /root/*.pt"))

SIZE = (728, 1316)
NAMES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT", "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]


@app.function(image=image, volumes={"/data": vol}, gpu="A100-40GB", timeout=60 * 60 * 6, memory=32768, cpu=8)
def run(model="yolov8m", fold=0, seed=0, epochs=60, imgsz=1280, batch=8, tag=""):
    import json, os, sys, numpy as np
    from pathlib import Path
    sys.path.insert(0, "/root")
    d = json.load(open("/data/audit/folds.json"))
    pool = d["pool"]
    rest = [r for r in pool if r["fold"] != fold]
    test = [r for r in pool if r["fold"] == fold]
    rng = np.random.default_rng(1000 + fold)  # identical calibration split to modal_train.py
    groups = sorted({r["group"] for r in rest}); rng.shuffle(groups)
    cal_g = set(groups[: int(round(0.12 * len(groups)))])
    cal = [r for r in rest if r["group"] in cal_g]
    train = [r for r in rest if r["group"] not in cal_g]
    w = Path(f"/data/yolo_lists/{model}_f{fold}_s{seed}{tag}"); w.mkdir(parents=True, exist_ok=True)
    for nm, rows in (("train", train), ("cal", cal)):
        (w / f"{nm}.txt").write_text("\n".join(f"/data/raw/{r['path']}" for r in rows))
    (w / "data.yaml").write_text(f"path: {w}\ntrain: train.txt\nval: cal.txt\nnames:\n" +
                                 "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES)))
    from ultralytics import YOLO, RTDETR
    cls = RTDETR if model.startswith("rtdetr") else YOLO
    m = cls(f"/root/{model}.pt")
    m.train(data=str(w / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, deterministic=True,
            val=False, workers=8, cos_lr=True, patience=0, project=f"/data/yolo_runs/{model}{tag}",
            name=f"f{fold}_s{seed}", exist_ok=True, plots=False, verbose=False, amp=True)
    last = f"/data/yolo_runs/{model}{tag}/f{fold}_s{seed}/weights/last.pt"
    best = cls(last)

    def pred(rows, prefix="/data/raw/"):
        out = {}
        paths = [prefix + r["path"] for r in rows]
        for i in range(0, len(paths), 16):
            for p, r in zip(paths[i:i + 16], best.predict(paths[i:i + 16], imgsz=imgsz, conf=0.001, iou=0.6, max_det=100,
                                                         verbose=False, half=True)):
                h, wd = r.orig_shape
                b = r.boxes.xyxy.cpu().numpy() * np.array([SIZE[1] / wd, SIZE[0] / h, SIZE[1] / wd, SIZE[0] / h])
                out[p[len(prefix):]] = dict(boxes=b.round(1).tolist(), scores=r.boxes.conf.cpu().numpy().round(4).tolist(),
                                            labels=r.boxes.cls.cpu().numpy().astype(int).tolist())
        return out

    res = dict(variant=model, fold=fold, seed=seed, epochs=epochs, n_train=len(train), n_cal=len(cal), n_test=len(test),
               test=pred(test), cal=pred(cal), external=pred([dict(path=e["path"]) for e in d["external"]]),
               test_groups={r["path"]: r["group"] for r in test}, cal_groups={r["path"]: r["group"] for r in cal})
    out = f"/data/results/{model}{tag}"; os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/fold{fold}_seed{seed}.json", "w"))
    vol.commit()
    return dict(model=model, fold=fold, n_train=len(train))


@app.local_entrypoint()
def smoke():
    print(run.remote(model="yolov8m", fold=0, seed=0, epochs=1, tag="_smoke"))


@app.local_entrypoint()
def main(model: str = "yolov8m", fold: int = 0, seed: int = 0, epochs: int = 60):
    print(run.remote(model=model, fold=fold, seed=seed, epochs=epochs))


@app.local_entrypoint()
def all_baselines(models: str = "yolov8m,yolo11m,rtdetr-l", folds: str = "0,1,2,3,4", seed: int = 0, epochs: int = 60,
                  skip: str = ""):
    """Queue every (model, fold) as its own container; Modal runs as many in parallel as the account allows.
    skip: comma list like 'yolov8m:0' for runs that already exist."""
    skipset = set(skip.split(",")) if skip else set()
    jobs = [(m, int(f), seed, epochs) for m in models.split(",") for f in folds.split(",") if f"{m}:{f}" not in skipset]
    print(len(jobs), "jobs")
    for r in run.starmap(jobs):
        print(r, flush=True)
