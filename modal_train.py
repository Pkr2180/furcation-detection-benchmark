"""Stage 2/3: train BiFurca-DINO variants (full | plain | no_mirror | no_coords | no_context) on grouped CV folds.
  modal run modal_train.py::prep                      # resize all images once to 1316x728
  modal run modal_train.py::smoke                      # 1 fold, 1 epoch, timing
  modal run modal_train.py --variant full --fold 0 --seed 0 --epochs 30
Pre-registered protocol: final-epoch weights (no checkpoint selection); test fold untouched until the end; conformal
calibration set = 12% of non-test groups, never used for training.
"""
import modal

app = modal.App("furcation-v4-train")
vol = modal.Volume.from_name("furcation-v4", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("transformers==4.44.2", "numpy==1.26.4", "pillow==10.3.0", "scipy==1.13.0")
         .run_commands("python -c \"from transformers import Dinov2Model; Dinov2Model.from_pretrained('facebook/dinov2-base'); "
                       "Dinov2Model.from_pretrained('facebook/dinov2-large')\"")
         .add_local_dir("furcadino", remote_path="/root/furcadino"))

SIZE = (728, 1316)
FUR = 9  # 0-based class id; model label = id + 1


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=8, memory=16384)
def prep():
    import json, os
    from pathlib import Path
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    man = json.load(open("/data/audit/manifest.json"))
    out = Path("/data/resized"); out.mkdir(exist_ok=True)

    def one(m):
        dst = out / (m["path"].replace("/", "__").rsplit(".", 1)[0] + ".jpg")
        if dst.exists():
            return
        im = Image.open(f"/data/raw/{m['path']}").convert("RGB").resize((SIZE[1], SIZE[0]), Image.BILINEAR)
        im.save(dst, quality=95)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, man))
    vol.commit()
    return len(man)


def _load_split(fold, seed):
    import json, numpy as np
    d = json.load(open("/data/audit/folds.json"))
    man = {m["path"]: m for m in json.load(open("/data/audit/manifest.json"))}
    pool = d["pool"]
    test = [r for r in pool if r["fold"] == fold]
    rest = [r for r in pool if r["fold"] != fold]
    rng = np.random.default_rng(1000 + fold)  # calibration split depends on the fold only, not on the model seed
    groups = sorted({r["group"] for r in rest})
    rng.shuffle(groups)
    cal_g = set(groups[: int(round(0.12 * len(groups)))])
    cal = [r for r in rest if r["group"] in cal_g]
    train = [r for r in rest if r["group"] not in cal_g]
    return man, train, cal, test, d["external"]


class DS:
    def __init__(self, man, rows, train, ext=False):
        self.man, self.rows, self.train, self.ext = man, rows, train, ext

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import numpy as np, torch
        from PIL import Image
        r = self.rows[i]
        p = r["path"]
        im = Image.open("/data/resized/" + p.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        H, W = SIZE
        b = torch.tensor([[(c[1] - c[3] / 2) * W, (c[2] - c[4] / 2) * H, (c[1] + c[3] / 2) * W, (c[2] + c[4] / 2) * H]
                          for c in self.man[p]["boxes"]], dtype=torch.float32).reshape(-1, 4)
        lab = torch.tensor([c[0] + 1 for c in self.man[p]["boxes"]], dtype=torch.int64)
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
        keep = ((b[:, 2] - b[:, 0]) >= 2) & ((b[:, 3] - b[:, 1]) >= 2)  # drop degenerate annotations (<2 px)
        b, lab = b[keep], lab[keep]
        if self.train:
            if torch.rand(1) < 0.5:
                x = x.flip(-1); b = b[:, [2, 1, 0, 3]].clone(); b[:, 0] = W - b[:, 0]; b[:, 2] = W - b[:, 2]
            x = (x * (1 + 0.2 * (torch.rand(1) - 0.5)) + 0.1 * (torch.rand(1) - 0.5)).clamp(0, 1)
        return x, {"boxes": b, "labels": lab}, p


def _collate(batch):
    return tuple(zip(*batch))


def _predict(model, rows, man, device, bs=4):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(DS(man, rows, False), batch_size=bs, shuffle=False, num_workers=4, collate_fn=_collate)
    model.eval(); out = {}
    with torch.no_grad():
        for xs, ts, ps in dl:
            res = model([x.to(device) for x in xs])
            for r, p in zip(res, ps):
                out[p] = dict(boxes=r["boxes"].float().cpu().numpy().round(1).tolist(),
                              scores=r["scores"].float().cpu().numpy().round(4).tolist(),
                              labels=(r["labels"].cpu().numpy() - 1).tolist())
    return out


@app.function(image=image, volumes={"/data": vol}, gpu="A100-40GB", timeout=60 * 60 * 6, memory=32768, cpu=8)
def train(variant="full", fold=0, seed=0, epochs=30, dino="facebook/dinov2-base", lr=2e-4, bs=4, max_train=0, tag=""):
    import json, time, math, os, sys
    import numpy as np, torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, "/root")
    from furcadino.model import build_detector
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    man, train_rows, cal, test, ext = _load_split(fold, seed)
    if max_train:
        train_rows = train_rows[:max_train]
    model = build_detector(variant, dino_name=dino, size=SIZE).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    dl = DataLoader(DS(man, train_rows, True), batch_size=bs, shuffle=True, num_workers=6, collate_fn=_collate,
                    drop_last=True, persistent_workers=True)
    total = epochs * len(dl); step = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / total)))
    log = []
    t0 = time.time()
    # resume: trainable weights + optimiser + scheduler are checkpointed to the volume every epoch, so a stopped
    # workspace loses at most one epoch (the first full grid was lost at epoch 16 of 20 when a workspace was disabled)
    ck = f"/data/ckpt/{variant}{tag}_f{fold}_s{seed}.pt"
    start = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"], strict=False); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, log = st["epoch"], st["log"]; step = start * len(dl)
        print(f"resumed {ck} at epoch {start}", flush=True)
    for ep in range(start, epochs):
        model.train(); tl = 0; n = 0
        for xs, ts, _ in dl:
            xs = [x.to(dev) for x in xs]
            ts = [{k: v.to(dev) for k, v in t.items()} for t in ts]
            losses = model(xs, ts)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                bad = {k: float(v) for k, v in losses.items()}
                wh = [((t["boxes"][:, 2] - t["boxes"][:, 0]).min().item() if len(t["boxes"]) else 9, (t["boxes"][:, 3] - t["boxes"][:, 1]).min().item() if len(t["boxes"]) else 9) for t in ts]
                raise RuntimeError(f"non-finite loss at step {step}: {bad}; min box wh {wh}; midline {model.backbone.last_midline.tolist()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step(); step += 1
            tl += float(loss); n += 1
        log.append(dict(epoch=ep + 1, loss=tl / max(n, 1), sec=round(time.time() - t0)))
        print(log[-1], flush=True)
        os.makedirs("/data/ckpt", exist_ok=True)
        trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.dino.model.")}
        torch.save(dict(model=trainable, opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, log=log), ck + ".tmp")
        os.replace(ck + ".tmp", ck); vol.commit()
    res = dict(variant=variant, fold=fold, seed=seed, epochs=epochs, dino=dino, log=log,
               n_train=len(train_rows), n_cal=len(cal), n_test=len(test),
               midline_examples=None)
    res["test"] = _predict(model, test, man, dev)
    res["cal"] = _predict(model, cal, man, dev)
    ext_rows = [dict(path=e["path"]) for e in ext]
    res["external"] = _predict(model, ext_rows, man, dev)
    res["test_groups"] = {r["path"]: r["group"] for r in test}
    res["cal_groups"] = {r["path"]: r["group"] for r in cal}
    d = f"/data/results/{variant}{tag}"
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(f"{d}/fold{fold}_seed{seed}.json", "w"))
    vol.commit()
    return dict(variant=variant, fold=fold, seed=seed, final_loss=log[-1]["loss"], sec=log[-1]["sec"], n_train=len(train_rows))


@app.local_entrypoint()
def smoke():
    print(train.remote(variant="full", fold=0, seed=0, epochs=1, max_train=48, tag="_smoke"))


@app.local_entrypoint()
def main(variant: str = "full", fold: int = 0, seed: int = 0, epochs: int = 30):
    print(train.remote(variant=variant, fold=fold, seed=seed, epochs=epochs))


@app.local_entrypoint()
def all_dino(variants: str = "full,plain,no_mirror,no_coords,no_context", folds: str = "0,1,2,3,4", seed: int = 0,
             epochs: int = 20, skip: str = ""):
    """Queue every (variant, fold). skip: comma list like 'full:0,plain:0' for runs that already exist."""
    skipset = set(skip.split(",")) if skip else set()
    jobs = [(v, int(f), seed, epochs) for v in variants.split(",") for f in folds.split(",") if f"{v}:{f}" not in skipset]
    print(len(jobs), "jobs")
    for r in train.starmap(jobs):
        print(r, flush=True)


@app.local_entrypoint()
def overfit(variant: str = "full", n: int = 96, epochs: int = 25):
    print(train.remote(variant=variant, fold=0, seed=0, epochs=epochs, max_train=n, tag="_overfit"))
