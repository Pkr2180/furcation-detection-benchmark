"""Model internals for the qualitative figures: label-free midline estimate and furcation score map of the DINOv2
detectors, computed with each image's OUT-OF-FOLD model (the fold model that never trained on it).
  modal run modal_explain.py      (profile saveetha; needs /ckpt/<variant>_f<k>_s0.pt and paper/qual_cases.json)
Writes /data/explain/<variant>/<image>.npz (heat = max over FCOS levels of sqrt(sigmoid(cls_FUR) * sigmoid(ctr)),
the score FCOS ranks boxes by; midline_px; similarity profile) and downloads nothing itself.
"""
import json
import modal

app = modal.App("furcation-v4-explain")
vol = modal.Volume.from_name("furcation-v4")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("transformers==4.44.2", "numpy==1.26.4", "pillow==10.3.0", "scipy==1.13.0")
         .run_commands("python -c \"from transformers import Dinov2Model; Dinov2Model.from_pretrained('facebook/dinov2-base')\"")
         .add_local_dir("furcadino", remote_path="/root/furcadino"))
SIZE = (728, 1316)
FUR_LABEL = 10  # class id 9 + 1 (label 0 unused)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=1800)
def explain(variant: str, fold: int, paths: list):
    import os, sys
    import numpy as np, torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, "/root")
    from furcadino.model import build_detector
    model = build_detector(variant, size=SIZE).cuda().eval()
    st = torch.load(f"/data/ckpt/{variant}_f{fold}_s0.pt", map_location="cuda")
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    assert not unexpected and all(k.startswith("backbone.dino.model.") for k in missing), (unexpected, missing[:5])
    os.makedirs(f"/data/explain/{variant}", exist_ok=True)
    out = []
    with torch.no_grad():
        for p in paths:
            im = Image.open("/data/resized/" + p.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
            x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).cuda()
            images, _ = model.transform([x])
            feats = list(model.backbone(images.tensors).values())
            c = model.backbone.last_midline.float().item()
            raw = model.backbone.dino(images.tensors)
            _, sim, cands = model.backbone.midline(raw)
            head = model.head(feats)
            cls, ctr = head["cls_logits"][0], head["bbox_ctrness"][0]
            maps, start = [], 0
            for f in feats:
                h, w = f.shape[-2:]
                n = h * w
                s = torch.sqrt(torch.sigmoid(cls[start:start + n, FUR_LABEL]) * torch.sigmoid(ctr[start:start + n, 0]))
                maps.append(F.interpolate(s.view(1, 1, h, w), size=SIZE, mode="bilinear", align_corners=False)[0, 0])
                start += n
            heat = torch.stack(maps).max(0).values.cpu().numpy().astype(np.float16)
            stride = SIZE[1] / raw.shape[-1]
            name = p.replace("/", "__").rsplit(".", 1)[0]
            np.savez_compressed(f"/data/explain/{variant}/{name}.npz", heat=heat, midline_px=np.float32((c + 0.5) * stride),
                                sim=sim[0].float().cpu().numpy(), cand_px=((cands.float() + 0.5) * stride).cpu().numpy(), fold=fold)
            out.append((p, round((c + 0.5) * stride, 1), float(heat.max())))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    q = json.load(open("paper/qual_cases.json"))
    ext = q["external"]
    paths = sorted(set(q["main"] + q["midline"] + [q["gt_example"]]))
    jobs = []
    for v in ("full", "plain"):
        for k in range(5):
            ps = [p for p in paths if q["fold_of"][p] == k] + (ext if k == 0 else [])  # external: fold-0 model
            if ps:
                jobs.append((v, k, ps))
    for r in explain.starmap(jobs):
        for row in r:
            print(row)
