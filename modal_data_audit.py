"""Stage 0: download YOLO-Dental (Zenodo 15487430), audit structure, labels, duplicates. CPU only.
Run: modal run modal_data_audit.py   (profile pradeepaiperio)
Writes /data/audit/data_audit.json to volume furcation-v4 and prints a summary.
"""
import modal

app = modal.App("furcation-v4-data-audit")
vol = modal.Volume.from_name("furcation-v4", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("curl", "unzip", "libgl1", "libglib2.0-0")
         .pip_install("numpy==1.26.4", "pillow==10.3.0", "imagehash==4.3.1", "pandas==2.2.2"))

URL = ("https://zenodo.org/records/15487430/files/"
       "panoramic_radiography_yolo_dataset_14_classes.zip?download=1")
NAMES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT", "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=16384)
def audit():
    import json, os, subprocess, glob, collections
    from pathlib import Path
    import numpy as np, imagehash
    from PIL import Image

    root = Path("/data/raw")
    if not any(root.glob("**/*.jpg")) and not any(root.glob("**/*.png")):
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-L", "-sS", "-o", "/data/yd.zip", URL], check=True)
        subprocess.run(["unzip", "-q", "-o", "/data/yd.zip", "-d", str(root)], check=True)
        os.remove("/data/yd.zip")
        vol.commit()

    tree = collections.Counter()
    for p in root.rglob("*"):
        if p.is_file():
            tree[(str(p.parent.relative_to(root)), p.suffix.lower())] += 1
    out = {"tree": {f"{k[0]}|{k[1]}": v for k, v in sorted(tree.items())}}

    imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    out["n_images"] = len(imgs)
    rows = []
    for p in imgs:
        lab = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        boxes = []
        if lab.exists():
            for ln in lab.read_text().split("\n"):
                s = ln.split()
                if len(s) >= 5:
                    boxes.append((int(float(s[0])), *map(float, s[1:5])))
        im = Image.open(p)
        rows.append(dict(path=str(p.relative_to(root)), split=str(p.relative_to(root)).split("/")[0:3],
                         w=im.size[0], h=im.size[1], boxes=boxes,
                         phash=str(imagehash.phash(im.convert("L"), hash_size=16))))
    # per split-ish directory
    by_dir = collections.defaultdict(list)
    for r in rows:
        by_dir[str(Path(r["path"]).parent)].append(r)
    summ = {}
    for d, rs in by_dir.items():
        c = collections.Counter(b[0] for r in rs for b in r["boxes"])
        imgs_with = collections.Counter(k for r in rs for k in set(b[0] for b in r["boxes"]))
        summ[d] = dict(n_images=len(rs), n_label_missing=sum(1 for r in rs if not r["boxes"]),
                       boxes_per_class={NAMES[k]: v for k, v in sorted(c.items()) if k < 14},
                       images_with_class={NAMES[k]: v for k, v in sorted(imgs_with.items()) if k < 14},
                       sizes=dict(collections.Counter(f"{r['w']}x{r['h']}" for r in rs).most_common(4)))
    out["by_dir"] = summ

    fur = [b for r in rows for b in r["boxes"] if b[0] == 9]
    if fur:
        a = np.array([[b[3] * b[4]] for b in fur])
        out["fur_box_area_frac"] = dict(n=len(fur), median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                                        p90=float(np.percentile(a, 90)))
    # duplicates (hamming <= 8 of 256 bits) across and within directories
    hs = [imagehash.hex_to_hash(r["phash"]) for r in rows]
    dup = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hs[i] - hs[j] <= 8:
                dup.append((rows[i]["path"], rows[j]["path"], int(hs[i] - hs[j])))
    out["near_duplicate_pairs"] = len(dup)
    out["near_duplicate_examples"] = dup[:15]
    out["cross_dir_dups"] = sum(1 for a, b, _ in dup if str(Path(a).parent).replace("images", "") != str(Path(b).parent).replace("images", ""))
    Path("/data/audit").mkdir(exist_ok=True)
    json.dump(out, open("/data/audit/data_audit.json", "w"), indent=1)
    json.dump([dict(path=r["path"], w=r["w"], h=r["h"], boxes=r["boxes"], phash=r["phash"]) for r in rows],
              open("/data/audit/manifest.json", "w"))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    o = audit.remote()
    print(json.dumps(o, indent=1)[:6000])
    open("data_audit_local.json", "w").write(json.dumps(o, indent=1))
