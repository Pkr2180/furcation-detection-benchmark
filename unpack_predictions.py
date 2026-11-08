"""Unpack predictions/<model>/fold*_seed0.json.gz into res_all/<model>/ for analyze.py and sensitivity.py."""
import glob, gzip, os, shutil

for f in glob.glob(os.path.join("predictions", "*", "*.json.gz")):
    out = os.path.join("res_all", os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-3])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(f, "rb") as a, open(out, "wb") as b:
        shutil.copyfileobj(a, b)
print("unpacked into res_all/")
