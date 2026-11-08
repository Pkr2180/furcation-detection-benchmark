"""Copy the public release into ../furcation-detection-benchmark (code, folds, predictions gzipped, results, figures,
tables). Excludes: raw images and manifest.json (dataset labels; rebuilt by modal_data_audit.py from Zenodo), logs,
Modal account tooling and the manuscript .docx.   python paper/export_release.py
"""
import glob, gzip, os, shutil

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(os.path.dirname(SRC), "furcation-detection-benchmark")
os.makedirs(DST, exist_ok=True)

FILES = ["analyze.py", "sensitivity.py", "modal_data_audit.py", "modal_make_folds.py", "modal_train.py", "modal_yolo.py",
         "spawn_dino.py", "modal_explain.py", "folds.json", "folds_summary_local.json", "data_audit_local.json", "PREREGISTERED_DECISIONS.md",
         "res_final.json", "sens_final.json"]
for f in FILES:
    shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
for d in ("furcadino", "tests"):
    shutil.copytree(os.path.join(SRC, d), os.path.join(DST, d), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
P = os.path.join(DST, "paper"); os.makedirs(P, exist_ok=True)
for f in ("stats_extra.py", "make_figs_tables.py", "build_manuscript.py", "audit_manuscript.py", "finalize.ps1",
          "export_release.py", "numbers.json", "stats_extra.json", "package_submission.py", "qual_select.py",
          "make_qual_figs.py", "qual_cases.json", "qual_numbers.json", "figure_map.json", "table_map.json"):
    shutil.copy2(os.path.join(SRC, "paper", f), os.path.join(P, f))
shutil.copytree(os.path.join(SRC, "paper", "tables"), os.path.join(P, "tables"), dirs_exist_ok=True)
os.makedirs(os.path.join(P, "figures"), exist_ok=True)
for f in glob.glob(os.path.join(SRC, "paper", "figures", "*.png")):
    shutil.copy2(f, os.path.join(P, "figures", os.path.basename(f)))
n = 0
for f in glob.glob(os.path.join(SRC, "res_all", "*", "fold*_seed0.json")):
    v = os.path.basename(os.path.dirname(f))
    out = os.path.join(DST, "predictions", v, os.path.basename(f) + ".gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(f, "rb") as a, gzip.open(out, "wb", compresslevel=9) as b:
        shutil.copyfileobj(a, b)
    n += 1
print("exported to", DST, "| prediction files:", n)
