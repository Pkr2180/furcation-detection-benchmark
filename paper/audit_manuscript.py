"""Audit: every number in the manuscript/supplement must trace to numbers.json, a table CSV or a fixed protocol constant;
and the headline values are recomputed from res_final.json and compared.  python paper/audit_manuscript.py"""
import csv, glob, json, os, re, sys
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
N = json.load(open(os.path.join(HERE, "numbers.json"))); RES = json.load(open(os.path.join(ROOT, "res_final.json")))
fails, checks = [], 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


# 1. allowed-number set from numbers.json and tables
allowed = set()


def walk(o):
    if isinstance(o, dict):
        [walk(v) for v in o.values()]
    elif isinstance(o, list):
        [walk(v) for v in o]
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        for fmt in ("{:.4f}", "{:.3f}", "{:+.3f}", "{:.2f}", "{:.1f}", "{:.0f}", "{:,}", "{}"):
            try:
                allowed.add(fmt.format(o).lstrip("+").replace("-", "−") if False else fmt.format(o).lstrip("+"))
                allowed.add(fmt.format(-o).lstrip("+"))
            except (ValueError, TypeError):
                pass
        if isinstance(o, float) and abs(o) < 100:
            allowed.add(f"{o:.2f}")


walk(N)
QN = json.load(open(os.path.join(HERE, "qual_numbers.json"))); walk(QN)
QC = json.load(open(os.path.join(HERE, "qual_cases.json"))); walk(QC["thresholds"])
allowed |= {str(len(QC["main"])), str(sum(QN["qual_counts"]["rtdetr-l"][k] for k in (0, 2))), "0.001", "0.05", "0.01", "658", "1,316",
            "9", "8", "7", "0.0", "256"}
# literature values quoted from verified references (PubMed abstracts): Atieh 2026 pooled sensitivity/specificity,
# tooth-level accuracy range (Zhang 92%, Mao 94.97%, Vilkomir 96.47%, Kurumlu 97.9%), Valente 2026 furcation AP@50,
# Karamuftuoglu 2025 mAP@50; protocol date
allowed |= {"0.93", "0.94", "92", "98", "0.577", "0.434", "2026", "5"}
for f in glob.glob(os.path.join(HERE, "tables", "*.csv")):
    for row in csv.reader(open(f, encoding="utf-8")):
        for c in row:
            allowed.update(x.lstrip("+") for x in re.findall(r"[-+]?\d[\d,]*\.?\d*", c))
# derived values used in text
S = N["sens"]; bb = N["best_benchmark"]
allowed.add(str(N["n_pool"] - S[bb]["fur_ap50_excluding_dup_images"]["n_images"]))
MSt = QN["map_stats"]; allowed.add(f"{MSt['mass_in_box_full_median'] / MSt['box_area_share_median']:.0f}")
leak = N[bb]["fur"] - S[bb]["fur_ap50_excluding_dup_images"]["ap"]
allowed |= {f"{-leak:+.3f}".lstrip("+"), f"{N['old_selected'] - N['old_final']:.3f}"}
allowed.add(f"{max(N[bb]['folds']) - min(N[bb]['folds']):.3f}")
allowed.add(f"{N[bb]['fur'] / N['old_selected']:.0%}".rstrip("%"))
allowed.add(f"{N[bb]['fur'] - min(N[b]['fur'] for b in ('rtdetr-l', 'yolo11m', 'yolov8m')):.3f}")
allowed.add(f"{N['full']['fur'] / N[bb]['fur']:.0%}".rstrip("%"))
allowed.add(str(5 * sum(1 for v in RES if not v.endswith("overfit"))))   # model-folds
CONST = {"14", "12", "13", "0.95", "0.02", "0.1", "0.25", "0.5", "0.3", "0.75", "90", "80", "60", "20", "1,280", "1280", "728", "1,316",
         "2", "4", "5", "3", "1", "0", "10", "2000", "2,000", "46", "72", "50", "101", "2017", "2024", "0.15", "0.999", "15", "6", "16", "11"}
allowed |= CONST

for name in ("Furcation_Benchmark_Manuscript.docx", "Furcation_Benchmark_Supplementary.docx"):
    d = Document(os.path.join(HERE, "manuscript", name))
    # citation superscripts and section headings are not results
    keep, in_refs = [], False
    for p in d.paragraphs:
        if p.style.name.startswith("Heading"):
            in_refs = p.text.strip() == "References"   # reference list (years, pages) is not a result
            continue
        if not in_refs:
            keep.append("".join(r.text for r in p.runs if not r.font.superscript))
    body = "\n".join(keep).replace("600077", "")
    body = re.sub(r"10\.5281/zenodo\.\d+", "", body)
    body = re.sub(r"\b\d+\.\d+\.\d+\b", "", body)            # software versions
    body = re.sub(r"\b(Figure|Table|Supplementary Table|Supplementary Figure|Section)\s+S?\d+[A-B]?", "", body)
    body = re.sub(r"criteri(a|on) \d(–\d)?", "", body)
    for m in re.finditer(r"(?<![\w^])[-+−]?\d[\d,]*\.?\d*", body):
        tok = m.group().replace("−", "-").lstrip("+").rstrip(".,").rstrip(",")
        if not tok or tok in ("-",):
            continue
        ok(tok in allowed or tok.lstrip("-") in allowed, f"{name}: untraced number '{tok}' near '{body[max(0, m.start() - 40):m.end() + 20]!r}'")

# 2. headline values recomputed from analyze.py output
d = Document(os.path.join(HERE, "manuscript", "Furcation_Benchmark_Manuscript.docx")); text = "\n".join(p.text for p in d.paragraphs)
for v, r in RES.items():
    if v.endswith("overfit"):
        continue
    b = r["by_seed"]["0"]
    ok(f"{b['fur_ap50']:.3f}" in text or v.startswith("no_"), f"{v} FUR AP50 {b['fur_ap50']:.3f} not in manuscript")
    ok(abs(N[v]["fur"] - b["fur_ap50"]) < 1e-12, f"{v} numbers.json differs from res_final.json")
ok(N["best_benchmark"] == max(["rtdetr-l", "yolo11m", "yolov8m"], key=lambda v: RES[v]["by_seed"]["0"]["fur_ap50"]), "best benchmark wrong")
ok("no new-architecture claim is made" in text, "pre-specified no-claim outcome not stated")

print(f"{checks} checks, {len(fails)} failures")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
