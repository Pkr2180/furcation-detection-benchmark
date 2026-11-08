"""Collect the submission set into paper/submission/: manuscript, supplement, a stand-alone Tables file, every figure as
a separate 300-dpi TIFF and PNG, and every table as CSV.   python paper/package_submission.py
Run after make_figs_tables.py and build_manuscript.py (finalize.ps1 does both)."""
import csv, glob, json, os, shutil
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Pt, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "submission")
shutil.rmtree(OUT, ignore_errors=True)
for d in ("Figures", "Supplementary_Figures", "Tables_CSV"):
    os.makedirs(os.path.join(OUT, d))

shutil.copy2(os.path.join(HERE, "manuscript", "Furcation_Benchmark_Manuscript.docx"), os.path.join(OUT, "Manuscript.docx"))
shutil.copy2(os.path.join(HERE, "manuscript", "Furcation_Benchmark_Supplementary.docx"), os.path.join(OUT, "Supplementary_Material.docx"))
FM = json.load(open(os.path.join(HERE, "figure_map.json")))   # written by build_manuscript.py: source -> final number
for kind, sub in (("main", "Figures"), ("supp", "Supplementary_Figures")):
    for src, label in FM[kind]:
        final = label.replace(" ", "")                          # "Figure 4" -> "Figure3", "Figure S1" -> "FigureS5"
        for ext in ("png", "tif"):
            shutil.copy2(os.path.join(HERE, "figures", f"{src}.{ext}"), os.path.join(OUT, sub, f"{final}.{ext}"))
TM = json.load(open(os.path.join(HERE, "table_map.json")))     # source csv -> final table number (citation order)
for src, label in TM["main_tables"] + TM["supp_tables"]:
    short = src.split("_", 1)[1]
    shutil.copy2(os.path.join(HERE, "tables", src + ".csv"), os.path.join(OUT, "Tables_CSV", f"{label.replace(' ', '')}_{short}.csv"))

# stand-alone Tables file (landscape, main tables 1-5) for journals that want tables separately
CAP = {"Table1_dataset_folds": "Table 1. Development folds and external set.",
       "Table2_main_results": "Table 2. Furcation detection and other findings, pooled out-of-fold over five group-disjoint folds.",
       "Table3_preregistered_criteria": "Table 4. Pre-specified criteria for claiming a new detector (all four required).",
       "Table4_paired_comparisons": "Table 3. Paired comparisons of furcation AP@50.",
       "Table5_minimum_evaluation_standard": "Table 5. Proposed minimum evaluation standard for radiographic furcation detection."}
doc = Document()
s = doc.sections[0]; s.orientation = WD_ORIENT.LANDSCAPE
s.page_width, s.page_height = s.page_height, s.page_width
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(s, m, Inches(0.7))
doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(10)
for k, (name, cap) in enumerate(sorted(CAP.items(), key=lambda x: int(x[1].split()[1].rstrip(".")))):
    if k:
        doc.add_page_break()
    rows = list(csv.reader(open(os.path.join(HERE, "tables", name + ".csv"), encoding="utf-8")))
    doc.add_paragraph().add_run(cap).bold = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            cell = t.cell(i, j); cell.text = ""
            r = cell.paragraphs[0].add_run(c); r.font.size = Pt(9); r.bold = i == 0
doc.save(os.path.join(OUT, "Tables.docx"))
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        print(os.path.relpath(os.path.join(root, f), OUT))
