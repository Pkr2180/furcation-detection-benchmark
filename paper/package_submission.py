"""Collect the submission set into paper/submission/: manuscript, supplement, a stand-alone Tables file, every figure as
a separate 300-dpi TIFF and PNG, and every table as CSV.   python paper/package_submission.py
Run after make_figs_tables.py and build_manuscript.py (finalize.ps1 does both)."""
import csv, glob, os, shutil
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
for f in sorted(glob.glob(os.path.join(HERE, "figures", "*.*"))):
    sub = "Supplementary_Figures" if os.path.basename(f).startswith("FigureS") else "Figures"
    shutil.copy2(f, os.path.join(OUT, sub, os.path.basename(f)))
for f in glob.glob(os.path.join(HERE, "tables", "*.csv")):
    shutil.copy2(f, os.path.join(OUT, "Tables_CSV", os.path.basename(f)))

# stand-alone Tables file (landscape, main tables 1-5) for journals that want tables separately
CAP = {"Table1_dataset_folds": "Table 1. Development folds and external set.",
       "Table2_main_results": "Table 2. Furcation detection and other findings, pooled out-of-fold over five group-disjoint folds.",
       "Table3_preregistered_criteria": "Table 3. Pre-registered criteria for claiming a new detector (all four required).",
       "Table4_paired_comparisons": "Table 4. Paired comparisons of furcation AP@50.",
       "Table5_minimum_evaluation_standard": "Table 5. Proposed minimum evaluation standard for radiographic furcation detection."}
doc = Document()
s = doc.sections[0]; s.orientation = WD_ORIENT.LANDSCAPE
s.page_width, s.page_height = s.page_height, s.page_width
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(s, m, Inches(0.7))
doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(10)
for k, (name, cap) in enumerate(CAP.items()):
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
