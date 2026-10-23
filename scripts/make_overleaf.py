"""Assemble the Overleaf project: release/overleaf_project.zip

Contents: main.tex, refs.bib, exactly the tables/figures the paper \\input's or
\\includegraphics's, the Urdu font, and a latexmkrc that selects XeLaTeX (the paper
uses fontspec and a Nastaliq font, so pdfLaTeX cannot compile it). Overleaf runs
BibTeX itself with ACM-Reference-Format.bst, which ships with the acmart class.
Upload on Overleaf with New Project > Upload Project, then check Menu > Compiler = XeLaTeX.
"""
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "release" / "overleaf"
ZIP = ROOT / "release" / "overleaf_project.zip"

tex = (PAPER / "main.tex").read_text(encoding="utf-8")
needed = set(re.findall(r"\\input\{([^}]+)\}", tex)) | set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
needed = {n if Path(n).suffix else n + ".tex" for n in needed}
needed |= {"main.tex", "refs.bib", "fonts/NotoNastaliqUrdu.ttf"}

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for rel in sorted(needed):
    src = PAPER / rel
    assert src.exists(), f"missing {rel}"
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUT / rel)
(OUT / "latexmkrc").write_text("$pdf_mode = 5;   # XeLaTeX\n", encoding="utf-8")

ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(OUT).as_posix())
print(f"{ZIP.name}: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {ZIP.stat().st_size / 1e6:.1f} MB")
