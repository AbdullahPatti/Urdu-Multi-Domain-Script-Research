"""Assemble the arXiv source package: release/arxiv_submission.zip

Contents: main.tex (build comments removed), main.bbl (arXiv does not run BibTeX),
exactly the tables/figures the paper \\input's or \\includegraphics's, the Urdu font,
and 00README.json selecting the xelatex compiler (supported by arXiv since Nov 2025).
Run after compiling the paper with:  tectonic -X compile --keep-intermediates main.tex
"""
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "release" / "arxiv"
ZIP = ROOT / "release" / "arxiv_submission.zip"

tex = (PAPER / "main.tex").read_text(encoding="utf-8")
# drop the leading block of build comments; keep everything from \documentclass on
tex = tex[tex.index(r"\documentclass"):]
# use the processed bibliography directly, so no BibTeX run is needed on arXiv
bib_cmds = "\\bibliographystyle{plainnat}\n\\bibliography{refs}"
assert bib_cmds in tex
tex = tex.replace(bib_cmds, "\\input{main.bbl}")
needed = set(re.findall(r"\\input\{([^}]+)\}", tex)) | set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
needed = {n if Path(n).suffix else n + ".tex" for n in needed}
needed |= {"fonts/NotoNastaliqUrdu.ttf", "main.bbl"}

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
(OUT / "main.tex").write_text(tex, encoding="utf-8")
for rel in sorted(needed):
    src = PAPER / rel
    assert src.exists(), f"missing {rel}"
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUT / rel)
(OUT / "00README.json").write_text(json.dumps({
    "process": {"compiler": "xelatex"},
    "sources": [{"filename": "main.tex", "usage": "toplevel"}],
}, indent=2), encoding="utf-8")

ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(OUT).as_posix())
print(f"{ZIP.name}: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {ZIP.stat().st_size / 1e6:.1f} MB")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print("  ", f.relative_to(OUT).as_posix())
