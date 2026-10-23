"""Second-annotator check of the generated domains (supervisor's final review, point 5).

  python -m scripts.make_annotation_sample sample    -> analysis/second_annotator/sheet.csv + key.csv
  python -m scripts.make_annotation_sample score     -> agreement of the filled sheet with the key

The sheet holds 30 test items from each of the 11 generated domains (330 items), stratified
by label and shuffled, with the gold label hidden. A second native Urdu speaker fills
`annotator_label` (one of `allowed_labels`) and `natural` (yes/no: could a person have
written this?). `score` reports raw agreement and Cohen's kappa per domain and overall.
The returned sheet goes to analysis/second_annotator/answer/sheet.xlsx. The annotator filled
only `annotator_label` (the `natural` column came back empty), so only label agreement is scored.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pipeline.common import load_split
from pipeline.config import ANALYSIS, DOMAIN_BY_KEY, TASKS

OUT = ANALYSIS / "second_annotator"
PER_DOMAIN = 30


def sample():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TASKS:
        for k in t.domains:
            d = DOMAIN_BY_KEY[k]
            if d.origin != "generated":
                continue
            df = load_split(k, "test", t)
            n_lab = df.label.nunique()
            per = [PER_DOMAIN // n_lab + (i < PER_DOMAIN % n_lab) for i in range(n_lab)]
            for (lab, g), n in zip(df.groupby("label"), per):
                for r in g.sample(min(n, len(g)), random_state=0).itertuples():
                    rows.append({"domain": f"{d.script[:3]} {d.sid}", "task": t.title,
                                 "text": r.text, "allowed_labels": " / ".join(t.label_names),
                                 "gold": t.label_names[t.labels.index(lab)]})
    s = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    s.insert(0, "item_id", np.arange(1, len(s) + 1))
    s[["item_id", "gold"]].to_csv(OUT / "key.csv", index=False, encoding="utf-8-sig")
    sheet = s.drop(columns="gold").assign(annotator_label="", natural="", comment="")
    sheet.to_csv(OUT / "sheet.csv", index=False, encoding="utf-8-sig")
    write_xlsx(sheet)
    print(f"{len(s)} items from {s.domain.nunique()} domains -> {OUT}")


def write_xlsx(sheet):
    """The copy the annotator fills: drop-down lists per row, wrapped right-to-left-safe text."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    cols = list(sheet.columns)
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in sheet.itertuples(index=False):
        ws.append(list(r))
    widths = {"item_id": 8, "domain": 10, "task": 26, "text": 70, "allowed_labels": 28,
              "annotator_label": 16, "natural": 10, "comment": 30}
    for i, name in enumerate(cols, start=1):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = widths.get(name, 15)
        for cell in ws[letter][1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    lab_col = ws.cell(1, cols.index("annotator_label") + 1).column_letter
    nat_col = ws.cell(1, cols.index("natural") + 1).column_letter
    by_labels = {}
    for i, labels in enumerate(sheet.allowed_labels, start=2):
        by_labels.setdefault(labels, []).append(f"{lab_col}{i}")
    for labels, cells in by_labels.items():
        dv = DataValidation(type="list", formula1='"' + labels.replace(" / ", ",") + '"', allow_blank=True)
        ws.add_data_validation(dv)
        for c in cells:
            dv.add(c)
    dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{nat_col}2:{nat_col}{len(sheet) + 1}")
    ws.freeze_panes = "A2"
    wb.save(OUT / "sheet.xlsx")


def score():
    xl = OUT / "answer" / "sheet.xlsx"
    sh = (pd.read_excel(xl, dtype=str, keep_default_na=False) if xl.exists()
          else pd.read_csv(OUT / "sheet.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False))
    sh["item_id"] = sh.item_id.astype(int)
    m = sh.merge(pd.read_csv(OUT / "key.csv", encoding="utf-8-sig"), on="item_id")
    m = m[m.annotator_label.str.strip() != ""]
    if m.empty:
        sys.exit("sheet has no labels yet")
    if len(m) < len(sh):
        print(f"warning: {len(sh) - len(m)} of {len(sh)} items are still unlabelled")
    m["agree"] = m.annotator_label.str.strip().str.lower() == m.gold.str.lower()
    out = []
    for dom, g in list(m.groupby("domain")) + [("ALL", m)]:
        out.append({"domain": dom, "n": len(g), "agreement": g.agree.mean(),
                    "kappa": cohen_kappa_score(g.gold.str.lower(), g.annotator_label.str.strip().str.lower())})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "agreement.csv", index=False)   # read by pipeline.make_tables
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
