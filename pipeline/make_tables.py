"""Stage 5 — write every number the paper prints into paper/tables/*.tex.

Nothing numeric is typed into main.tex by hand: in-text figures come from
\\newcommand macros in tables/macros.tex, tables are \\input from this directory.

Usage:  python -m pipeline.make_tables            (dataset tables only, before results)
        python -m pipeline.make_tables --results  (also the results tables from analysis/)
"""
import argparse
import json

import numpy as np
import pandas as pd

from pipeline.config import ANALYSIS, DOMAINS, DROPPED_V11, ROOT, TASKS, TASK_BY_ID

OUT = ROOT / "paper" / "tables"
TASK_ORDER = ["N-SA", "N-HS", "N-FN", "N-QA", "R-SA", "R-CA", "R-P3"]
MODEL_LABEL = {"Length-only": "Length-only (shortcut)", "XLM-R": "XLM-R (fine-tuned)", "mBERT": "mBERT (fine-tuned)",
               "Llama-3.1-8B": "Llama-3.1-8B (few-shot)", "Qwen2.5-7B": "Qwen2.5-7B (few-shot)",
               "Mistral-7B": "Mistral-7B (few-shot)"}
MODEL_ORDER = list(MODEL_LABEL)
SHORT = {"N-SA": "Sentiment", "N-HS": "Hate Speech", "N-FN": "Fake News", "N-QA": "QA Validation",
         "R-SA": "Sentiment (binary)", "R-CA": "Cyber Abuse", "R-P3": "3-class Polarity"}


def fmt_int(x):
    return f"{int(x):,}"


def macros(d):
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in d.items()]
    return "\n".join(lines) + "\n"


def dataset_tables():
    audit = pd.read_csv(ANALYSIS / "data_audit_v11.csv", encoding="utf-8-sig")
    audit = audit.set_index("key")
    task_of = {k: t for t in TASKS for k in t.domains}
    rows, mac = [], {}
    tot_tr = tot_te = synth_rows = 0
    n_synth_domains = 0
    for tid in TASK_ORDER:
        t = TASK_BY_ID[tid]
        for k in t.domains:
            d = next(x for x in DOMAINS if x.key == k)
            a = audit.loc[k]
            ntr, nte = int(a["v1.1_train"]), int(a["v1.1_test"])
            tot_tr += ntr; tot_te += nte
            if d.origin == "generated":
                n_synth_domains += 1
                synth_rows += ntr + nte
            dist = json.loads(a["v1.1_test_dist"])
            dist_s = "/".join(str(dist.get(l, 0)) for l in t.labels)
            rows.append({
                "script": t.script, "task": SHORT[tid], "id": d.sid, "domain": d.name,
                "C": len(t.labels), "ntr": fmt_int(ntr), "nte": fmt_int(nte), "dist": dist_s,
                "synth": d.origin,
                "nd": f"{100 * float(a['near_dup_test_rate']):.1f}",
                "split": "cluster" if a.get("cluster_resplit", 0) == 1 else "original",
            })
    total = tot_tr + tot_te
    v10 = audit[["v1.0_train", "v1.0_test"]].sum().sum()
    mac.update({
        "NumDomains": len(rows), "NumDomainsVten": len(DOMAINS), "NumDropped": len(DROPPED_V11),
        "NumTrain": fmt_int(tot_tr), "NumTest": fmt_int(tot_te), "NumTotal": fmt_int(total),
        "NumTotalVten": fmt_int(v10), "NumExactRemovedVten": "4,196",
        "NumSynthDomains": n_synth_domains, "NumSynthRows": fmt_int(synth_rows),
        "PctSynth": f"{100 * synth_rows / total:.1f}",
        "NumResplit": sum(r["split"] == "cluster" for r in rows),
        "NumNastaliqDomains": sum(r["script"] == "Nastaliq" for r in rows),
        "NumRomanDomains": sum(r["script"] == "Roman" for r in rows),
        "NumMatrices": len(TASKS),
        "NumShifts": sum(len(t.domains) * (len(t.domains) - 1) for t in TASKS),
    })

    # Table: dataset inventory (caption lives in main.tex; this file is the tabular body)
    body = []
    last = None
    for r in rows:
        head = (r["script"], r["task"])
        if last is not None and head != last:
            body.append("\\midrule")
        s = r["script"] if (last is None or head[0] != last[0]) else ""
        tk = r["task"] if head != last else ""
        body.append(f"{s} & {tk} & {r['id']} & {r['domain']} & {r['C']} & {r['ntr']} & {r['nte']} & "
                    f"{r['dist']} & {r['synth']} & {r['nd']} & {r['split']} \\\\")
        last = head
    tab = ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llllrrrllrl}\n\\toprule\n"
           "Script & Transfer matrix & ID & Domain & $C$ & $N_\\text{train}$ & $N_\\text{test}$ & "
           "Test dist. & Origin & Near-dup.\\,\\% & Split \\\\\n\\midrule\n"
           + "\n".join(body) + f"\n\\midrule\n\\multicolumn{{5}}{{l}}{{Total ({len(rows)} domains)}} & "
           f"{fmt_int(tot_tr)} & {fmt_int(tot_te)} & & & & \\\\\n\\bottomrule\n\\end{{tabular}}}}\n")
    (OUT / "datasets.tex").write_text(tab, encoding="utf-8")

    # Table: domains removed in v1.1
    drop = "\n".join(
        f"{next(d for d in DOMAINS if d.key == k).script} & {next(d for d in DOMAINS if d.key == k).collection} & "
        f"{next(d for d in DOMAINS if d.key == k).name} & "
        f"{why.replace('%', chr(92) + '%').replace(chr(34) + '92 News' + chr(34), '``92 News' + chr(39) * 2)} \\\\"
        for k, why in DROPPED_V11.items())
    (OUT / "dropped.tex").write_text(
        "\\begin{tabular}{lllp{6.0cm}}\n\\toprule\nScript & Collection & Domain & Reason \\\\\n\\midrule\n"
        + drop + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return mac


def synthetic_audit_table():
    """Appendix table: measured generated-text signals vs. the unmeasured card figures."""
    a = pd.read_csv(ANALYSIS / "synthetic_audit.csv", encoding="utf-8-sig")
    origin = {f"{d.script[:3]} {d.sid}": d.origin for d in DOMAINS}
    dropped = {f"{d.script[:3]} {d.sid}" for d in DOMAINS if d.key in DROPPED_V11}
    lines = []
    for r in a.itertuples():
        o = "excluded" if r.domain in dropped else origin[r.domain]
        lines.append(f"{r.domain} & {r.name} & {fmt_int(r.n)} & {100 * r.template_rate:.1f} & "
                     f"{r.distinct2:.2f} & {r.len_cv:.2f} & {100 * r.seed_match:.1f} & {o} \\\\")
    (OUT / "synthetic_audit.tex").write_text(
        "\\begin{tabular}{llrrrrrl}\n\\toprule\n"
        "Domain & Name & $N$ & Template\\,\\% & Distinct-2 & Len.\\,CV & Seed\\,\\% & Origin \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    gen = a[a.domain.map(lambda x: origin.get(x) == "generated")]
    org = a[a.domain.map(lambda x: origin.get(x) == "organic")]
    return {"GenDistinctMax": f"{gen.distinct2.max():.2f}", "OrgDistinctMin": f"{org.distinct2.min():.2f}",
            "GenLenCVMax": f"{gen.len_cv.max():.2f}", "OrgLenCVMin": f"{org.len_cv.min():.2f}",
            "GenTemplateMax": f"{100 * gen.template_rate.max():.0f}"}


def length_ss(tid, domains=None):
    """In-domain score of the length-only shortcut (mean diagonal), optionally on a submatrix."""
    m = pd.read_csv(ANALYSIS / f"matrix_{tid}_Length-only.csv", index_col=0)
    if domains:
        m = m.loc[domains, domains]
    return float(np.mean(np.diag(m.values)))


def rob_str(model, r, len_ss):
    """%Rob with the two guards: dagger = SS within 0.05 of uniform random or %Rob >= 100
    (analyze.apply_guard); double dagger = SS not above the length-only shortcut's SS."""
    if r.degenerate:
        return "n/a$^\\dagger$"
    below_len = model != "Length-only" and r.SS <= len_ss
    return f"{100 * r.rob:.1f}" + ("$^\\ddagger$" if below_len else "")


def provenance_table():
    """Appendix table: where each v1.1 domain comes from, with a URL when one is known.
    Source: analysis/provenance_worksheet.csv (author-facing notes are not printed)."""
    w = pd.read_csv(ANALYSIS / "provenance_worksheet.csv", encoding="utf-8-sig")
    w = w[w.status != "DROPPED"]
    esc = lambda s: str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
    lines, last = [], None
    for r in w.itertuples():
        if last is not None and r.script != last:
            lines.append("\\midrule")
        src = esc(r.original_source)
        if isinstance(r.citation_key, str):
            src += f" \\citep{{{r.citation_key}}}"
        url = f"\\url{{{r.url}}}" if isinstance(r.url, str) else "--"
        status = {"VERIFIED": "verified", "UNTRACED": "untraced", "RESOLVED": "generated"}[r.status]
        lic = esc(r.licence) if isinstance(r.licence, str) else "--"
        lines.append(f"{r.script[:3]} {r.domain_id} & {r.origin} & {status} & {src} & {url} & {lic} \\\\")
        last = r.script
    (OUT / "provenance.tex").write_text(
        "\\begin{tabular}{llp{1.3cm}p{5.0cm}p{5.6cm}p{2.4cm}}\n\\toprule\n"
        "Domain & Origin & Source & Description & URL & Licence \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def shortcuts_table():
    """Appendix table: top single-phrase shortcut per included domain (pipeline.audit_shortcuts).
    Returns values sc.TASK.SID.{in,out,phrase} for every audited domain, included or not."""
    import re
    a = pd.read_csv(ANALYSIS / "shortcut_audit.csv", encoding="utf-8-sig")
    top = a[a["rank"] == 1]
    urdu = lambda s: f"\\ur{{{s}}}" if re.search(r"[\u0600-\u06FF]", s) else f"\\textit{{{s}}}"
    lines, v = [], {}
    for tid in TASK_ORDER:
        for _, r in top[(top.task == tid) & top.included].iterrows():
            lines.append(f"{SHORT[tid]} ({tid[0]}) & {r.domain} & {urdu(str(r.phrase))} & {r['class']} & "
                         f"{100 * r.df_in_class:.1f} & {100 * r.df_other:.1f} \\\\")
    for _, r in top.iterrows():
        v[f"sc.{r.task}.{r.domain}.in"] = f"{100 * r.df_in_class:.1f}"
        v[f"sc.{r.task}.{r.domain}.out"] = f"{100 * r.df_other:.1f}"
    (OUT / "shortcuts.tex").write_text(
        "\\begin{tabular}{lllcrr}\n\\toprule\nMatrix & Domain & Phrase & Class & \\% of class & \\% of others \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return v


def fmt(x, sd=None, pct=False):
    if pd.isna(x):
        return "--"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:.3f}" if sd is None or sd == 0 else f"{x:.3f}\\,$\\pm$\\,{sd:.3f}"


def results_tables(mac):
    main = pd.read_csv(ANALYSIS / "table_main.csv")
    base = pd.read_csv(ANALYSIS / "baselines.csv").set_index("task")
    for tid in TASK_ORDER:
        g = main[main.task == tid].set_index("model")
        if g.empty:
            continue
        b = base.loc[tid]
        lss = length_ss(tid)
        lines = [f"Majority class & {b.majority:.3f} & -- & -- & -- & -- & -- & -- & -- \\\\",
                 f"Uniform random & {b.random:.3f} & -- & -- & -- & -- & -- & -- & -- \\\\"]
        for m in MODEL_ORDER:
            if m not in g.index:
                continue
            r = g.loc[m]
            rob = rob_str(m, r, lss)
            lines.append(
                f"{MODEL_LABEL[m]} & {fmt(r.SS, r.SS_sd)} & {fmt(r.ST, r.ST_sd)} & {100 * r.dS:.1f} & "
                f"{100 * r.WSD:.1f} & {100 * r.WTD:.1f} & {fmt(r.pos_dT, pct=True)} & "
                f"{fmt(r.harder_target, pct=True)} & {rob} \\\\")
            if m in ("Length-only", "mBERT"):
                lines.append("\\midrule")
        (OUT / f"results_{tid}.tex").write_text(
            "\\begin{tabular}{lcccccccc}\n\\toprule\n"
            "Model & SS & ST & $\\overline{\\Delta_S}$ & WSD & WTD & $\\Delta_T{>}0$\\,\\% & Harder-tgt\\,\\% & \\%Rob \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # per-target Target Drop table (Calderon: is the target intrinsically hard?)
    pdm = pd.read_csv(ANALYSIS / "table_per_domain.csv")

    # per-target tables, one per script, all matrices stacked
    for script, tids in (("nastaliq", TASK_ORDER[:4]), ("roman", TASK_ORDER[4:])):
        models = [m for m in MODEL_ORDER if m in set(pdm.model) and m != "Length-only"]
        hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in models)
        rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
        sub = " & ".join("TT & $\\overline{\\Delta_T}$" for _ in models)
        lines = []
        for tid in tids:
            g = pdm[pdm.task == tid]
            lines.append(f"\\midrule\n\\multicolumn{{{1 + 2 * len(models)}}}{{l}}{{\\textit{{{SHORT[tid]}}}}} \\\\")
            for d in dict.fromkeys(g.domain):
                cells = []
                for m in models:
                    r = g[(g.domain == d) & (g.model == m)].iloc[0]
                    cells += [f"{r['SS=TT']:.3f}", f"{100 * r.dT_target:.1f}"]
                lines.append(f"{d} & " + " & ".join(cells) + " \\\\")
        (OUT / f"per_target_{script}.tex").write_text(
            "\\begin{tabular}{l" + "rr" * len(models) + "}\n\\toprule\n"
            f"Target & {hdr} \\\\\n{rules}\n & {sub} \\\\\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # organic-only table
    org = pd.read_csv(ANALYSIS / "table_organic_only.csv")
    lines, last = [], None
    for tid in TASK_ORDER:
        g = org[org.task == tid].set_index("model")
        if g.empty:
            continue
        if last is not None:
            lines.append("\\midrule")
        first = True
        for m in MODEL_ORDER:
            if m not in g.index:
                continue
            r = g.loc[m]
            lab = f"{SHORT[tid]} ({r.domains.replace(',', ', ')})" if first else ""
            rob = rob_str(m, r, length_ss(tid, r.domains.split(",")))
            lines.append(f"{lab} & {m} & {fmt(r.SS, r.SS_sd)} & {fmt(r.ST, r.ST_sd)} & {100 * r.dS:.1f} & {rob} \\\\")
            first = False
        last = tid
    (OUT / "organic.tex").write_text(
        "\\begin{tabular}{llcccr}\n\\toprule\nMatrix (organic domains) & Model & SS & ST & $\\overline{\\Delta_S}$ & \\%Rob \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # bootstrap table: fine-tuned minus few-shot mean ST, pp, 95% CI
    bt = pd.read_csv(ANALYSIS / "bootstrap_ft_vs_llm.csv")
    llms = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]
    lines = []
    for tid in TASK_ORDER:
        for a in ("XLM-R", "mBERT"):
            cells = []
            for b in llms:
                r = bt[(bt.task == tid) & (bt.a == a) & (bt.b == b)].iloc[0]
                star = "$^{*}$" if r.p_holm < 0.05 else ""
                cells.append(f"{100 * r.diff_ST:+.1f}{star} [{100 * r.ci_lo:+.1f}, {100 * r.ci_hi:+.1f}]")
            lines.append(f"{SHORT[tid] if a == 'XLM-R' else ''} & {a} & " + " & ".join(cells) + " \\\\")
        if tid != TASK_ORDER[-1]:
            lines.append("\\addlinespace[2pt]")
    (OUT / "bootstrap.tex").write_text(
        "\\begin{tabular}{llccc}\n\\toprule\nMatrix & Fine-tuned & $-$ Llama-3.1-8B & $-$ Qwen2.5-7B & $-$ Mistral-7B \\\\\n"
        "\\midrule\n" + "\n".join(lines).replace("\\\\\n\\addlinespace", "\\\\\\addlinespace")
        + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # parse failures
    pf = pd.read_csv(ANALYSIS / "parse_failures.csv")
    lines = [f"{SHORT[tid]} ({tid[0]}) & " + " & ".join(
        f"{100 * pf[(pf.task == tid) & (pf.model == b)].parse_fail_rate.iloc[0]:.2f}" for b in llms) + " \\\\"
        for tid in TASK_ORDER]
    (OUT / "parse_failures.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\nMatrix & Llama-3.1-8B & Qwen2.5-7B & Mistral-7B \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # error categories for the worst shift of each matrix
    ec = pd.read_csv(ANALYSIS / "errors_categories.csv")
    lines = []
    for r in ec.set_index("task").loc[TASK_ORDER].reset_index().itertuples():
        pc = lambda x: "--" if pd.isna(x) else f"{100 * x:.0f}"
        lines.append(f"{SHORT[r.task]} ({r.task[0]}) & {r.src}$\\rightarrow${r.tgt} & {r.TT:.2f} / {r.ST:.2f} & "
                     f"{r.n_err:,}/{r.n_test:,} & {r.top_dir} ({pc(r.top_dir_share)}) & "
                     f"{pc(r.len_shift_err)} / {pc(r.len_shift_ok)} & {pc(r.unseen_err)} / {pc(r.unseen_ok)} & "
                     f"{pc(r.hard_in_err)} / {pc(r.hard_in_ok)} \\\\")
    (OUT / "error_categories.tex").write_text(
        "\\begin{tabular}{llccllll}\n\\toprule\n"
        "Matrix & Shift & TT / ST & Errors & Main confusion (\\%) & Length shift & Unseen vocab. & Hard in-domain \\\\\n"
        " & & & & & err / ok & err / ok & err / ok \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    mac["NumDegenerate"] = int(main.degenerate.sum())
    mac["NumLLMCells"] = int((main.family == "llm").sum())
    return mac


# ----------------------------------------------------------------------------- \val{key} lookup
def compute_stats():
    """Run counts and measured GPU time, read from the result cells themselves.
    Fine-tuning trains once per (model, seed, source); its train_seconds is repeated
    in every target cell of that run, so it is counted once. Encoder inference time
    is not recorded and is not included."""
    from pipeline.config import RUNS
    active = {k for t in TASKS for k in t.domains}
    ft, llm_secs, n_ft_cells, n_llm_cells = {}, 0.0, 0, 0
    for p in RUNS.glob("*/*/*/seed*/*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["source"] not in active or r["target"] not in active:
            continue
        if r["family"] == "ft":
            n_ft_cells += 1
            ft[(r["task"], r["model"], r["seed"], r["source"])] = r["train_seconds"]
        else:
            n_llm_cells += 1
            llm_secs += r["seconds"]
    return {"ftruns": len(ft), "ftcells": f"{n_ft_cells:,}", "llmcells": f"{n_llm_cells:,}",
            "fthours": f"{sum(ft.values()) / 3600:.1f}", "llmhours": f"{llm_secs / 3600:.1f}"}


def pp(x):
    return f"{100 * x:.1f}"


def value_table():
    """Every number quoted in the prose, as \\val{key}. Keys:
    TASK.MODEL.{SS,SSsd,ST,STsd,dS,WSD,WTD,posdT,harder,rob}; TASK.{maj,rand};
    org.TASK.MODEL.*; boot.TASK.FT.LLM.{diff,lo,hi,p}; pf.TASK.LLM;
    pt.TASK.MODEL.DOMAIN.{TT,dT}; err.TASK.*; range.TASK.{ft,llm}.STAT; compute.*"""
    v = {}

    def put_stats(prefix, r):
        v[f"{prefix}.SS"], v[f"{prefix}.SSsd"] = f"{r.SS:.3f}", f"{r.SS_sd:.3f}"
        v[f"{prefix}.ST"], v[f"{prefix}.STsd"] = f"{r.ST:.3f}", f"{r.ST_sd:.3f}"
        v[f"{prefix}.dS"], v[f"{prefix}.WSD"], v[f"{prefix}.WTD"] = pp(r.dS), pp(r.WSD), pp(r.WTD)
        v[f"{prefix}.posdT"], v[f"{prefix}.harder"] = f"{100 * r.pos_dT:.0f}", f"{100 * r.harder_target:.0f}"
        v[f"{prefix}.rob"] = "n/a" if r.degenerate else f"{100 * r.rob:.1f}"

    main = pd.read_csv(ANALYSIS / "table_main.csv")
    for r in main.itertuples():
        put_stats(f"{r.task}.{r.model}", r)
    for tid in TASK_ORDER:
        v[f"{tid}.lenSS"] = f"{length_ss(tid):.3f}"
    # summaries across all matrices
    for fam in ("ft", "llm"):
        s = main[main.family == fam]
        v[f"all.{fam}.dSmin"], v[f"all.{fam}.dSmax"] = pp(s.dS.min()), pp(s.dS.max())
        v[f"all.{fam}.posdTmin"], v[f"all.{fam}.posdTmax"] = f"{100 * s.pos_dT.min():.0f}", f"{100 * s.pos_dT.max():.0f}"
        v[f"all.{fam}.hardermax"] = f"{100 * s.harder_target.max():.0f}"
    wins = sum(main[(main.task == t) & (main.family == "ft")].SS.min() > main[(main.task == t) & (main.family == "llm")].SS.max()
               for t in TASK_ORDER)
    v["all.ftSSwins"] = str(int(wins))
    v["all.numLenFlag"] = str(int(sum((r.family == "llm" or r.family == "ft") and r.SS <= length_ss(r.task)
                                      for r in main.itertuples())))
    # 3-class polarity: mean ST within the review collection, within the opinion
    # collection, and across the two (mean matrices over seeds)
    for model in MODEL_ORDER[1:]:
        m = pd.read_csv(ANALYSIS / f"matrix_R-P3_{model}.csv", index_col=0)
        coll = {d: d.split("-")[0] for d in m.index}
        groups = {"withinPR": [], "withinTW": [], "across": []}
        for i in m.index:
            for j in m.columns:
                if i == j:
                    continue
                key = "across" if coll[i] != coll[j] else f"within{coll[i]}"
                groups[key].append(m.loc[i, j])
        for key, xs in groups.items():
            v[f"R-P3.{model}.{key}"] = f"{np.mean(xs):.3f}"
    for tid, g in main.groupby("task"):
        for fam in ("ft", "llm"):
            s = g[g.family == fam]
            for col, f in (("SS", "{:.3f}"), ("ST", "{:.3f}"), ("dS", None), ("rob", None)):
                lo, hi = s[col].min(), s[col].max()
                fm = (lambda x: pp(x)) if f is None else (lambda x, f=f: f.format(x))
                v[f"range.{tid}.{fam}.{col}"] = f"{fm(lo)}--{fm(hi)}"
    base = pd.read_csv(ANALYSIS / "baselines.csv")
    for r in base.itertuples():
        v[f"{r.task}.maj"], v[f"{r.task}.rand"] = f"{r.majority:.3f}", f"{r.random:.3f}"
    for r in pd.read_csv(ANALYSIS / "table_organic_only.csv").itertuples():
        put_stats(f"org.{r.task}.{r.model}", r)
    bt = pd.read_csv(ANALYSIS / "bootstrap_ft_vs_llm.csv")
    for r in bt.itertuples():
        k = f"boot.{r.task}.{r.a}.{r.b}"
        v[k + ".diff"], v[k + ".lo"], v[k + ".hi"] = f"{100 * r.diff_ST:+.1f}", f"{100 * r.ci_lo:+.1f}", f"{100 * r.ci_hi:+.1f}"
        v[k + ".p"] = "$p_\\text{Holm}<0.001$" if r.p_holm < 0.001 else f"$p_\\text{{Holm}}={r.p_holm:.3f}$"
    sig = bt.p_holm < 0.05
    v["boot.nsraw"] = str(int((bt.p >= 0.05).sum()))
    v["boot.n"] = str(len(bt))
    v["boot.llmbetter"] = str(int((sig & (bt.diff_ST < 0)).sum()))
    v["boot.ftbetter"] = str(int((sig & (bt.diff_ST > 0)).sum()))
    v["boot.ns"] = str(int((~sig).sum()))
    pf = pd.read_csv(ANALYSIS / "parse_failures.csv")
    for r in pf.itertuples():
        v[f"pf.{r.task}.{r.model}"] = f"{100 * r.parse_fail_rate:.2f}"
    v["pf.max"] = f"{100 * pf.parse_fail_rate.max():.1f}"
    v["pf.nonmistralmax"] = f"{100 * pf[pf.model != 'Mistral-7B'].parse_fail_rate.max():.2f}"
    for _, r in pd.read_csv(ANALYSIS / "table_per_domain.csv").iterrows():
        v[f"pt.{r.task}.{r.model}.{r.domain}.TT"] = f"{r['SS=TT']:.3f}"
        v[f"pt.{r.task}.{r.model}.{r.domain}.dT"] = pp(r.dT_target)
    for r in pd.read_csv(ANALYSIS / "errors_categories.csv").itertuples():
        k = f"err.{r.task}"
        v[k + ".src"], v[k + ".tgt"] = r.src, r.tgt
        v[k + ".dT"], v[k + ".ST"], v[k + ".TT"] = pp(r.dT), f"{r.ST:.3f}", f"{r.TT:.3f}"
        v[k + ".nerr"], v[k + ".ntest"], v[k + ".rate"] = f"{r.n_err:,}", f"{r.n_test:,}", f"{100 * r.err_rate:.0f}"
        v[k + ".dir"], v[k + ".dirshare"] = r.top_dir, f"{100 * r.top_dir_share:.0f}"
        v[k + ".bestllm"] = r.best_llm
        for c in ("len_shift", "unseen", "hard_in"):
            for s in ("err", "ok"):
                x = getattr(r, f"{c}_{s}")
                v[f"{k}.{c.replace('_', '')}.{s}"] = "--" if pd.isna(x) else f"{100 * x:.0f}"
    for k, x in compute_stats().items():
        v[f"compute.{k}"] = str(x)
    v.update(shortcuts_table())
    body = "\n".join(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{x}}}" for k, x in sorted(v.items()))
    (OUT / "values.tex").write_text(body + "\n", encoding="utf-8")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mac = dataset_tables()
    mac.update(synthetic_audit_table())
    provenance_table()
    if a.results:
        mac = results_tables(mac)
        value_table()
    (OUT / "macros.tex").write_text(macros(mac), encoding="utf-8")
    print(macros(mac))


if __name__ == "__main__":
    main()
