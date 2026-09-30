"""Readers for the four public corpora that state no licence and are therefore not
redistributed: each returns the raw item texts from the authors' own release, so
that release/rebuild.py can reconstruct our splits from a text-hash manifest.

Every reader takes the directory into which the authors' GitHub repository was
unpacked and returns a list of item texts. Labels are not taken from the source:
the manifest carries our labels, so the rebuild is identical to the benchmark
even where our label differs from the source's (e.g. ISE-Hate, 98.7% agreement).
"""
import glob
import io
import os
import zipfile

import unicodedata

import pandas as pd

REPOS = {
    "usc": "MuhammadYaseenKhan/Urdu-Sentiment-Corpus",
    "ise": "hammad7007/ISE_dataset",
    "btt": "MaazAmjad/Datasets-for-Urdu-news",
    "axg": "HjH-Whu-CRC/Ax-to-Grind-Urdu",
}
# benchmark domain -> source reader
DOMAIN_SOURCE = {
    "Nastaliq/Sentiment Analysis/Domain_A_Twitter_Social_Media": "usc",
    "Nastaliq/Hate Speech/Domain_A_Social_Media_Offensive": "ise",
    "Nastaliq/Fake News Detection/Domain_A_Multi_Topic_News": "btt",
    "Nastaliq/Fake News Detection/Domain_B_Ax_to_Grind": "axg",
}


def norm(text):
    """Whitespace-collapsed text without byte-order marks or zero-width characters."""
    t = unicodedata.normalize("NFKC", str(text))
    for ch in ("﻿", "​", "‌", "‍", "‎", "‏"):
        t = t.replace(ch, "")
    return " ".join(t.split())


def _read_csv_any(f):
    for enc in ("utf-8-sig", "cp1256", "utf-16", "cp1252"):
        try:
            return pd.read_csv(f, encoding=enc)
        except (UnicodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


def _one(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return hits[0]


def read_usc(root):
    return pd.read_csv(_one(root, "urdu-sentiment-corpus-v1.tsv"), sep="\t").Tweet.astype(str).tolist()


def read_ise(root):
    out = []
    for f in glob.glob(os.path.join(root, "**", "*.xlsx"), recursive=True):
        df = pd.read_excel(f)
        col = next((c for c in df.columns if "text" in str(c).lower() or "tweet" == str(c).lower()), None)
        if col is not None:
            out += df[col].astype(str).tolist()
    return out


def read_btt(root):
    out = []
    for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith(".txt"):
                    raw = zf.read(name)
                    for enc in ("utf-8-sig", "utf-16", "cp1256"):
                        try:
                            out.append(raw.decode(enc))
                            break
                        except UnicodeDecodeError:
                            continue
    return out


def read_axg(root):
    out = []
    for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        df = _read_csv_any(f)
        col = next((c for c in df.columns if "news" in str(c).lower()), None)
        if col is not None:
            out += df[col].astype(str).tolist()
    return out


READERS = {"usc": read_usc, "ise": read_ise, "btt": read_btt, "axg": read_axg}
