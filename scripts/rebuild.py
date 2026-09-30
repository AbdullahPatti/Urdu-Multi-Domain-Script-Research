"""Rebuild the benchmark domains whose source corpora state no licence.

Downloads each source from the authors' GitHub repository, matches every item of our
split by the SHA-1 of its normalised text (manifests/), and writes
data/<domain>/{train,test}.csv with our labels in our row order.

Usage (from the release folder):   python rebuild.py
Requires: pandas, openpyxl
"""
import hashlib
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from release_sources import DOMAIN_SOURCE, READERS, REPOS, norm

HERE = Path(__file__).resolve().parent


def fetch(repo, dest):
    url = f"https://codeload.github.com/{repo}/zip/HEAD"
    with urllib.request.urlopen(url) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        texts = {}
        for src, repo in REPOS.items():
            print(f"downloading {repo}")
            root = Path(tmp) / src
            fetch(repo, root)
            texts[src] = {hashlib.sha1(norm(t).encode("utf-8")).hexdigest(): t for t in READERS[src](str(root))}
        for key, src in DOMAIN_SOURCE.items():
            for split in ("train", "test"):
                man = pd.read_csv(HERE / "manifests" / key / f"{split}.csv", dtype=str)
                man["text"] = man.sha1.map(texts[src])
                missing = man.text.isna().sum()
                out = HERE / "data" / key / f"{split}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                man.dropna(subset=["text"])[["text", "label"]].to_csv(out, index=False, encoding="utf-8")
                print(f"  {key}/{split}: {len(man) - missing:,} of {len(man):,} items"
                      + (f" ({missing} not found in the current source file)" if missing else ""))


if __name__ == "__main__":
    main()
