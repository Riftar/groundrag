#!/usr/bin/env python3
"""Fetch and clean the large RFC test corpus for groundrag.

Sources are IETF RFCs (rfc-editor.org). RFCs are freely reproducible
under the IETF Trust's BSD-style license (see each file's header and
https://trustee.ietf.org/license-info/), which permits redistribution.

The cleaned .txt output is committed so the repo stays offline-testable;
this script exists to document provenance and allow regeneration:

    python3 scripts/make_corpus_large.py            # fetch + clean
    python3 scripts/make_corpus_large.py --no-fetch # clean local copies only
"""

import argparse
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fixtures", "corpus_large")

RFCS = {
    "rfc9110.txt": "https://www.rfc-editor.org/rfc/rfc9110.txt",  # HTTP semantics
    "rfc9112.txt": "https://www.rfc-editor.org/rfc/rfc9112.txt",  # HTTP/1.1 messaging
    "rfc1035.txt": "https://www.rfc-editor.org/rfc/rfc1035.txt",  # DNS implementation
    "rfc5321.txt": "https://www.rfc-editor.org/rfc/rfc5321.txt",  # SMTP
}

# page furniture to strip: footer/header lines and form feeds
FOOTER = re.compile(r"^\s*(RFC \d+.*\[Page \d+\]|\[Page \d+\]|"
                    r"\f|.*\f)\s*$")


def clean(text: str, rfc_id: str) -> str:
    lines = []
    for line in text.splitlines():
        if "\f" in line:
            continue
        if re.match(rf"^\s*{rfc_id}\b.*\d{{4}}\s*$", line):
            continue  # running header
        if re.match(r"^\s*RFC \d+.*\[Page \d+\]\s*$", line):
            continue  # running footer
        lines.append(line.rstrip())
    # collapse >2 blank lines to 1 (paragraph boundaries for chunker)
    out, blanks = [], 0
    for l in lines:
        if not l.strip():
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        out.append(l)
    return "\n".join(out).strip() + "\n"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    total = 0
    for name, url in RFCS.items():
        path = os.path.join(OUT, name)
        rfc_id = name.replace(".txt", "")
        if args.no_fetch:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            # local copies are pre-cleaned; just measure
            total += len(text)
            print(f"{name}: {len(text)} chars (local)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "groundrag-fixture/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", "replace")
        cleaned = clean(raw, rfc_id.upper())
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        total += len(cleaned)
        print(f"{name}: {len(raw)} -> {len(cleaned)} chars")
    print(f"total: {total} chars ({total/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
