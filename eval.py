"""Offline eval: run the labeled gold set through the ask() loop, print
the metrics table, assert the two hard rules (no confident hallucination,
no missed answerable). Exits nonzero on regression.

Run:  python3 eval.py            (uses fixtures/, builds index in /tmp)
      python3 eval.py --index-only /path/corpus /path/out.db
"""

import argparse
import json
import os
import re
import sys
import tempfile

import rag

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "fixtures", "corpus")
EVAL = os.path.join(HERE, "fixtures", "eval_set.json")
CORPUS_LARGE = os.path.join(HERE, "fixtures", "corpus_large")
EVAL_LARGE = os.path.join(HERE, "fixtures", "eval_set_large.json")


def _norm(s: str) -> str:
    """Collapse all whitespace runs to single spaces (RFC text wraps lines
    mid-phrase); casefold for matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def run_eval(db_path: str, verbose: bool = True,
             eval_path: str = EVAL, k: int = 3) -> dict:
    with open(eval_path) as f:
        gold = json.load(f)

    stats = {"answerable": [0, 0], "unanswerable": [0, 0], "rows": []}
    hallucinations = []
    misses = []
    hard = gold.get("hard_traps", [])

    for case in gold["answerable"]:
        v = rag.ask(db_path, case["query"], k=k)
        ok = (v.status == "grounded"
              and any(c.startswith(case["expect_doc"] + "#") for c in v.citations)
              and _norm(case["expect_fragment"]) in _norm(v.answer or ""))
        stats["answerable"][0] += int(ok)
        stats["answerable"][1] += 1
        stats["rows"].append(("A", ok, case["query"], v.status,
                              round(v.best_overlap, 2), v.iterations))
        if not ok:
            misses.append(case["query"])

    for case in gold["unanswerable"]:
        v = rag.ask(db_path, case["query"], k=k)
        ok = v.status == "abstain"
        stats["unanswerable"][0] += int(ok)
        stats["unanswerable"][1] += 1
        stats["rows"].append(("U", ok, case["query"], v.status,
                              round(v.best_overlap, 2), v.iterations))
        if not ok:
            hallucinations.append((case["query"], v.best_overlap,
                                   v.citations))

    if verbose:
        print(f"{'kind':4} {'ok':3} {'status':9} {'ovlp':4} {'it':2} query")
        for kind, ok, q, status, ov, it in stats["rows"]:
            mark = "PASS" if ok else "FAIL"
            print(f"{kind:4} {mark:3} {status:9} {ov:<4} {it:<2} {q}")
        a = stats["answerable"]
        u = stats["unanswerable"]
        print(f"\nanswerable retrieved+grounded : {a[0]}/{a[1]}")
        print(f"unanswerable abstained         : {u[0]}/{u[1]}")
        print(f"confident hallucinations       : {len(hallucinations)}")
        if hard:
            caught = sum(1 for c in hard
                         if rag.ask(db_path, c["query"], k=k).status == "abstain")
            print(f"hard traps caught (reported, not gated) : "
                  f"{caught}/{len(hard)}  [lexical ceiling: Checkpoint 2/3 "
                  f"must clear these]")
        if hallucinations:
            for q, ov, cites in hallucinations:
                print(f"  !! answered anyway: {q!r} overlap={ov} cites={cites}")
    return {"stats": stats, "hallucinations": hallucinations, "misses": misses,
            "hard_traps": hard}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-only", nargs=2, metavar=("CORPUS", "OUT"),
                    help="just build an index from a directory")
    ap.add_argument("--large", action="store_true",
                    help="use the RFC corpus + large eval set")
    args = ap.parse_args()

    if args.index_only:
        n = rag.ingest(args.index_only[0], args.index_only[1])
        print(f"indexed {n} chunks -> {args.index_only[1]}")
        return 0

    corpus = CORPUS_LARGE if args.large else CORPUS
    eval_path = EVAL_LARGE if args.large else EVAL
    tmp = tempfile.mkdtemp(prefix="groundrag-")
    db = os.path.join(tmp, "index.db")
    n = rag.ingest(corpus, db)
    result = run_eval(db, eval_path=eval_path, k=5 if args.large else 3)
    # hard gates: zero hallucination, no missed answerable
    if result["hallucinations"] or result["stats"]["answerable"][0] < len(
            result["stats"]["rows"]) - result["stats"]["unanswerable"][1]:
        return 1
    if result["stats"]["answerable"][0] != result["stats"]["answerable"][1]:
        return 1
    print(f"\n(index: {n} chunks; gates passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
