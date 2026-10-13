"""groundrag CLI - index a corpus and ask questions, offline.

  python3 cli.py --corpus fixtures/corpus --build index.db
  python3 cli.py --db index.db --query "How many vacation days per month?"
"""

import argparse
import os
import sys

import rag


def main() -> int:
    ap = argparse.ArgumentParser(prog="groundrag")
    ap.add_argument("--corpus", help="directory of .md/.txt docs")
    ap.add_argument("--build", metavar="DB", help="build index from --corpus")
    ap.add_argument("--db", help="existing index to query")
    ap.add_argument("--query", help="question to ask")
    ap.add_argument("--log", default="abstain.jsonl",
                    help="abstention log path (append)")
    args = ap.parse_args()

    if args.build:
        if not args.corpus:
            ap.error("--build requires --corpus")
        n = rag.ingest(args.corpus, args.build)
        print(f"indexed {n} chunks -> {args.build}")
        return 0

    if not args.db or not args.query:
        ap.error("need --db and --query (or --corpus + --build)")

    v = rag.ask(args.db, args.query)
    if v.status == "grounded":
        print(f"[grounded] overlap={v.best_overlap:.2f} "
              f"iters={v.iterations}")
        print(f"citations: {', '.join(v.citations)}\n")
        print(v.answer)
        return 0
    rag.log_abstain(args.log, v, args.query)
    print(f"[abstain] grounding too weak (best overlap "
          f"{v.best_overlap:.2f}); logged to {args.log}")
    for line in v.log:
        print(f"  {line}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
