# groundrag

Self-correcting retrieval with a hard abstain gate — SQLite FTS5/BM25,
a grounding judge, one-step query rewrite, and a refusal log. Pure
stdlib Python: no vector DB, no embeddings service, no framework, fully
offline.

The discipline this enforces: **when grounding is weak, it abstains and
logs it — it never guesses.** Every decision (retrieve / judge / rewrite
/ abstain) is a visible, deterministic step you can read in one sitting.

## Quick start

```bash
python3 cli.py --corpus fixtures/corpus --build index.db
python3 cli.py --db index.db --query "How many vacation days per month?"
python3 cli.py --db index.db --query "recipe for banana bread"   # -> abstains
```

Grounded answers print the ranked excerpts with `doc#chunk` citations;
abstentions print the score trail and append to `abstain.jsonl`.

## Eval (labeled gold set, hard gates)

```bash
python3 eval.py            # toy corpus: 3 synthetic docs, 11 cases
python3 eval.py --large    # RFC corpus: 4 RFCs, ~1850 chunks, 30 cases
```

Both sets enforce the **hard gates:** zero confident hallucinations,
zero missed answerable. Current — toy: 6/6 grounded, 5/5 abstained.
Large: 20/20 grounded, 3/3 abstained, 0 hallucinations across 1,853
chunks of RFC 9110/9112/1035/5321 (~938 KB, freely redistributable
under the IETF Trust license; fetched by `scripts/make_corpus_large.py`,
committed so tests stay offline).

The large set also reports **hard traps** (plausible, absent facts like
"SSH port" that share vocabulary with the corpus): a purely lexical
judge cannot reject them — that's the measured ceiling, and Checkpoints
2-3 (LLM judge, embeddings) are judged against clearing it.

## Tests

```bash
python3 tests/test_rag.py    # 9 offline tests
```

Covers ingest, porter-aligned stemming (repotted → repot), table-aware
chunking, citation format, scaffolding-tolerant queries, the abstain
gate, the abstention log, and the eval harness integration.

## Layout

- `rag.py` — ingest (table-aware chunking) / IDF-scored retrieve / judge
  / rewrite / abstain / log
- `eval.py` — labeled gold-set harness with hard gates (toy + RFC-large)
- `cli.py` — build + ask
- `fixtures/corpus/` — 3 synthetic documents (handbook, product notes,
  plant guide; all fictitious)
- `fixtures/corpus_large/` + `eval_set_large.json` — the RFC corpus and
  its 30-case labeled gold set (facts grep-verified against the texts)
- `scripts/make_corpus_large.py` — RFC fetcher/cleaner (provenance)
- `tests/test_rag.py` — offline suite
- `PLAN.md` — problem, goals, 3 checkpoints, rollback

## Honest ceiling

Pure lexical retrieval fails on paraphrase ("car" never matches
"vehicle"). That ceiling is documented in PLAN.md (Checkpoint 3) and
measured by the eval harness rather than hidden behind an embedding
index you can't inspect.

## License

MIT
