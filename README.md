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
python3 eval.py
```

- 6 answerable queries with expected doc + fragment
- 5 unanswerable queries that must abstain
- **Hard gates:** zero confident hallucinations, zero missed answerable.
  Current: 6/6 grounded, 5/5 abstained, 0 hallucinations.

## Tests

```bash
python3 tests/test_rag.py    # 9 offline tests
```

Covers ingest, porter-aligned stemming (repotted → repot), citation
format, scaffolding-tolerant queries, the abstain gate, the abstention
log, and the eval harness integration.

## Layout

- `rag.py` — ingest / retrieve / judge / rewrite / abstain / log
- `eval.py` — labeled gold-set harness with hard gates
- `cli.py` — build + ask
- `fixtures/corpus/` — 3 synthetic documents (handbook, product notes,
  plant guide; all fictitious)
- `fixtures/eval_set.json` — the labeled gold set
- `tests/test_rag.py` — offline suite
- `PLAN.md` — problem, goals, 3 checkpoints, rollback

## Honest ceiling

Pure lexical retrieval fails on paraphrase ("car" never matches
"vehicle"). That ceiling is documented in PLAN.md (Checkpoint 3) and
measured by the eval harness rather than hidden behind an embedding
index you can't inspect.

## License

MIT
