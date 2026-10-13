# PLAN — groundrag

## The real problem

RAG systems fail in the worst way: confidently. When retrieval is weak
they answer anyway, inventing details the corpus never contained. The
hard part is not generation — it is the discipline: knowing when the
retrieved evidence does not actually ground an answer, rewriting the
question when it was asked badly, and refusing to answer otherwise.
Frameworks hide this behind "top-k chunks + prompt", which is exactly
where hallucination is born.

**groundrag's answer:** the self-correcting loop as a small, readable
state machine over SQLite FTS5 (in the Python standard library) with an
explicit abstain gate and an audit log. Every decision — retrieve, judge,
rewrite, abstain — is visible and deterministic in Checkpoint 1.

## Goals — what this can become at its highest

The ceiling is a **dependency-free retrieval core with calibrated
refusal**:

- Any directory of markdown/text becomes a queryable knowledge base in
  one command, on any machine with CPython — no vector DB, no embedding
  service, no framework.
- A judge interface (`judge_fn`) that is deterministic today (lexical
  grounding score) and swappable for an LLM tomorrow without touching
  the loop.
- A labeled eval harness (`eval.py`) with two hard gates — zero
  confident hallucinations, zero missed answerable — so every change to
  thresholds or judges is judged against the same bar, offline.
- An abstention log (jsonl) that accumulates every refusal with its
  scores, turning "when does it not know" into inspectable data.

## Non-goals (deliberate)

- No generation step. Checkpoint 1 returns grounded excerpts with
  citations; prose generation is the LLM's job downstream.
- No embeddings/vector search at Checkpoint 1 — BM25 over a real stemmer
  is the honest baseline; the eval harness will say when that ceiling is
  hit (paraphrase queries), and Checkpoint 3 addresses it.
- No server, no API. CLI + library.

## Checkpoint 1 — Deterministic loop + eval gates (DONE in this commit)

`rag.py`: ingest (paragraph-aware chunking → FTS5 with porter), retrieve
(BM25-ranked, term-overlap scored with a porter-aligned stemmer), judge
(overlap thresholds: grounded / rewrite / abstain), one-step lexical
rewrite loop (max 2 iterations), excerpt answers with citations, jsonl
abstention log. `eval.py`: labeled gold set (6 answerable with expected
doc+fragment, 5 unanswerable), metrics table, hard gates. `cli.py`:
build + ask. Result on fixtures: 6/6 answerable grounded with correct
citations, 5/5 unanswerable abstained, 0 confident hallucinations.

## Checkpoint 2 — LLM judge + semantic rewrite (NEXT)

- Swap `judge_fn` for an LLM-backed judge (local llama.cpp subprocess or
  remote API; the interface is one callable returning
  grounded/rewrite/abstain).
- Semantic query rewriting in place of the lexical strip; eval must not
  regress on the abstain gate.
- Prose answer generation from the grounded excerpts, with citations
  preserved; generation only ever runs on grounded verdicts.

## Checkpoint 3 — Paraphrase ceiling + hybrid retrieval

- Add paraphrase-variant queries to the eval set (where pure BM25 is
  expected to fail: "car" vs "vehicle") and measure the ceiling honestly.
- Optional embedding leg via a local CLI embedder (same pattern as
  whisper.cpp/piper adapters: a subprocess, not a Python dependency),
  fused with BM25; gated by the same eval harness.

## Rollback / recovery

Greenfield library: rollback is `git revert`. The index is a disposable
sqlite file rebuilt from the corpus by `--build`; the abstention log is
append-only telemetry. Nothing runs as a service, nothing mutates the
corpus.

## Offline guarantee

Stdlib only (`sqlite3` with FTS5, `re`, `json`, `dataclasses`). Corpus,
eval set, and tests all ship in `fixtures/`; the whole loop — index,
query, judge, eval — runs with `python3` alone on an air-gapped machine.
The Checkpoint 2/3 adapters invoke local CLI binaries by choice, never
Python dependencies.
