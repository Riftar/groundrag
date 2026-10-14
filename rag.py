"""groundrag - self-correcting retrieval with a hard abstain gate, stdlib only.

Pipeline: query -> [retrieve: FTS5/BM25] -> [judge: grounding score]
  grounded  -> answer with citations
  partial   -> rewrite query once, re-retrieve
  weak      -> ABSTAIN (logged; never guess)

The judge is a callable. The default one is deterministic (lexical overlap
+ bm25); Checkpoint 2 swaps in an LLM judge with the same interface.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "how", "what", "which", "who", "when", "where", "why", "can", "could",
    "should", "would", "i", "you", "we", "it", "in", "on", "for", "of",
    "to", "and", "or", "with", "by", "at", "as", "be", "been", "my",
    # conversational scaffolding: dilutes overlap without adding meaning
    "so", "tell", "me", "please", "thanks", "thank", "hi", "hey",
    "hello", "okay", "ok", "well", "like", "know", "want", "need",
    "get", "got", "just", "really",
}

CHUNK_TARGET = 600  # merge paragraphs up to ~this many chars


def _stem(w: str) -> str:
    """Tiny suffix stripper aligned (approximately) with porter: handles
    plural 's', 'ing'/'ed' with the doubled-consonant rule, so 'repotted'
    and 'repotting' both reduce to 'repot' like FTS5's porter tokenizer."""
    for suf in ("ing", "ed", "es"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            # un-doubled consonant: repott -> repot, running -> runn->run
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def terms(text: str) -> list[str]:
    """Content terms: lowercase words, stopwords dropped, plural-folded."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        out.append(_stem(w))
    return out


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def _group_tables(paras: list[str]) -> list[str]:
    """Merge runs of short single-line paragraphs (table rows separated by
    blank lines) into single blocks so tables survive chunking."""
    out: list[str] = []
    run: list[str] = []
    for p in paras:
        single = "\n" not in p and len(p) < 80
        if single:
            run.append(p)
        else:
            if run:
                out.append("\n".join(run))
                run = []
            out.append(p)
    if run:
        out.append("\n".join(run))
    return out


def chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunking: split on blank lines, keep tables whole,
    merge small paragraphs up to ~CHUNK_TARGET chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = _group_tables(paras)
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > CHUNK_TARGET:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def ingest(corpus_dir: str, db_path: str) -> int:
    """Index every .md/.txt in corpus_dir into an FTS5 database."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5("
        "doc, idx, text, tokenize='porter unicode61')")
    n = 0
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith((".md", ".txt")):
            continue
        with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            conn.execute("INSERT INTO chunks (doc, idx, text) VALUES (?,?,?)",
                         (name, i, chunk))
            n += 1
    conn.commit()
    conn.close()
    return n


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

@dataclass
class Hit:
    doc: str
    idx: int
    text: str
    score: float          # bm25 (negative = better)
    overlap: float = 0.0  # query-term coverage in chunk

    @property
    def citation(self) -> str:
        return f"{self.doc}#{self.idx}"


def _match_expr(q_terms: list[str]) -> str:
    """Build a safe FTS5 MATCH expression from sanitized terms."""
    quoted = [f'"{t}"' for t in q_terms if re.fullmatch(r"[a-z0-9]+", t)]
    return " OR ".join(quoted) if quoted else '""'


def retrieve(db_path: str, query: str, k: int = 3) -> list[Hit]:
    q_terms = terms(query)
    if not q_terms:
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT doc, idx, text, bm25(chunks) FROM chunks "
            "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (_match_expr(q_terms), k),
        ).fetchall()
    finally:
        conn.close()
    hits = [Hit(d, i, t, s) for d, i, t, s in rows]
    for h in hits:
        chunk_terms = set(terms(h.text))
        present = sum(1 for t in set(q_terms) if t in chunk_terms)
        h.overlap = present / len(set(q_terms))
    return hits


# --------------------------------------------------------------------------
# Judge + rewrite + abstain
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    status: str                 # grounded | abstain
    answer: str | None
    citations: list[str] = field(default_factory=list)
    iterations: int = 1
    best_overlap: float = 0.0
    query_used: str = ""
    log: list[str] = field(default_factory=list)


GROUNDED_MIN = 1 / 3    # >= 1/3 of query-term MASS in the best chunk
REWRITE_MIN = 0.15      # some signal worth one rewrite attempt
MAX_ITERATIONS = 2


def _df(db_path: str, q_terms: list[str]) -> dict[str, int]:
    """Document frequency per query term across the indexed chunks."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df = {}
    try:
        for t in set(q_terms):
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE chunks MATCH ?",
                (f'"{t}"',)).fetchone()
            df[t] = row[0] if row else 0
    finally:
        conn.close()
    return df


def score_hits(db_path: str, hits: list[Hit], q_terms: list[str]) -> None:
    """Set hit.overlap to the IDF-weighted fraction of query-term mass
    present in each chunk. Rare terms carry the verdict; terms that appear
    everywhere (port, number, server) carry almost none. This is what
    separates 'all words present in some unrelated chunk' from grounding."""
    df = _df(db_path, q_terms)
    n_chunks = None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    weights = {}
    if n_chunks >= 50:
        # IDF needs corpus statistics to mean anything; on small corpora
        # (toy sets, single docs) fall back to unweighted overlap.
        for t in set(q_terms):
            weights[t] = math.log(1 + n_chunks / (1 + df[t]))
    else:
        weights = {t: 1.0 for t in set(q_terms)}
    total_w = sum(weights.values()) or 1.0
    for h in hits:
        chunk_terms = set(terms(h.text))
        hit_w = sum(w for t, w in weights.items() if t in chunk_terms)
        h.overlap = hit_w / total_w


def judge(hits: list[Hit], query: str) -> str:
    """Deterministic grounding judge: IDF-weighted term-mass coverage."""
    if not hits:
        return "abstain"
    best = max(h.overlap for h in hits)
    if best >= GROUNDED_MIN:
        return "grounded"
    if best >= REWRITE_MIN:
        return "rewrite"
    return "abstain"


def rewrite_query(query: str) -> str:
    """Deterministic lexical rewrite: content terms only (drops question
    scaffolding), which widens the FTS5 OR-match. An LLM judge at
    Checkpoint 2 replaces this with semantic rewriting."""
    return " ".join(terms(query))


def ask(db_path: str, query: str, k: int = 3,
        judge_fn=None, rewrite_fn=None) -> Verdict:
    """The self-correcting loop. Never guesses: weak grounding abstains."""
    judge_fn = judge_fn or judge
    rewrite_fn = rewrite_fn or rewrite_query
    v = Verdict("abstain", None, query_used=query)
    current = query
    for iteration in range(1, MAX_ITERATIONS + 1):
        v.iterations = iteration
        hits = retrieve(db_path, current, k)
        q_terms = terms(current)
        if hits:
            score_hits(db_path, hits, q_terms)
            v.best_overlap = max(h.overlap for h in hits)
        verdict = judge_fn(hits, current)
        v.log.append(f"iter{iteration}: q='{current}' best_overlap="
                     f"{v.best_overlap:.2f} -> {verdict}")
        if verdict == "grounded":
            v.status = "grounded"
            v.citations = [h.citation for h in hits]
            # excerpt-style answer: all retrieved chunks, best-ranked first
            v.answer = "\n\n---\n\n".join(h.text for h in hits)
            v.query_used = current
            return v
        if verdict == "rewrite" and iteration < MAX_ITERATIONS:
            current = rewrite_fn(current)
            continue
        break
    v.status = "abstain"
    v.query_used = current
    return v


# --------------------------------------------------------------------------
# Abstention log (the audit trail)
# --------------------------------------------------------------------------

def log_abstain(log_path: str, v: Verdict, original_query: str) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": original_query,
        "rewritten": v.query_used,
        "best_overlap": round(v.best_overlap, 3),
        "iterations": v.iterations,
        "log": v.log,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
