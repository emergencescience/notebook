# Notebook — AI-Native Research Workspace

A document-first research notebook that verifies your math and
connects your ideas across papers.

## Quick Start

```bash
cd backend
pip install -e .
python -m notebook.cross_ref paper_a.md paper_b.md
```

## Cross-Reference Engine

The `notebook.cross_ref` module extracts entities from academic documents
and finds cross-references between them:

1. **Entity extraction** — theorems, equations, citations, claims (regex-based, V1)
2. **Equation comparison** — SymPy canonical-form matching
3. **Claim consistency** — LLM-powered contradiction/confirmation detection
4. **Citation matching** — shared bibliographic references

```bash
# Basic usage
python -m notebook.cross_ref paper.md notes.md

# With LLM claim comparison
OPENAI_API_KEY=sk-... python -m notebook.cross_ref paper.md notes.md

# JSON output
python -m notebook.cross_ref paper.md notes.md --json
```

## Architecture

```
notebook/
  cross_ref/
    __init__.py    # Entity extraction + consistency checking
    ...
  verify.py        # SymPy equation verification (from exobrain)
  storage/         # PostgreSQL/SQLite persistence
```

## Status

**Phase 1: Core cross-reference engine** ✅
- Entity extraction (theorems, definitions, equations, citations, claims)
- Equation comparison via SymPy canonical forms
- Citation matching across documents
- CLI MVP with formatted and JSON output

**Next: LLM claim consistency + pgvector indexing**
