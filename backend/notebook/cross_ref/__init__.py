"""
Cross-reference and consistency checking engine for Notebook.

Pipelines:
  1. Entity extraction — regex-based theorem/equation/citation/claim detection
  2. Equation comparison — SymPy canonical-form matching
  3. Claim consistency — LLM-powered contradiction/confirmation detection
  4. Cross-project search — vector similarity over pgvector (future)

Usage (CLI MVP):
    python -m notebook.cross_ref consistency file_a.md file_b.md
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("notebook.cross_ref")


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════


class EntityType(str, Enum):
    THEOREM = "theorem"
    DEFINITION = "definition"
    EQUATION = "equation"
    CITATION = "citation"
    CLAIM = "claim"
    VARIABLE = "variable"


class RelationType(str, Enum):
    CONFIRMS = "confirms"         # two equations are identical or equivalent
    CONTRADICTS = "contradicts"   # two claims/equations disagree
    GENERALIZES = "generalizes"   # one extends another
    CITES = "cites"               # a references b
    USES = "uses"                 # a depends on b


@dataclass
class Entity:
    """A single extracted entity from a document chunk."""
    id: str                     # e.g. "eq_3", "thm_1"
    type: EntityType
    raw_text: str               # original LaTeX/markdown
    display_text: str           # human-readable form
    canonical_form: str = ""    # SymPy-normalized (equations) or normalized name
    line_number: int = 0
    project: str = ""           # which document this came from
    section: str = ""           # e.g. "§2.1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossReference:
    """A relationship between two entities across (or within) documents."""
    source: Entity
    target: Entity
    relation: RelationType
    confidence: float           # 0.0–1.0
    evidence: str               # human-readable justification
    suggested_action: str = ""  # "cite", "resolve", "review", "none"


@dataclass
class ConsistencyReport:
    """Full output of a consistency check between two documents."""
    file_a: str
    file_b: str
    entities_a: list[Entity] = field(default_factory=list)
    entities_b: list[Entity] = field(default_factory=list)
    cross_refs: list[CrossReference] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            f"📄 Comparing: {self.file_a} ↔ {self.file_b}",
            f"   Entities: {len(self.entities_a)} (A) + {len(self.entities_b)} (B)",
            "",
        ]
        for ref in self.cross_refs:
            icon = {
                RelationType.CONFIRMS: "✅",
                RelationType.CONTRADICTS: "⚠️",
                RelationType.GENERALIZES: "🔗",
                RelationType.CITES: "📎",
                RelationType.USES: "🧩",
            }.get(ref.relation, "•")

            lines.append(
                f"  {icon} {ref.relation.value.upper()}: "
                f"'{ref.source.display_text[:60]}...' "
                f"↔ '{ref.target.display_text[:60]}...'"
            )
            if ref.evidence:
                lines.append(f"     └─ {ref.evidence}")
            if ref.suggested_action and ref.suggested_action != "none":
                lines.append(f"     💡 {ref.suggested_action}")
            lines.append("")

        # Summary
        lines.append("📊 Summary:")
        for k, v in self.summary.items():
            lines.append(f"   {k}: {v}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Entity extraction (regex-based, V1)
# ══════════════════════════════════════════════════════════════════════

# LaTeX structural patterns — captures ~70% of academic entities
THEOREM_PATTERN = re.compile(
    r'\\begin\{(theorem|lemma|proposition|corollary|conjecture)\}'
    r'(.*?)'
    r'\\end\{\1\}',
    re.DOTALL,
)
DEFINITION_PATTERN = re.compile(
    r'\\begin\{definition\}(.*?)\\end\{definition\}',
    re.DOTALL,
)
EQUATION_BLOCK_PATTERN = re.compile(
    r'\$\$(.+?)\$\$|\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}',
    re.DOTALL,
)
EQUATION_INLINE_PATTERN = re.compile(
    r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)',
)
CITATION_PATTERN = re.compile(
    r'\\cite\{([^}]+)\}|\\citep\{([^}]+)\}|\\citet\{([^}]+)\}',
)
LABEL_PATTERN = re.compile(r'\\label\{([^}]+)\}')
REF_PATTERN = re.compile(r'\\ref\{([^}]+)\}|\\eqref\{([^}]+)\}')

# Claim detection: sentences containing claim-indicating words
CLAIM_KEYWORDS = [
    r'\bwe (propose|show|demonstrate|prove|establish|argue|conjecture)\b',
    r'\bthis (shows|demonstrates|proves|establishes|suggests|indicates)\b',
    r'\bour (result|finding|contribution|main theorem|key insight)\b',
    r'\bin contrast\b', r'\bhowever\b', r'\btherefore\b', r'\bthus\b',
    r'\bimportantly\b', r'\bsignificantly\b',
]
CLAIM_PATTERN = re.compile('|'.join(CLAIM_KEYWORDS), re.IGNORECASE)


def extract_entities(
    markdown: str,
    project: str = "",
    section: str = "",
) -> list[Entity]:
    """Extract all structured entities from a Markdown/LaTeX document.

    This is the V1 regex-based extractor. It captures ~70% of entities.
    The remaining 30% (informal claims, inline math in prose) are accepted
    as a gap for V1.
    """
    entities: list[Entity] = []
    counter: dict[EntityType, int] = {t: 0 for t in EntityType}

    def _next_id(etype: EntityType) -> str:
        counter[etype] += 1
        return f"{etype.value}_{counter[etype]}"

    # ── Theorems ──
    for match in THEOREM_PATTERN.finditer(markdown):
        body = match.group(2).strip()
        etype = match.group(1)  # theorem, lemma, etc.
        entities.append(Entity(
            id=_next_id(EntityType.THEOREM),
            type=EntityType.THEOREM,
            raw_text=match.group(0),
            display_text=body[:120],
            canonical_form=_normalize_text(body),
            project=project,
            section=section,
            metadata={"latex_env": etype},
        ))

    # ── Definitions ──
    for match in DEFINITION_PATTERN.finditer(markdown):
        body = match.group(1).strip()
        entities.append(Entity(
            id=_next_id(EntityType.DEFINITION),
            type=EntityType.DEFINITION,
            raw_text=match.group(0),
            display_text=body[:120],
            canonical_form=_normalize_text(body),
            project=project,
            section=section,
        ))

    # ── Equations ──
    # Block equations: $$...$$ or \begin{equation}...\end{equation}
    for match in EQUATION_BLOCK_PATTERN.finditer(markdown):
        eq = (match.group(1) or match.group(2) or "").strip()
        if len(eq) < 3:
            continue
        canonical = _sympy_canonical(eq) if _has_sympy() else eq.strip()
        entities.append(Entity(
            id=_next_id(EntityType.EQUATION),
            type=EntityType.EQUATION,
            raw_text=eq,
            display_text=eq[:120],
            canonical_form=canonical,
            project=project,
            section=section,
        ))

    # Inline equations: $...$
    for match in EQUATION_INLINE_PATTERN.finditer(markdown):
        eq = match.group(1).strip()
        if len(eq) < 3:
            continue
        canonical = _sympy_canonical(eq) if _has_sympy() else eq.strip()
        entities.append(Entity(
            id=_next_id(EntityType.EQUATION),
            type=EntityType.EQUATION,
            raw_text=eq,
            display_text=eq[:120],
            canonical_form=canonical,
            project=project,
            section=section,
        ))

    # ── Citations ──
    for match in CITATION_PATTERN.finditer(markdown):
        cites = [c for c in match.groups() if c]
        for cite_key in cites:
            for key in cite_key.split(","):
                key = key.strip()
                if key:
                    entities.append(Entity(
                        id=_next_id(EntityType.CITATION),
                        type=EntityType.CITATION,
                        raw_text=key,
                        display_text=key,
                        canonical_form=key.lower(),
                        project=project,
                        section=section,
                    ))

    # ── Claims ──
    # Split into sentences and match claim patterns
    sentences = re.split(r'(?<=[.!?])\s+', markdown)
    line_start = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if CLAIM_PATTERN.search(sentence) and len(sentence) > 30:
            line_start += 1
            entities.append(Entity(
                id=_next_id(EntityType.CLAIM),
                type=EntityType.CLAIM,
                raw_text=sentence,
                display_text=sentence[:150],
                canonical_form=_normalize_text(sentence),
                line_number=line_start,
                project=project,
                section=section,
            ))

    return entities


def _normalize_text(text: str) -> str:
    """Strip whitespace, LaTeX commands, and punctuation for fuzzy matching."""
    text = re.sub(r'\\[a-zA-Z]+(\{.*?\})*', '', text)  # remove LaTeX commands
    text = re.sub(r'[^\w\s]', ' ', text)                # remove punctuation
    text = re.sub(r'\s+', ' ', text)                    # collapse whitespace
    return text.strip().lower()


# ══════════════════════════════════════════════════════════════════════
# SymPy integration — equation canonical-form comparison
# ══════════════════════════════════════════════════════════════════════

def _has_sympy() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except ImportError:
        return False


def _sympy_canonical(latex: str) -> str:
    """Convert LaTeX equation to a canonical token string for comparison.

    Uses a simplified approach:
    1. Strip LaTeX commands → keep variable names and operators
    2. Normalize whitespace and ordering
    3. Try SymPy simplification for simple expressions

    This prioritizes recall (finding matches) over precision (formal verification).
    """
    # Try SymPy parse_latex if available
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex)
        from sympy import simplify as sp_simplify
        return str(sp_simplify(expr))
    except Exception:
        pass

    # Manual tokenization: strip LaTeX commands, keep core symbols
    s = latex.strip()
    # Remove LaTeX commands like \varepsilon, \text, \frac, etc.
    s = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', s)
    # Remove braces
    s = s.replace('{', '').replace('}', '')
    # Normalize operators
    s = s.replace('^', '**').replace('_', '_')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _manual_latex_convert(latex: str) -> str | None:
    """Manual conversion of common LaTeX to SymPy-compatible Python expressions.

    Adapted from exobrain's verify.py — handles the most common
    LaTeX patterns in STEM papers.
    """
    s = latex.strip()

    # Implicit multiplication
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 2x → 2*x
    s = re.sub(r'\)(\()', r')*(', s)       # (a)(b) → (a)*(b)
    s = re.sub(r'\)([a-zA-Z])', r')*\1', s)  # (a)b → (a)*b
    s = re.sub(r'([a-zA-Z])\(', r'\1*(', s)  # a(b) → a*(b)

    # Handle \\frac{numerator}{denominator} first
    while "\\frac" in s:
        idx = s.index("\\frac")
        brace_open = s.index("{", idx)
        depth, pos = 1, brace_open + 1
        while depth > 0 and pos < len(s):
            if s[pos] == "{": depth += 1
            elif s[pos] == "}": depth -= 1
            pos += 1
        num = s[brace_open + 1:pos - 1]
        if pos < len(s) and s[pos] == "{":
            depth, denom_start = 1, pos + 1
            pos = denom_start
            while depth > 0 and pos < len(s):
                if s[pos] == "{": depth += 1
                elif s[pos] == "}": depth -= 1
                pos += 1
            denom = s[denom_start:pos - 1]
        else:
            break
        s = s[:idx] + f"({num})/({denom})" + s[pos:]

    # Standard replacements
    for old, new in [
        (r"\left", ""), (r"\right", ""), (r"\cdot", "*"), (r"\times", "*"),
        (r"\sqrt{", "sqrt("), (r"\sin", "sin"), (r"\cos", "cos"),
        (r"\tan", "tan"), (r"\log", "log"), (r"\ln", "ln"),
        (r"\exp", "exp"), (r"\pi", "pi"), (r"\infty", "oo"),
        (r"\varepsilon", "varepsilon"), (r"\epsilon", "epsilon"),
        (r"\alpha", "alpha"), (r"\beta", "beta"), (r"\gamma", "gamma"),
        (r"\delta", "delta"), (r"\theta", "theta"), (r"\zeta", "zeta"),
        (r"\eta", "eta"), (r"\xi", "xi"), (r"\rho", "rho"),
        (r"\sigma", "sigma"), (r"\tau", "tau"), (r"\phi", "phi"),
        (r"\chi", "chi"), (r"\psi", "psi"), (r"\omega", "omega"),
        (r"\lambda", "lambda"), (r"\mu", "mu"), (r"\nu", "nu"),
        (r"\Gamma", "Gamma"), (r"\Delta", "Delta"), (r"\Theta", "Theta"),
        (r"\Lambda", "Lambda"), (r"\Sigma", "Sigma"), (r"\Omega", "Omega"),
        (r"\mathbf{", ""), (r"\mathcal{", ""),
        (r"\mathbb{", ""), (r"\operatorname{", ""), (r"\text{", ""),
        (r"\quad", " "), (r"\qquad", "  "), (r"\\{", "("), (r"\\}", ")"),
        ("{", "("), ("}", ")"),
        (r"\pm", " "), (r"\mp", " "), (r"\to", "->"),
        (r"\rightarrow", "->"), (r"\Rightarrow", "=>"),
        (r"\neq", "!="), (r"\leq", "<="), (r"\geq", ">="),
        (r"\approx", "~="), (r"\equiv", "=="), (r"\propto", "~"),
        (r"\dot{", "diff("), (r"\hat{", ""), (r"\bar{", ""), (r"\vec{", ""),
        ("^T", "**T"),
    ]:
        s = s.replace(old, new)

    # Balance braces
    while "(" in s and s.count("(") > s.count(")"):
        s += ")"
    while ")" in s and s.count(")") > s.count("("):
        s = "(" + s

    return s if s else None


def compare_equations(
    eq_a: Entity,
    eq_b: Entity,
) -> CrossReference | None:
    """Compare two equation entities for equivalence or contradiction.

    Uses canonical forms pre-computed during entity extraction.
    Falls back to text comparison if canonical forms are empty.
    """
    canon_a = eq_a.canonical_form
    canon_b = eq_b.canonical_form

    # If canonical forms exist and match → confirmed
    if canon_a and canon_b:
        if canon_a == canon_b:
            return CrossReference(
                source=eq_a, target=eq_b,
                relation=RelationType.CONFIRMS,
                confidence=0.95,
                evidence=f"Canonical forms match: {canon_a[:80]}",
                suggested_action="cite",
            )

    # Try SymPy direct comparison as backup
    if _has_sympy():
        try:
            from sympy import simplify
            expr_a = _parse_equation(eq_a.raw_text)
            expr_b = _parse_equation(eq_b.raw_text)
            if expr_a is not None and expr_b is not None:
                diff = simplify(expr_a - expr_b)
                if diff == 0:
                    return CrossReference(
                        source=eq_a, target=eq_b,
                        relation=RelationType.CONFIRMS,
                        confidence=0.95,
                        evidence="Equations are algebraically equivalent (diff=0)",
                        suggested_action="cite",
                    )
        except Exception as e:
            logger.debug(f"SymPy direct comparison failed: {e}")

    # Text-based fuzzy match as last resort
    if canon_a and canon_b and _fuzzy_text_match(canon_a, canon_b) > 0.85:
        return CrossReference(
            source=eq_a, target=eq_b,
            relation=RelationType.CONFIRMS,
            confidence=0.6,
            evidence="Text-normalized forms are very similar",
            suggested_action="review",
        )

    return None


def _parse_equation(latex: str):
    """Try to parse a LaTeX equation, with manual fallback."""
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except Exception:
        pass
    try:
        import sympy as sp
        converted = _manual_latex_convert(latex)
        if converted:
            return sp.sympify(converted)
    except Exception:
        pass
    return None


def _fuzzy_text_match(a: str, b: str) -> float:
    """Simple Jaccard similarity on tokenized text."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ══════════════════════════════════════════════════════════════════════
# LLM-powered claim consistency judge
# ══════════════════════════════════════════════════════════════════════

CONSISTENCY_PROMPT = """You are a research consistency judge. Compare two claims from
academic papers and determine their relationship.

Claim A (from "{project_a}"):
{claim_a}

Claim B (from "{project_b}"):
{claim_b}

Classify the relationship as EXACTLY ONE of:
- CONFIRMS: The claims are consistent and mutually supportive
- CONTRADICTS: The claims disagree — they cannot both be true
- GENERALIZES: Claim A extends or subsumes Claim B (or vice versa)
- UNRELATED: The claims are about different topics

Respond with ONLY a JSON object:
{{"relation": "CONFIRMS|CONTRADICTS|GENERALIZES|UNRELATED", "evidence": "one sentence explaining why", "confidence": 0.0-1.0}}
"""


def _llm_judge_claims(
    claim_a: Entity,
    claim_b: Entity,
    api_key: str = "",
) -> CrossReference | None:
    """Use an LLM to judge the logical relationship between two claims.

    Returns None if claims are unrelated, or a CrossReference with the
    determined relation type.
    """
    prompt = CONSISTENCY_PROMPT.format(
        project_a=claim_a.project or "Document A",
        claim_a=claim_a.raw_text,
        project_b=claim_b.project or "Document B",
        claim_b=claim_b.raw_text,
    )

    response = _call_llm(prompt, api_key=api_key)
    if not response:
        return None

    try:
        data = json.loads(response)
        relation_str = data.get("relation", "UNRELATED")
        if relation_str == "UNRELATED":
            return None

        relation_map = {
            "CONFIRMS": RelationType.CONFIRMS,
            "CONTRADICTS": RelationType.CONTRADICTS,
            "GENERALIZES": RelationType.GENERALIZES,
        }
        relation = relation_map.get(relation_str)
        if not relation:
            return None

        return CrossReference(
            source=claim_a,
            target=claim_b,
            relation=relation,
            confidence=float(data.get("confidence", 0.5)),
            evidence=data.get("evidence", ""),
            suggested_action=(
                "resolve" if relation == RelationType.CONTRADICTS
                else "cite" if relation == RelationType.CONFIRMS
                else "review"
            ),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return None


def _call_llm(prompt: str, api_key: str = "") -> str | None:
    """Call an LLM for claim consistency judgment.

    Uses OPENAI_API_KEY from env. Falls back gracefully.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("No OPENAI_API_KEY set — claim comparison skipped")
        return None

    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            }).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# Main pipeline: consistency check between two documents
# ══════════════════════════════════════════════════════════════════════

def run_consistency_check(
    file_a: str,
    file_b: str,
    api_key: str = "",
) -> ConsistencyReport:
    """Run a full consistency check between two Markdown/LaTeX documents.

    This is the primary entry point. It:
    1. Extracts entities from both documents
    2. Compares equations via SymPy canonical forms
    3. Judges claim relationships via LLM
    4. Matches citations across documents
    5. Returns a structured report with suggestions

    Parameters
    ----------
    file_a : str
        Path to the first Markdown/LaTeX document.
    file_b : str
        Path to the second Markdown/LaTeX document.
    api_key : str
        OpenAI API key for LLM claim comparison. If empty, claims are skipped.

    Returns
    -------
    ConsistencyReport
        Structured report with all cross-references, summary, and suggestions.
    """
    text_a = _read_file(file_a)
    text_b = _read_file(file_b)

    entities_a = extract_entities(text_a, project=file_a)
    entities_b = extract_entities(text_b, project=file_b)

    report = ConsistencyReport(
        file_a=file_a,
        file_b=file_b,
        entities_a=entities_a,
        entities_b=entities_b,
    )
    cross_refs: list[CrossReference] = []

    # ── Equation comparison (deterministic, SymPy) ──
    eqs_a = [e for e in entities_a if e.type == EntityType.EQUATION]
    eqs_b = [e for e in entities_b if e.type == EntityType.EQUATION]
    for ea in eqs_a:
        for eb in eqs_b:
            result = compare_equations(ea, eb)
            if result:
                cross_refs.append(result)

    # ── Citation matching ──
    cites_a = {e.canonical_form: e for e in entities_a if e.type == EntityType.CITATION}
    cites_b = {e.canonical_form: e for e in entities_b if e.type == EntityType.CITATION}
    shared = set(cites_a) & set(cites_b)
    if shared:
        for key in shared:
            cross_refs.append(CrossReference(
                source=cites_a[key], target=cites_b[key],
                relation=RelationType.CITES,
                confidence=0.99,
                evidence=f"Both documents cite: {key}",
                suggested_action="review",
            ))

    # ── Claim comparison (LLM, only if API key available) ──
    claims_a = [e for e in entities_a if e.type == EntityType.CLAIM]
    claims_b = [e for e in entities_b if e.type == EntityType.CLAIM]
    if api_key and claims_a and claims_b:
        # Only compare promising pairs (fuzzy pre-filter)
        for ca in claims_a[:5]:   # limit to 5 claims per doc for speed
            for cb in claims_b[:5]:
                # Quick pre-filter: any word overlap?
                words_a = set(ca.canonical_form.split())
                words_b = set(cb.canonical_form.split())
                overlap = words_a & words_b
                if len(overlap) < 3:
                    continue  # skip unrelated claims
                result = _llm_judge_claims(ca, cb, api_key=api_key)
                if result:
                    cross_refs.append(result)

    report.cross_refs = cross_refs
    report.summary = {
        "total_cross_refs": len(cross_refs),
        "confirms": sum(1 for r in cross_refs if r.relation == RelationType.CONFIRMS),
        "contradictions": sum(1 for r in cross_refs if r.relation == RelationType.CONTRADICTS),
        "generalizations": sum(1 for r in cross_refs if r.relation == RelationType.GENERALIZES),
        "citations": sum(1 for r in cross_refs if r.relation == RelationType.CITES),
        "uses": sum(1 for r in cross_refs if r.relation == RelationType.USES),
        "entities_extracted": len(entities_a) + len(entities_b),
        "equations_compared": len(eqs_a) * len(eqs_b),
        "claims_compared": len(claims_a) * len(claims_b) if api_key else 0,
    }
    return report


def _read_file(path: str) -> str:
    """Read a file, supporting both .md and .tex extensions."""
    with open(path) as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Notebook Cross-Reference Consistency Checker"
    )
    parser.add_argument("file_a", help="First Markdown/LaTeX document")
    parser.add_argument("file_b", help="Second Markdown/LaTeX document")
    parser.add_argument(
        "--api-key", default="",
        help="OpenAI API key for LLM claim comparison (or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of formatted text",
    )
    args = parser.parse_args()

    report = run_consistency_check(args.file_a, args.file_b, api_key=args.api_key)

    if args.json:
        print(json.dumps({
            "file_a": report.file_a,
            "file_b": report.file_b,
            "summary": report.summary,
            "cross_refs": [
                {
                    "source": asdict(r.source),
                    "target": asdict(r.target),
                    "relation": r.relation.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "suggested_action": r.suggested_action,
                }
                for r in report.cross_refs
            ],
        }, indent=2, default=str))
    else:
        print(report.format())


if __name__ == "__main__":
    main()
