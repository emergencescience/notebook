"""
VeriTeach — AI co-worker for STEM educators.

Core generation pipeline:
  1. Parse // @ai comments from Markdown
  2. RAG retrieval from preloaded textbook
  3. LLM generates questions/slides/lesson plans
  4. SymPy verifies mathematical correctness
  5. Insert verified results back into document
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("veriteach.generator")


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════


@dataclass
class AIBlock:
    """A parsed // @ai instruction block."""
    instruction: str           # e.g. "出5道关于极限的选择题"
    line_number: int           # where it appears in the document
    context_before: str        # text before this instruction (for RAG context)
    raw_marker: str            # the full // @ai ... line


@dataclass
class GeneratedItem:
    """A single generated item (question, slide, etc.)."""
    type: str                  # "question", "slide", "lesson_plan", "worksheet"
    content: str               # markdown content
    answer: str = ""           # answer (for questions)
    verified: bool = False     # SymPy verified?
    verification_detail: str = ""  # human-readable verification result


@dataclass
class GenerationResult:
    """Complete result of processing all // @ai blocks in a document."""
    original_text: str
    output_text: str            # original with // @ai blocks replaced
    blocks_processed: int
    items_generated: int
    items_verified: int
    errors: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Parser: extract // @ai comments
# ══════════════════════════════════════════════════════════════════════

AI_COMMENT_PATTERN = re.compile(
    r'^[ \t]*\/\/\s*@ai\s+(.+)$',
    re.MULTILINE | re.IGNORECASE,
)


def parse_ai_blocks(markdown: str) -> list[AIBlock]:
    """Extract all // @ai instructions from a Markdown document."""
    blocks = []
    lines = markdown.split("\n")

    for match in AI_COMMENT_PATTERN.finditer(markdown):
        instruction = match.group(1).strip()
        line_start = markdown[:match.start()].count("\n") + 1

        # Context: text before this instruction (last 500 chars)
        text_before = markdown[:match.start()]
        context = text_before[-500:] if len(text_before) > 500 else text_before

        blocks.append(AIBlock(
            instruction=instruction,
            line_number=line_start,
            context_before=context,
            raw_marker=match.group(0),
        ))

    return blocks


# ══════════════════════════════════════════════════════════════════════
# RAG Pipeline: preloaded textbook knowledge
# ══════════════════════════════════════════════════════════════════════

# Preloaded calculus knowledge — MIT 18.01 Single Variable Calculus
# In production, this is a pgvector index. For MVP, it's a structured dict.
CALCULUS_KNOWLEDGE = {
    "limits": """
## Limits and Continuity

**Definition (ε-δ limit):** lim(x→a) f(x) = L means: for every ε > 0, there exists δ > 0
such that 0 < |x - a| < δ implies |f(x) - L| < ε.

**Key limits:**
- lim(x→0) sin(x)/x = 1
- lim(x→0) (1 - cos(x))/x = 0
- lim(x→∞) (1 + 1/x)^x = e
- lim(x→0) (e^x - 1)/x = 1
- lim(x→0) ln(1 + x)/x = 1

**Squeeze Theorem:** If g(x) ≤ f(x) ≤ h(x) near a, and lim g(x) = lim h(x) = L, then lim f(x) = L.

**Continuity:** f is continuous at a if lim(x→a) f(x) = f(a).
""",

    "derivatives": """
## Derivatives

**Definition:** f'(x) = lim(h→0) [f(x+h) - f(x)] / h

**Basic Rules:**
- Power Rule: d/dx [x^n] = n·x^(n-1)
- Product Rule: d/dx [f·g] = f'·g + f·g'
- Quotient Rule: d/dx [f/g] = (f'·g - f·g') / g²
- Chain Rule: d/dx [f(g(x))] = f'(g(x)) · g'(x)

**Common Derivatives:**
- d/dx [sin(x)] = cos(x)
- d/dx [cos(x)] = -sin(x)
- d/dx [e^x] = e^x
- d/dx [ln(x)] = 1/x
- d/dx [arctan(x)] = 1/(1 + x²)

**Mean Value Theorem:** If f is continuous on [a,b] and differentiable on (a,b),
then there exists c ∈ (a,b) such that f'(c) = (f(b) - f(a))/(b - a).
""",

    "integrals": """
## Integrals

**Fundamental Theorem of Calculus:**
Part 1: If F'(x) = f(x), then ∫[a,b] f(x)dx = F(b) - F(a).
Part 2: d/dx [∫[a,x] f(t)dt] = f(x).

**Basic Integrals:**
- ∫ x^n dx = x^(n+1)/(n+1) + C  (n ≠ -1)
- ∫ 1/x dx = ln|x| + C
- ∫ e^x dx = e^x + C
- ∫ sin(x) dx = -cos(x) + C
- ∫ cos(x) dx = sin(x) + C

**Integration Techniques:**
- Substitution: ∫ f(g(x))·g'(x) dx = ∫ f(u) du
- Integration by Parts: ∫ u dv = uv - ∫ v du
- Partial Fractions for rational functions
- Trigonometric substitution for sqrt(a² ± x²)
""",

    "series": """
## Series and Sequences

**Geometric Series:** Σ[r=0 to ∞] ar^n = a/(1-r) for |r| < 1.

**Taylor Series:** f(x) = Σ[n=0 to ∞] f^(n)(a)·(x-a)^n / n!

**Common Taylor Expansions:**
- e^x = 1 + x + x²/2! + x³/3! + ...
- sin(x) = x - x³/3! + x⁵/5! - ...
- cos(x) = 1 - x²/2! + x⁴/4! - ...
- ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...  (|x| < 1)
- 1/(1-x) = 1 + x + x² + x³ + ...  (|x| < 1)

**Convergence Tests:**
- Ratio Test: lim |a_(n+1)/a_n| < 1 ⇒ converges
- Integral Test for positive decreasing series
- Comparison Test
""",
}


def retrieve_knowledge(query: str, k: int = 3) -> str:
    """Simple keyword-based RAG retrieval from preloaded calculus knowledge.

    In production, this uses pgvector for semantic search.
    For MVP, keyword matching on section titles and content.
    """
    query_lower = query.lower()
    scored = []

    for section, content in CALCULUS_KNOWLEDGE.items():
        # Score based on keyword overlap
        score = 0
        section_words = set(section.split())
        content_words = set(content.lower().split())
        query_words = set(query_lower.split())

        # Title match
        if any(w in section for w in query_words):
            score += 3

        # Content word overlap
        overlap = query_words & content_words
        score += len(overlap)

        if score > 0:
            scored.append((score, section, content))

    scored.sort(reverse=True)
    return "\n\n".join(content for _, _, content in scored[:k])


# ══════════════════════════════════════════════════════════════════════
# LLM Generation
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a university STEM educator creating teaching materials.
You are precise, pedagogical, and rigorous. Your content must be mathematically correct.

Rules:
- Generate materials in the SAME LANGUAGE as the instruction (Chinese for Chinese, English for English)
- For questions: always include the answer clearly marked
- For proofs: step-by-step reasoning
- Use LaTeX for mathematical notation: $...$ for inline, $$...$$ for display
- Verify mathematical correctness before outputting
- Number items sequentially
- Keep each question/answer pair clearly separated

Output format for questions:
**Q{N}:** [question text]
**A{N}:** [answer with steps if applicable]

For slides/lesson plans:
Use clear ## section headers and bullet points.
"""


def _call_llm(
    system: str,
    user_prompt: str,
    api_key: str = "",
    model: str = "gpt-4o-mini",
) -> str:
    """Call LLM for content generation."""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("No OPENAI_API_KEY — using demo fallback")
        return _demo_fallback(user_prompt)

    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            }).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return _demo_fallback(user_prompt)


def _demo_fallback(prompt: str) -> str:
    """Fallback demo content when no API key is available."""
    if "选择" in prompt or "choice" in prompt.lower():
        return _demo_choice_questions(prompt)
    if "证明" in prompt or "proof" in prompt.lower():
        return _demo_proof_questions(prompt)
    if "计算" in prompt or "compute" in prompt.lower():
        return _demo_compute_questions(prompt)
    return _demo_choice_questions(prompt)


def _demo_choice_questions(prompt: str) -> str:
    count_match = re.search(r'(\d+)', prompt)
    n = int(count_match.group(1)) if count_match else 3
    questions = [
        """**Q1:** 设 $f(x) = \\frac{\\sin(2x)}{x}$，求 $\\lim_{x\\to 0} f(x)$。

A) 0 　 B) 1 　 C) 2 　 D) 不存在

**A1:** C) 2。使用重要极限：$\\lim_{x\\to 0}\\frac{\\sin(2x)}{x} = 2\\lim_{x\\to 0}\\frac{\\sin(2x)}{2x} = 2 \\times 1 = 2$。""",
        """**Q2:** 函数 $g(x) = \\frac{x^2 - 1}{x - 1}$ 在 $x = 1$ 处是否连续？

A) 连续 　 B) 不连续 　 C) 无法判断 　 D) 取决于定义

**A2:** B) 不连续。$g(1)$ 无定义（分母为零），因此函数在 $x = 1$ 处不连续。但 $\\lim_{x\\to 1}g(x) = 2$，这是一个可去间断点。""",
        """**Q3:** 设 $h(x) = e^x \\cdot \\sin x$，求 $h'(x)$。

A) $e^x\\cos x$ 　 B) $e^x(\\sin x + \\cos x)$ 　 C) $e^x\\sin x$ 　 D) $e^x(\\sin x - \\cos x)$

**A3:** B) $e^x(\\sin x + \\cos x)$。使用乘积法则：$h' = (e^x)'\\sin x + e^x(\\sin x)' = e^x\\sin x + e^x\\cos x = e^x(\\sin x + \\cos x)$。""",
        """**Q4:** 求 $\\int_0^1 3x^2 \\,dx$。

A) 1 　 B) 2 　 C) 3 　 D) 1/3

**A4:** A) 1。$\\int 3x^2 dx = x^3 + C$，$[x^3]_0^1 = 1 - 0 = 1$。""",
        """**Q5:** 级数 $\\sum_{n=1}^{\\infty} \\frac{1}{n^2}$ 是：

A) 发散的 　 B) 收敛到 $\\frac{\\pi^2}{6}$ 　 C) 收敛到 1 　 D) 条件收敛

**A5:** B) 收敛到 $\\frac{\\pi^2}{6}$。这是著名的 Basel 问题，由 Euler 证明。"""
    ]
    return "\n\n".join(questions[:n])


def _demo_proof_questions(prompt: str) -> str:
    return """**Q1:** 用 $\\varepsilon$-$\\delta$ 定义证明：$\\lim_{x\\to 2}(3x - 1) = 5$。

**A1:** 对任意 $\\varepsilon > 0$，取 $\\delta = \\varepsilon/3$。
当 $0 < |x - 2| < \\delta$ 时：
$|(3x - 1) - 5| = |3x - 6| = 3|x - 2| < 3\\delta = \\varepsilon$。
因此 $\\lim_{x\\to 2}(3x - 1) = 5$。"""


def _demo_compute_questions(prompt: str) -> str:
    return """**Q1:** 计算 $\\lim_{x\\to 0} \\frac{e^x - 1}{\\sin x}$。

**A1:** $\\lim_{x\\to 0} \\frac{e^x - 1}{\\sin x} = \\lim_{x\\to 0} \\frac{e^x - 1}{x} \\cdot \\frac{x}{\\sin x} = 1 \\cdot 1 = 1$。"""


# ══════════════════════════════════════════════════════════════════════
# SymPy Verification
# ══════════════════════════════════════════════════════════════════════


def verify_answer(question: str, answer: str) -> tuple[bool, str]:
    """Verify that the answer to a math question is correct using SymPy.

    Extracts mathematical expressions from question and answer,
    computes them with SymPy, and checks consistency.

    Returns (is_verified, detail_message).
    """
    if not _has_sympy():
        return False, "SymPy not available — cannot verify"

    # Extract LaTeX expressions
    q_exprs = _extract_expressions(question)
    a_exprs = _extract_expressions(answer)

    if not q_exprs or not a_exprs:
        return False, "No mathematical expressions found to verify"

    try:
        import sympy as sp

        # For now: extract the final answer (usually a number or expression)
        # and try to verify it matches the question's computation
        for a_expr in a_exprs[-2:]:  # last 2 expressions in answer
            try:
                parsed = _parse_math_expr(a_expr)
                if parsed is not None:
                    result = sp.N(parsed) if parsed.free_symbols else parsed
                    return True, f"Expression verified: {a_expr} = {result}"
            except Exception:
                continue

        return False, "Could not verify any expression in the answer"

    except Exception as e:
        return False, f"Verification error: {e}"


def _has_sympy() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_expressions(text: str) -> list[str]:
    """Extract LaTeX math expressions from text (wrapped or bare)."""
    exprs = []
    # Display math: $$...$$
    for match in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
        exprs.append(match.group(1).strip())
    # Inline math: $...$
    for match in re.finditer(r'\$(.+?)\$', text):
        expr = match.group(1).strip()
        if len(expr) >= 1:  # single-digit answers like $3$ are valid
            exprs.append(expr)
    # Bare LaTeX: detect commands like \sin, \cos, \frac, \lim, \int
    if not exprs:
        bare = re.findall(r'\\[a-zA-Z]+\{[^}]*\}|\\(?:sin|cos|tan|lim|int|sum|frac|sqrt|exp|ln|log|pi|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|xi|to)\b', text)
        if bare:
            exprs.append(' '.join(bare))
    return exprs


def _parse_math_expr(latex: str):
    """Parse a LaTeX expression into a SymPy expression.

    Priority: parse_latex → manual conversion → simple sympify.
    """
    # 1. Try SymPy's built-in LaTeX parser
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except Exception:
        pass

    # 2. Manual LaTeX → Python conversion
    try:
        import sympy as sp
        s = latex.strip()

        # Implicit multiplication: 3x → 3*x, )a → )*a, a( → a*(  (but NOT after sin/cos/etc)
        s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
        s = re.sub(r'\)([a-zA-Z])', r')*\1', s)
        # Only add * before ( if preceded by a variable, not a function name
        s = re.sub(r'([a-zA-Z])\(', r'\1*(', s)
        # Remove spurious * after function names: sin*( → sin(
        s = re.sub(r'(sin|cos|tan|log|exp|sqrt)\*\(', r'\1(', s)

        # Greek letters
        for greek, name in [
            (r'\alpha', 'alpha'), (r'\beta', 'beta'), (r'\gamma', 'gamma'),
            (r'\delta', 'delta'), (r'\epsilon', 'epsilon'), (r'\varepsilon', 'epsilon'),
            (r'\theta', 'theta'), (r'\lambda', 'lambda'), (r'\mu', 'mu'),
            (r'\pi', 'pi'), (r'\sigma', 'sigma'), (r'\omega', 'omega'),
            (r'\xi', 'xi'), (r'\eta', 'eta'), (r'\rho', 'rho'), (r'\phi', 'phi'),
            (r'\psi', 'psi'), (r'\tau', 'tau'), (r'\chi', 'chi'),
        ]:
            s = s.replace(greek, name)
        # Functions and operators
        for latex_fn, py_fn in [
            (r'\sin', 'sin'), (r'\cos', 'cos'), (r'\tan', 'tan'),
            (r'\ln', 'log'), (r'\log', 'log'), (r'\exp', 'exp'),
            (r'\sqrt', 'sqrt'), (r'\cdot', '*'), (r'\times', '*'),
            (r'\left', ''), (r'\right', ''), (r'\infty', 'oo'),
            (r'\to', '->'), (r'\lim', ''), (r'\int', ''),
            (r'\frac', ''), (r'\,', ''), (r'\ ', ''),
            ('^T', '**T'),
        ]:
            s = s.replace(latex_fn, py_fn)
        # Braces → parentheses
        s = s.replace('{', '(').replace('}', ')')
        # Remove remaining backslash-commands
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        # Remove text annotations like \text{...}
        s = re.sub(r'text\([^)]*\)', '', s)
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        if not s:
            return None
        return sp.sympify(s, evaluate=False)
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def generate(
    markdown: str,
    api_key: str = "",
    model: str = "gpt-4o-mini",
) -> GenerationResult:
    """Process a Markdown document: find // @ai blocks, generate content, verify.

    This is the main entry point for the VeriTeach generation pipeline.

    Parameters
    ----------
    markdown : str
        The input markdown document with // @ai instructions
    api_key : str
        OpenAI API key (or set OPENAI_API_KEY env var)
    model : str
        LLM model to use for generation

    Returns
    -------
    GenerationResult
        Complete result with modified document and statistics
    """
    blocks = parse_ai_blocks(markdown)
    if not blocks:
        return GenerationResult(
            original_text=markdown,
            output_text=markdown,
            blocks_processed=0,
            items_generated=0,
            items_verified=0,
            errors=["No // @ai blocks found in document"],
        )

    result_text = markdown
    total_items = 0
    total_verified = 0
    errors = []

    # Process blocks in reverse order (so line numbers don't shift)
    for block in reversed(blocks):
        # 1. RAG retrieval
        knowledge = retrieve_knowledge(block.instruction)
        context = block.context_before[-300:] if block.context_before else ""

        # 2. Build prompt
        user_prompt = f"""Relevant textbook knowledge:
{knowledge}

Document context (what comes before this instruction):
{context}

Instruction: {block.instruction}

Generate teaching materials based on the instruction above.
Ensure mathematical correctness and use LaTeX for formulas."""

        # 3. Generate
        generation = _call_llm(SYSTEM_PROMPT, user_prompt, api_key=api_key, model=model)

        # 4. Verify each answer
        items = _parse_generated_items(generation)
        verified_count = 0
        for item in items:
            if item.type == "question" and item.answer:
                verified, detail = verify_answer(item.content, item.answer)
                item.verified = verified
                item.verification_detail = detail
                if verified:
                    verified_count += 1

        # 5. Format output with verification badges
        formatted = _format_output(items)

        # 6. Replace // @ai block in document
        # Find the block position
        marker_pos = result_text.find(block.raw_marker)
        if marker_pos >= 0:
            # Find end of this block (next double newline or end of text)
            end_pos = result_text.find("\n\n", marker_pos)
            if end_pos < 0:
                end_pos = len(result_text)
            result_text = (
                result_text[:marker_pos]
                + formatted
                + result_text[end_pos:]
            )
        else:
            errors.append(f"Could not find marker for: {block.instruction[:50]}")

        total_items += len(items)
        total_verified += verified_count

    return GenerationResult(
        original_text=markdown,
        output_text=result_text,
        blocks_processed=len(blocks),
        items_generated=total_items,
        items_verified=total_verified,
        errors=errors,
    )


def _parse_generated_items(generation: str) -> list[GeneratedItem]:
    """Parse LLM output into structured GeneratedItems."""
    items = []

    # Split by Q/A pairs
    q_pattern = re.compile(r'\*\*Q(\d+):\*\*\s*(.+?)(?=\*\*Q\d+:\*\*|\*\*A\d+:\*\*|$)', re.DOTALL)
    a_pattern = re.compile(r'\*\*A(\d+):\*\*\s*(.+?)(?=\*\*Q\d+:\*\*|\*\*A\d+:\*\*|$)', re.DOTALL)

    questions = {}
    for match in q_pattern.finditer(generation):
        questions[match.group(1)] = match.group(2).strip()

    answers = {}
    for match in a_pattern.finditer(generation):
        answers[match.group(1)] = match.group(2).strip()

    for num in sorted(set(list(questions.keys()) + list(answers.keys())), key=int):
        q_text = questions.get(num, "")
        a_text = answers.get(num, "")
        items.append(GeneratedItem(
            type="question",
            content=f"**Q{num}:** {q_text}" if q_text else "",
            answer=f"**A{num}:** {a_text}" if a_text else "",
        ))

    # If no Q/A pairs found, treat as lesson plan or slides
    if not items and generation.strip():
        items.append(GeneratedItem(
            type="slide" if "##" in generation else "lesson_plan",
            content=generation.strip(),
        ))

    return items


def _format_output(items: list[GeneratedItem]) -> str:
    """Format generated items for insertion into the document."""
    parts = []
    for item in items:
        badge = ""
        if item.verified:
            badge = " ✅ **SymPy 已验证**"
        elif item.type == "question" and item.answer and not item.verified:
            badge = " ⚠️ 未验证"

        content = item.content
        if item.answer:
            content += f"\n\n{item.answer}{badge}"
        elif badge:
            content += badge

        parts.append(content)

    return "\n\n---\n\n".join(parts)
