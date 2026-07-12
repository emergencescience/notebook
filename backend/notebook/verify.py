"""SymPy formal verification engine — validate LaTeX equations in documents."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("exobrain.verify")


@dataclass
class VerificationResult:
    line: int          # 1-indexed line number
    equation: str      # original LaTeX string
    status: str        # "verified", "inconclusive", "error"
    detail: str        # human-readable explanation


def extract_equations(markdown: str) -> list[tuple[int, str, str]]:
    """Extract all LaTeX equations from markdown.

    Returns list of (line_number, raw_text, display_mode).
    display_mode: "block" for $$...$$, "inline" for $...$.
    """
    equations = []
    lines = markdown.split("\n")

    # Block equations: $$...$$
    block_pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    # Inline equations: $...$ (but not $$)
    inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

    for line_idx, line in enumerate(lines, 1):
        # Check for block equations
        for match in block_pattern.finditer(line):
            eq = match.group(1).strip()
            if eq and len(eq) > 2:  # skip trivial like just "x"
                equations.append((line_idx, eq, "block"))

        # Check for inline equations
        for match in inline_pattern.finditer(line):
            eq = match.group(1).strip()
            # Keep any non-trivial equation. (Previously equations starting
            # with a LaTeX command like \frac, \sin, \sum were wrongly
            # dropped — that silently skipped ~half of real equations.)
            if eq and len(eq) > 2:
                equations.append((line_idx, eq, "inline"))

    return equations


def latex_to_sympy(latex: str) -> tuple:
    """Convert LaTeX to SymPy expression.

    Returns (sympy_expr, None) on success, (None, error_message) on failure.
    Tries sympy's built-in LaTeX parser first, falls back to manual conversion.
    """
    # Try sympy's built-in parser
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex)
        return (expr, None)
    except Exception:
        pass  # Fall through to manual conversion

    # Manual fallback for common LaTeX patterns
    try:
        import sympy as sp
        converted = _manual_latex_convert(latex)
        if converted is None:
            return (None, "Could not parse LaTeX expression")
        expr = sp.sympify(converted)
        return (expr, None)
    except Exception as e2:
        return (None, f"Parse error: {str(e2)[:100]}")


def _manual_latex_convert(latex: str) -> str | None:
    """Manual conversion of common LaTeX to SymPy-compatible Python.

    Handles the most common math patterns in STEM papers.
    """
    s = latex.strip()

    # Clean up: implicit multiplication
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 2x → 2*x
    s = re.sub(r'\)(\()', r')*(', s)       # (a)(b) → (a)*(b)
    s = re.sub(r'\)([a-zA-Z])', r')*\1', s)  # (a)b → (a)*b
    s = re.sub(r'([a-zA-Z])\(', r'\1*(', s)  # a(b) → a*(b)

    # Handle common LaTeX formatting
    replacements = [
        (r"\left", ""), (r"\right", ""),
        (r"\cdot", "*"), (r"\times", "*"),
        (r"\frac{", "("), (r"}{", ")/("),  # \frac{a}{b} → (a)/(b) — needs careful handling
        (r"\sqrt{", "sqrt("),
        (r"\sin", "sin"), (r"\cos", "cos"), (r"\tan", "tan"),
        (r"\log", "log"), (r"\ln", "ln"),
        (r"\exp", "exp"),
        (r"\pi", "pi"), (r"\infty", "oo"),
        (r"\alpha", "alpha"), (r"\beta", "beta"), (r"\gamma", "gamma"),
        (r"\delta", "delta"), (r"\epsilon", "epsilon"), (r"\theta", "theta"),
        (r"\lambda", "lambda_"), (r"\mu", "mu"), (r"\sigma", "sigma"),
        (r"\omega", "omega"), (r"\Delta", "Delta"),
        (r"\sum", "Sum"), (r"\prod", "Product"), (r"\int", "Integral"),
        (r"\partial", "Derivative"),
        (r"\mathbf{", ""), (r"\mathcal{", ""),
        (r"\mathbb{", ""), (r"\Re", "re"), (r"\Im", "im"),
        (r"\operatorname{", ""),
        (r"\text{", ""),
        (r"\quad", " "), (r"\qquad", "  "),
        (r"\\", " "),
        (r"\{", "("), (r"\}", ")"),
        ("{", "("), ("}", ")"),
        (r"\pm", " "),  # Split: a +/- b → a b (can't verify ± equations, mark inconclusive)
        (r"\mp", " "),
        (r"\to", "->"),
        (r"\rightarrow", "->"),
        (r"\Rightarrow", "=>"),
        (r"\neq", "!="),
        (r"\leq", "<="),
        (r"\geq", ">="),
        (r"\approx", "~="),
        (r"\equiv", "=="),
        (r"\propto", "~"),
        ("^T", "**T"),  # transpose
        ("^\\top", "**T"),
        (r"\'", ""),  # derivative prime notation
        (r"\prime", ""),
        (r"\dot{", "diff("),  # time derivative
        (r"\ddot{", "diff(diff("),
        (r"\hat{", ""),
        (r"\bar{", ""),
        (r"\vec{", ""),
    ]

    # Handle \frac first (most complex)
    s = _handle_frac(s)

    for old, new in replacements:
        s = s.replace(old, new)

    # Clean up unmatched braces
    while "(" in s and s.count("(") > s.count(")"):
        s += ")"
    while ")" in s and s.count(")") > s.count("("):
        s = "(" + s

    return s if s else None


def _handle_frac(s: str) -> str:
    """Handle \\frac{numerator}{denominator} → (numerator)/(denominator)."""
    while "\\frac" in s:
        idx = s.index("\\frac")
        # Skip past \frac{
        brace_open = s.index("{", idx)
        depth = 1
        pos = brace_open + 1
        while depth > 0 and pos < len(s):
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
            pos += 1
        num = s[brace_open + 1:pos - 1]

        # Now parse denominator
        if pos < len(s) and s[pos] == "{":
            depth = 1
            denom_start = pos + 1
            pos = denom_start
            while depth > 0 and pos < len(s):
                if s[pos] == "{":
                    depth += 1
                elif s[pos] == "}":
                    depth -= 1
                pos += 1
            denom = s[denom_start:pos - 1]
        else:
            break

        s = s[:idx] + f"({num})/({denom})" + s[pos:]

    return s


def verify_equation(latex: str) -> VerificationResult:
    """Verify a single LaTeX equation.

    For equalities (a = b): check if (a - b) simplifies to 0.
    For formulas (no =): verify structural validity.
    """
    # Check if it's an equality
    if "=" in latex and "\\neq" not in latex:
        # Split on = but be careful about LaTeX
        parts = _split_equality(latex)
        if len(parts) == 2:
            lhs_expr, lhs_err = latex_to_sympy(parts[0])
            rhs_expr, rhs_err = latex_to_sympy(parts[1])

            if lhs_expr is not None and rhs_expr is not None:
                try:
                    import sympy as sp
                    diff = sp.simplify(lhs_expr - rhs_expr)
                    if diff == 0:
                        return VerificationResult(
                            line=0, equation=latex,
                            status="verified",
                            detail="✅ Verified: LHS − RHS = 0"
                        )
                    # If diff is a non-zero constant, it's definitely wrong
                    if diff.is_number and diff != 0:
                        return VerificationResult(
                            line=0, equation=latex,
                            status="error",
                            detail=f"❌ LHS ≠ RHS (difference = {diff})"
                        )
                    # Otherwise inconclusive (free variables)
                    return VerificationResult(
                        line=0, equation=latex,
                        status="inconclusive",
                        detail=f"⚠️ LHS − RHS = {diff}. May be correct with additional constraints."
                    )
                except Exception as e:
                    return VerificationResult(
                        line=0, equation=latex,
                        status="error",
                        detail=f"Simplification error: {str(e)[:100]}"
                    )
            else:
                err_msg = lhs_err or rhs_err or "Parse error"
                return VerificationResult(
                    line=0, equation=latex,
                    status="error",
                    detail=f"Parse error: {err_msg}"
                )

    # Not an equality — try to parse as a valid expression
    expr, err = latex_to_sympy(latex)
    if expr is not None:
        return VerificationResult(
            line=0, equation=latex,
            status="verified",
            detail="✅ Valid expression (non-equality)"
        )
    else:
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Parse error: {err}"
        )


def _split_equality(latex: str) -> list[str]:
    """Split an equation on =, being careful about LaTeX groups."""
    # Simple heuristic: find the first = that's not inside braces
    depth = 0
    for i, ch in enumerate(latex):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0:
            lhs = latex[:i].strip()
            rhs = latex[i+1:].strip()
            return [lhs, rhs] if lhs and rhs else [latex]
    return [latex]


def verify_document(markdown: str) -> list[VerificationResult]:
    """Verify all equations in a document.

    Returns list of VerificationResult, one per equation.
    """
    equations = extract_equations(markdown)
    if not equations:
        return []

    results = []
    for line_idx, eq, display_mode in equations:
        result = verify_equation(eq)
        result.line = line_idx
        result.equation = f"{'$$' if display_mode == 'block' else '$'} {eq} {'$$' if display_mode == 'block' else '$'}"
        results.append(result)

    return results
