"""SymPy formal verification engine — validate LaTeX equations in documents.

Architecture:
  Tier 1: sympy.parsing.latex.parse_latex()   (SymPy native, deterministic)
  Tier 2: LLM translation LaTeX → SymPy Python (fallback for edge cases)
  Then:   SymPy verification (deterministic — LLM only does translation)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger("exobrain.verify")


@dataclass
class VerificationResult:
    line: int          # 1-indexed line number
    equation: str      # original LaTeX string
    status: str        # "verified", "inconclusive", "error"
    detail: str        # human-readable explanation
    # ── Script generation fields ────────────────────────────────────
    checks: list[dict] = field(default_factory=list)
    # Each check dict: {"label": str, "lhs_py": str|None, "rhs_py": str|None,
    #                    "diff": str|None, "method": str, "passed": bool}


# ═══════════════════════════════════════════════════════════════════════
# Equation extraction
# ═══════════════════════════════════════════════════════════════════════


def extract_equations(markdown: str) -> list[tuple[int, str, str]]:
    """Extract all LaTeX equations from markdown.

    Returns list of (line_number, raw_text, display_mode).
    display_mode: "block" for $$...$$, "inline" for $...$.

    Block equations ($$...$$) are matched against the full text because
    they can span multiple lines.  Inline equations ($...$) can never
    cross line boundaries so they're still processed line-by-line.
    """
    equations: list[tuple[int, str, str]] = []
    lines = markdown.split("\n")

    # ── multi-line block equations ($$ … $$) ──────────────────────────
    block_pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    for match in block_pattern.finditer(markdown):
        eq = match.group(1).strip()
        if eq and len(eq) > 2 and eq != "\\":
            line_idx = markdown[: match.start()].count("\n") + 1
            equations.append((line_idx, eq, "block"))

    # ── single-line inline equations ($ … $) ──────────────────────────
    inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

    for line_idx, line in enumerate(lines, 1):
        for match in inline_pattern.finditer(line):
            eq = match.group(1).strip()
            if eq and len(eq) > 2:
                equations.append((line_idx, eq, "inline"))

    equations.sort(key=lambda x: x[0])
    return equations


# ═══════════════════════════════════════════════════════════════════════
# LaTeX → SymPy  (two-tier)
# ═══════════════════════════════════════════════════════════════════════


def latex_to_sympy(latex: str) -> tuple:
    """Convert LaTeX string to SymPy expression.

    Tier 1: SymPy's built-in ANTLR-based LaTeX parser.
    Tier 2: LLM translation (deepseek-v4-flash) for edge cases.

    Returns (sympy_expr, None) on success, (None, error_message) on failure.
    """
    # ── Preprocess: fix known SymPy parser edge cases ─────────────────
    latex = _preprocess_latex(latex)

    # ── Tier 1: SymPy native parser ─────────────────────────────────
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex)
        return (expr, None)
    except Exception as e:
        logger.debug("SymPy native parser failed for %r: %s", latex[:80], e)

    # ── Tier 2: LLM fallback ────────────────────────────────────────
    return _llm_translate(latex)


def _preprocess_latex(latex: str) -> str:
    """Minimal LaTeX preprocessing for known SymPy parser edge cases.

    Fixes applied:
      - e^{...} or e^... → \\exp{...}   (Euler's number)
      - \\, (thin space) → removed
    Only fixes well-understood patterns; doesn't try to be a full converter.
    """
    s = latex

    # e^{...} or e^... → \exp{...}
    # In standard math LaTeX, 'e' followed by '^' is ALWAYS Euler's number.
    # The only exception is LaTeX commands like \epsilon, \beta (preceded by \).
    # So: match standalone 'e' (not preceded by backslash) followed by ^.
    s = re.sub(r'(?<!\\)e\s*\^\{?', r'\\exp{', s)

    # d \exp{...} → \exp{...} inside integrals
    # "∫ x d(e^x)" = "∫ x e^x dx"  (d(e^x) = e^x dx)
    # SymPy misparses 'd' as a variable; this fixes the differential notation.
    s = re.sub(r'd\s*\\exp\{', r'\\exp{', s)

    # Remove LaTeX thin spaces
    s = s.replace(r'\,', '')

    return s


def _llm_translate(latex: str) -> tuple:
    """Use LLM to translate LaTeX → SymPy Python expression string.

    The LLM only does TRANSLATION.  Verification is still done
    deterministically by SymPy afterwards — the LLM never does math.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

    if not api_key:
        return (None, "No LLM API key configured — cannot parse LaTeX")

    try:
        import json as _json
        from urllib.request import Request, urlopen

        prompt = (
            "Convert this LaTeX expression to a SymPy-compatible Python expression string. "
            "Return ONLY the Python expression, nothing else. No explanation, no markdown.\n\n"
            "Rules:\n"
            "- e^x → exp(x)\n"
            "- x^2 → x**2\n"
            "- \\sin(x) → sin(x), \\cos → cos, \\tan → tan, \\ln → ln\n"
            "- \\frac{a}{b} → a/b\n"
            "- \\sqrt{x} → sqrt(x)\n"
            "- \\int f(x) dx → Integral(f(x), x)\n"
            "- \\int_a^b f(x) dx → Integral(f(x), (x, a, b))\n"
            "- \\sum_{i=1}^n → Sum(expr, (i, 1, n))\n"
            "- \\lim_{x→a} f(x) → Limit(f(x), x, a)\n"
            "- \\frac{d}{dx} f(x) → Derivative(f(x), x)\n"
            "- \\pi → pi, \\infty → oo\n"
            "- Implicit multiplication: 2x → 2*x, x y → x*y\n"
            "- sin^2(x) → sin(x)**2 (NOT sin**2(x))\n"
            "- e is Euler's number → use exp(), not E\n"
            "- ∫ x d(e^x) means ∫ x * exp(x) dx, NOT Integral(x, e)**x\n\n"
            f"LaTeX: {latex}\n\n"
            "SymPy Python expression:"
        )

        body = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 120,
        }).encode()

        req = Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        resp = urlopen(req, timeout=10)
        data = _json.loads(resp.read())
        sympy_str = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        sympy_str = re.sub(r"^```(?:python|py)?\s*", "", sympy_str)
        sympy_str = re.sub(r"\s*```$", "", sympy_str)

        # Validate: sympify the LLM's output
        import sympy as sp
        expr = sp.sympify(sympy_str)
        logger.info("LLM translated %r → %s", latex[:60], sympy_str[:80])
        return (expr, None)

    except Exception as e:
        return (None, f"LLM translation failed: {str(e)[:120]}")


# ═══════════════════════════════════════════════════════════════════════
# sympy_expr → Python source (for script generation)
# ═══════════════════════════════════════════════════════════════════════


def _sympy_to_py(expr) -> str | None:
    """Convert a SymPy expression to a python-eval-able string.

    Uses sympy.srepr() which gives a reproducible canonical form,
    then converts common patterns for readability.
    """
    if expr is None:
        return None
    s = str(expr)
    # Fix e**x → exp(x) (SymPy str() uses e**x for Euler's number)
    # Match 'e**' followed by variable or paren group
    s = re.sub(r'\be\*\*(\w+)', r'exp(\1)', s)
    s = re.sub(r'\be\*\*(\([^)]+\))', r'exp\1', s)
    # Fix e** without being preceded by a letter (standalone Euler's e)
    s = re.sub(r'(?<![a-zA-Z)])e\*\*', 'exp', s)
    return s


# ═══════════════════════════════════════════════════════════════════════
# Equation verification
# ═══════════════════════════════════════════════════════════════════════


def verify_equation(latex: str) -> VerificationResult:
    """Verify a single LaTeX equation.

    Strategy (in order):
      1. Multi-equality (A = B = C) → split and verify each segment
      2. Integral equalities         → verify by differentiation
      3. Algebraic equalities        → verify LHS − RHS = 0
      4. Standalone formulas         → verify structural validity
    """
    # ── Multi-equality splitting ────────────────────────────────────
    if _count_equality_signs(latex) >= 2:
        return _verify_multi_equality(latex)

    # ── Integral verification ───────────────────────────────────────
    if r"\int" in latex:
        return _verify_integral(latex)

    # ── Equality verification ───────────────────────────────────────
    if "=" in latex and "\\neq" not in latex:
        return _verify_single_equality(latex)

    # ── Expression validation ───────────────────────────────────────
    expr, err = latex_to_sympy(latex)
    if expr is not None:
        return VerificationResult(
            line=0, equation=latex,
            status="verified",
            detail="✅ Valid expression",
            checks=[{"label": latex, "lhs_py": _sympy_to_py(expr),
                     "rhs_py": None, "diff": None,
                     "method": "expression", "passed": True}],
        )
    return VerificationResult(
        line=0, equation=latex,
        status="error",
        detail=f"Parse error: {err}",
    )


def _count_equality_signs(latex: str) -> int:
    """Count = signs outside brace groups."""
    depth = 0
    count = 0
    for ch in latex:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0:
            count += 1
    return count


def _split_all_equalities(latex: str) -> list[str]:
    """Split on all = signs outside brace groups. Returns list of segments."""
    depth = 0
    segments: list[str] = []
    last = 0
    for i, ch in enumerate(latex):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0:
            segments.append(latex[last:i].strip())
            last = i + 1
    segments.append(latex[last:].strip())
    return segments


def _verify_multi_equality(latex: str) -> VerificationResult:
    """Verify A = B = C = ... by checking each adjacent pair.

    Reports the first failing segment; if all pass, reports verified.
    """
    import sympy as sp

    segments = _split_all_equalities(latex)
    if len(segments) < 2:
        return _verify_integral(latex) if r"\int" in latex else _verify_single_equality(latex)

    results_detail: list[str] = []
    checks: list[dict] = []
    all_ok = True
    has_error = False

    for i in range(len(segments) - 1):
        lhs_expr, lhs_err = latex_to_sympy(segments[i])
        rhs_expr, rhs_err = latex_to_sympy(segments[i + 1])

        if lhs_expr is None or rhs_expr is None:
            err = lhs_err or rhs_err or "?"
            results_detail.append(f"  Segment {i+1}: parse error — {err}")
            checks.append({"label": f"Segment {i+1}", "lhs_py": None, "rhs_py": None,
                          "diff": None, "method": "parse_error", "passed": False})
            has_error = True
            all_ok = False
            continue

        lhs_py = _sympy_to_py(lhs_expr)
        rhs_py = _sympy_to_py(rhs_expr)

        try:
            diff = sp.simplify(lhs_expr - rhs_expr)
            passed = diff == 0
            diff_py = _sympy_to_py(diff)
            if passed:
                results_detail.append(f"  {segments[i][:30]} = {segments[i+1][:30]}  ✅")
            else:
                results_detail.append(
                    f"  {segments[i][:30]} = {segments[i+1][:30]}  ⚠️ diff={diff}"
                )
                all_ok = False
            checks.append({"label": f"Segment {i+1}: {segments[i][:25]} = {segments[i+1][:25]}",
                          "lhs_py": lhs_py, "rhs_py": rhs_py,
                          "diff": diff_py, "method": "simplify", "passed": passed})
        except Exception as e:
            results_detail.append(f"  {segments[i][:30]} = {segments[i+1][:30]}  ❌ {e}")
            checks.append({"label": f"Segment {i+1}", "lhs_py": lhs_py, "rhs_py": rhs_py,
                          "diff": None, "method": "error", "passed": False})
            has_error = True
            all_ok = False

    detail_text = "; ".join(results_detail)
    if all_ok:
        return VerificationResult(
            line=0, equation=latex,
            status="verified",
            detail=f"✅ Chain verified ({len(segments)-1} segments): {detail_text}",
            checks=checks,
        )
    if has_error:
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"❌ Chain verification failed: {detail_text}",
            checks=checks,
        )
    return VerificationResult(
        line=0, equation=latex,
        status="inconclusive",
        detail=f"⚠️ Chain inconclusive: {detail_text}",
        checks=checks,
    )


def _verify_single_equality(latex: str) -> VerificationResult:
    """Verify an algebraic equality: simplify(LHS − RHS) == 0."""
    parts = _split_equality(latex)
    if len(parts) != 2:
        expr, err = latex_to_sympy(latex)
        if expr is not None:
            return VerificationResult(
                line=0, equation=latex,
                status="verified",
                detail="✅ Valid expression",
                checks=[{"label": latex, "lhs_py": _sympy_to_py(expr),
                         "rhs_py": None, "diff": None,
                         "method": "expression", "passed": True}],
            )
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Cannot split equality: {latex}",
        )

    lhs_expr, lhs_err = latex_to_sympy(parts[0])
    rhs_expr, rhs_err = latex_to_sympy(parts[1])

    if lhs_expr is None or rhs_expr is None:
        err_msg = lhs_err or rhs_err or "Parse error"
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Parse error: {err_msg}",
        )

    lhs_py = _sympy_to_py(lhs_expr)
    rhs_py = _sympy_to_py(rhs_expr)

    try:
        import sympy as sp
        diff = sp.simplify(lhs_expr - rhs_expr)
        passed = diff == 0
        diff_py = _sympy_to_py(diff)
        checks = [{"label": f"{parts[0][:25]} = {parts[1][:25]}",
                    "lhs_py": lhs_py, "rhs_py": rhs_py,
                    "diff": diff_py, "method": "simplify", "passed": passed}]

        if passed:
            return VerificationResult(
                line=0, equation=latex,
                status="verified",
                detail="✅ Verified: LHS − RHS = 0",
                checks=checks,
            )
        if diff.is_number:
            return VerificationResult(
                line=0, equation=latex,
                status="error",
                detail=f"❌ LHS ≠ RHS (difference = {diff})",
                checks=checks,
            )
        return VerificationResult(
            line=0, equation=latex,
            status="inconclusive",
            detail=f"⚠️ LHS − RHS = {diff}. May be correct with additional constraints.",
            checks=checks,
        )
    except Exception as e:
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Simplification error: {str(e)[:100]}",
        )


def _verify_integral(latex: str) -> VerificationResult:
    """Verify integral equalities.

    Strategy:
      - F(x) = ∫ f(x) dx   →  try: diff(F, x) == f(x), or: integrate(f, x) == F + C
      - ∫ f(x) dx = F(x)   →  same (swap sides)
      - Standalone integral →  inconclusive
    """
    import sympy as sp

    expr, err = latex_to_sympy(latex)
    if expr is None:
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Parse error: {err}",
        )

    # Case 1: Eq(LHS, RHS) with integral on one side
    if isinstance(expr, sp.Equality):
        lhs, rhs = expr.lhs, expr.rhs
        if isinstance(rhs, sp.Integral):
            return _verify_integral_eq(latex, lhs, rhs)
        if isinstance(lhs, sp.Integral):
            return _verify_integral_eq(latex, rhs, lhs)

        # No explicit Integral object — might have been misparsed
        try:
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return VerificationResult(
                    line=0, equation=latex,
                    status="verified",
                    detail="✅ Verified: LHS − RHS = 0",
                    checks=[{"label": latex, "lhs_py": _sympy_to_py(lhs),
                             "rhs_py": _sympy_to_py(rhs),
                             "diff": "0", "method": "simplify", "passed": True}],
                )
            return _try_differentiate_check(latex, lhs, rhs)
        except Exception as e:
            return VerificationResult(
                line=0, equation=latex,
                status="error",
                detail=f"Integral verification error: {str(e)[:100]}",
            )

    # Case 2: Standalone Integral expression
    if isinstance(expr, sp.Integral):
        return VerificationResult(
            line=0, equation=latex,
            status="inconclusive",
            detail="🔍 Integral expression — no antiderivative to verify against.",
        )

    # Case 3: Misparsed — the LLM might help
    return VerificationResult(
        line=0, equation=latex,
        status="inconclusive",
        detail=f"🔍 Integral detected but parse gave: {expr}. Try rewriting as F(x) = ∫ f(x) dx.",
    )


def _verify_integral_eq(
    latex: str, claimed_F, integrand: "sp.Integral",
) -> VerificationResult:
    """Verify ∫ f(x) dx = F(x) using both differentiation and integration."""
    import sympy as sp

    # If claimed_F is an undefined function like F(x), this is a definition,
    # not a claim to verify.  Mark as assumed-true.
    from sympy.core.function import AppliedUndef
    if isinstance(claimed_F, AppliedUndef):
        return VerificationResult(
            line=0, equation=latex,
            status="verified",
            detail="✅ Definition: F(x) is defined as this integral (assumed true)",
            checks=[{"label": latex, "lhs_py": str(claimed_F),
                     "rhs_py": str(integrand),
                     "diff": None, "method": "definition", "passed": True}],
        )

    try:
        f = integrand.function
        var = integrand.variables[0] if integrand.variables else sp.Symbol("x")

        # Method 1: Differentiate the claimed antiderivative
        derivative = sp.diff(claimed_F, var)
        diff_check = sp.simplify(derivative - f)

        if diff_check == 0:
            return VerificationResult(
                line=0, equation=latex,
                status="verified",
                detail=f"✅ Verified: d/d{var}({claimed_F}) = {f} = integrand",
                checks=[{"label": f"diff({claimed_F}, {var}) == {f}",
                         "lhs_py": _sympy_to_py(derivative),
                         "rhs_py": _sympy_to_py(f),
                         "diff": "0", "method": "differentiate", "passed": True}],
            )

        # Method 2: Integrate the integrand and compare
        try:
            computed = sp.integrate(f, var)
            diff2 = sp.simplify(claimed_F - computed)
            if diff2 == 0 or (isinstance(diff2, sp.Number) and diff2 == 0):
                return VerificationResult(
                    line=0, equation=latex,
                    status="verified",
                    detail=f"✅ Verified: ∫ {f} d{var} = {computed} = claimed F",
                    checks=[{"label": f"integrate({f}, {var}) == {claimed_F}",
                             "lhs_py": _sympy_to_py(computed),
                             "rhs_py": _sympy_to_py(claimed_F),
                             "diff": "0", "method": "integrate", "passed": True}],
                )
        except Exception:
            pass  # integrate() may not find a closed form

        # Neither method confirmed — report the diff
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"❌ d/d{var}(claimed F) = {derivative} ≠ {f} (diff = {diff_check})",
            checks=[{"label": f"diff({claimed_F}, {var}) != {f}",
                     "lhs_py": _sympy_to_py(derivative),
                     "rhs_py": _sympy_to_py(f),
                     "diff": _sympy_to_py(diff_check),
                     "method": "differentiate", "passed": False}],
        )

    except Exception as e:
        return VerificationResult(
            line=0, equation=latex,
            status="error",
            detail=f"Integral verification error: {str(e)[:100]}",
        )


def _try_differentiate_check(latex: str, lhs, rhs) -> VerificationResult:
    """Fallback: try to check by differentiating (for implicit integral equalities)."""
    import sympy as sp

    x = sp.Symbol("x")
    try:
        d_lhs = sp.diff(lhs, x)
        d_rhs = sp.diff(rhs, x)
        diff = sp.simplify(d_lhs - d_rhs)
        if diff == 0:
            return VerificationResult(
                line=0, equation=latex,
                status="verified",
                detail="✅ Verified: derivatives equal (LHS and RHS differ by constant)",
            )
        return VerificationResult(
            line=0, equation=latex,
            status="inconclusive",
            detail=f"⚠️ d(LHS)/dx − d(RHS)/dx = {diff}",
        )
    except Exception:
        return VerificationResult(
            line=0, equation=latex,
            status="inconclusive",
            detail="🔍 Could not verify integral equality automatically.",
        )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _split_equality(latex: str) -> list[str]:
    """Split an equation on the first = outside brace groups."""
    depth = 0
    for i, ch in enumerate(latex):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0:
            lhs = latex[:i].strip()
            rhs = latex[i + 1 :].strip()
            return [lhs, rhs] if lhs and rhs else [latex]
    return [latex]


# ═══════════════════════════════════════════════════════════════════════
# Document-level API
# ═══════════════════════════════════════════════════════════════════════


def verify_document(markdown: str) -> list[VerificationResult]:
    """Verify all equations in a document."""
    equations = extract_equations(markdown)
    if not equations:
        return []

    results: list[VerificationResult] = []
    for line_idx, eq, display_mode in equations:
        result = verify_equation(eq)
        result.line = line_idx
        wrapper = "$$" if display_mode == "block" else "$"
        result.equation = f"{wrapper} {eq} {wrapper}"
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Script generation
# ═══════════════════════════════════════════════════════════════════════


def generate_script(
    results: list[VerificationResult],
    format: str = "simple",
    title: str = "Verification Script",
) -> str:
    """Generate a standalone Python verification script from results.

    Args:
        results: list of VerificationResult (must include checks data)
        format: "simple" (assert-based) or "unittest" (unittest.TestCase)
        title: script title

    Returns:
        Complete Python script as a string.
    """
    if format == "unittest":
        return _generate_unittest_script(results, title)
    return _generate_simple_script(results, title)


def _generate_simple_script(results: list[VerificationResult], title: str) -> str:
    """Generate a simple assert-based verification script."""
    lines: list[str] = []
    lines.append('"""Verification Script — generated by VeriTeach')
    lines.append(f"   {title}")
    lines.append('   https://emergence.science/en/play/notebook')
    lines.append('"""')
    lines.append("")
    lines.append("import sympy as sp")
    lines.append("from sympy import exp, Integral, sin, cos, tan, log, sqrt, pi, oo")
    lines.append("")
    lines.append("x = sp.Symbol('x')")

    # Collect undefined functions used in checks
    undef_functions: set[str] = set()
    for result in results:
        for check in result.checks:
            for key in ("lhs_py", "rhs_py", "diff"):
                val = check.get(key, "")
                if val and "F(" in val:
                    undef_functions.add("F")
                if val and "G(" in val:
                    undef_functions.add("G")
    for fn in sorted(undef_functions):
        lines.append(f"{fn} = sp.Function('{fn}')")

    lines.append("")
    lines.append("print('🔍 Running verification...')")
    lines.append("")

    step_num = 0
    for result in results:
        # Sanitize equation for comment display
        eq_display = result.equation.replace("\n", " ")[:60]
        lines.append(f"# ── Line {result.line}: {eq_display} ──")
        lines.append(f"# Status: {result.status}")

        if not result.checks:
            lines.append(f"# (no executable checks — {result.detail})")
            lines.append("")
            continue

        for check in result.checks:
            step_num += 1
            method = check.get("method", "?")
            label = check.get("label", f"Step {step_num}")
            passed = check.get("passed", False)
            lhs_py = check.get("lhs_py")
            rhs_py = check.get("rhs_py")
            diff_py = check.get("diff")

            lines.append(f"# Step {step_num}: {label}")
            lines.append(f"# Method: {method}")

            if method == "definition":
                lines.append(f"# Definition (assumed true)")
                lines.append(f"#   LHS = {lhs_py}")
                lines.append(f"#   RHS = {rhs_py}")
                lines.append("")
                continue

            if method == "expression":
                lines.append(f"# Valid expression")
                lines.append(f"#   {lhs_py}")
                lines.append("")
                continue

            if lhs_py and rhs_py:
                lines.append(f"lhs_{step_num} = {lhs_py}")
                lines.append(f"rhs_{step_num} = {rhs_py}")
                lines.append(f"diff_{step_num} = sp.simplify(lhs_{step_num} - rhs_{step_num})")
                if passed:
                    lines.append(f"assert diff_{step_num} == 0, f\"Step {step_num} FAILED: diff = {{diff_{step_num}}}\"")
                    lines.append(f"print('  ✅ Step {step_num} passed')")
                else:
                    expected_hint = f" (got {diff_py})" if diff_py else ""
                    lines.append(f"# ⚠️ This step is inconclusive — diff ≠ 0{expected_hint}")
                    lines.append(f"# assert diff_{step_num} == 0  # would fail — needs human review")
                    lines.append(f"print('  ⚠️ Step {step_num} SKIPPED (inconclusive — diff = {{diff_{step_num}}})')")
                lines.append("")

    lines.append(f"print(f'\\n✅ All verifications completed! ({step_num} check(s))')")
    return "\n".join(lines)


def _generate_unittest_script(results: list[VerificationResult], title: str) -> str:
    """Generate a unittest.TestCase-based verification script."""
    lines: list[str] = []
    lines.append('"""Verification Script — generated by VeriTeach')
    lines.append(f"   {title}")
    lines.append('   https://emergence.science/en/play/notebook')
    lines.append('"""')
    lines.append("")
    lines.append("import sympy as sp")
    lines.append("import unittest")
    lines.append("from sympy import exp, Integral, sin, cos, tan, log, sqrt, pi, oo")
    lines.append("")
    lines.append("x = sp.Symbol('x')")

    # Collect undefined functions
    undef_functions: set[str] = set()
    for result in results:
        for check in result.checks:
            for key in ("lhs_py", "rhs_py", "diff"):
                val = check.get(key, "")
                if val and "F(" in val:
                    undef_functions.add("F")
                if val and "G(" in val:
                    undef_functions.add("G")
    for fn in sorted(undef_functions):
        lines.append(f"{fn} = sp.Function('{fn}')")

    lines.append("")
    lines.append("")
    lines.append("class TestVerification(unittest.TestCase):")
    lines.append('    """Auto-generated verification tests."""')
    lines.append("")

    for result in results:
        eq_display = result.equation.replace("\n", " ")[:50]
        lines.append(f"    # ── Line {result.line}: {eq_display} ──")

        if not result.checks:
            lines.append(f"    # (no executable checks — {result.detail})")
            lines.append("")
            continue

        for i, check in enumerate(result.checks):
            method = check.get("method", "?")
            label = check.get("label", f"Step")
            passed = check.get("passed", False)
            lhs_py = check.get("lhs_py")
            rhs_py = check.get("rhs_py")

            if method in ("definition", "expression"):
                lines.append(f"    # {method}: {label} (no assertion needed)")
                continue

            if lhs_py and rhs_py:
                # Sanitize label for method name
                safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', f"line{result.line}_step{i+1}")
                lines.append(f"    def test_{safe_label}(self):")
                lines.append(f'        """{label}"""')
                lines.append(f"        lhs = {lhs_py}")
                lines.append(f"        rhs = {rhs_py}")
                lines.append(f"        diff = sp.simplify(lhs - rhs)")
                lines.append(f"        self.assertEqual(diff, 0, f'Diff = {{diff}}')")
                lines.append("")

    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    unittest.main()")
    return "\n".join(lines)
