"""VeriTeach API server — FastAPI backend for the educator AI co-worker."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from notebook.teach.generator import generate, parse_ai_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veriteach.server")

app = FastAPI(title="VeriTeach", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════


class GenerateRequest(BaseModel):
    markdown: str
    api_key: str = ""


class GenerateResponse(BaseModel):
    output: str
    blocks_processed: int
    items_generated: int
    items_verified: int
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    sympy_available: bool = False


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        import sympy  # noqa: F401
        sympy_ok = True
    except ImportError:
        sympy_ok = False
    return HealthResponse(sympy_available=sympy_ok)


@app.post("/generate", response_model=GenerateResponse)
async def generate_content(req: GenerateRequest):
    """Process markdown with // @ai blocks and return generated content."""
    if not req.markdown.strip():
        raise HTTPException(400, "No content provided")

    result = generate(
        markdown=req.markdown,
        api_key=req.api_key or os.getenv("OPENAI_API_KEY", ""),
    )
    return GenerateResponse(
        output=result.output_text,
        blocks_processed=result.blocks_processed,
        items_generated=result.items_generated,
        items_verified=result.items_verified,
        errors=result.errors,
    )


@app.post("/parse")
async def parse_document(req: GenerateRequest):
    """Parse // @ai blocks from a document (preview mode)."""
    blocks = parse_ai_blocks(req.markdown)
    return {
        "blocks_found": len(blocks),
        "instructions": [
            {"line": b.line_number, "instruction": b.instruction}
            for b in blocks
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def editor():
    """Serve the VeriTeach editor UI."""
    html_path = os.path.join(os.path.dirname(__file__), "editor.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>VeriTeach Editor</h1><p>editor.html not found</p>")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
