FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# NOTE: Do NOT set ENV PORT — Railway injects $PORT dynamically.
# Shell-form CMD expands $PORT at runtime.
CMD sh -c "python3 -m uvicorn notebook.server.api:app --host 0.0.0.0 --port ${PORT:-8080}"
