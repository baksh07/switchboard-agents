FROM python:3.11-slim

# uv, copied from its official image (pinned, like everything else in a build)
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# 1. Dependencies only. This layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. The project itself. Code changes rebuild only from here.
COPY README.md ./
COPY src ./src
COPY mcp_server ./mcp_server
RUN uv sync --frozen --no-dev

# Don't run as root; /data is where the databases live (a volume in Compose).
RUN useradd --create-home app && mkdir /data && chown app /data
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CRM_DATABASE_URL=sqlite:////data/crm.db \
    CHECKPOINT_DB=/data/checkpoints.db

EXPOSE 8000
CMD ["uvicorn", "switchboard.usecases.lead_qualification.api:app", "--host", "0.0.0.0", "--port", "8000"]