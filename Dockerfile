# Dockerfile using uv for podcaster
ARG UV_VERSION=0.9.2
ARG PYTHON_VERSION=3.11
ARG BASE_LAYER=bookworm

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-${BASE_LAYER}

WORKDIR /app

# Copy dependency files and README (needed for package metadata)
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Install dependencies
RUN uv sync --locked --no-dev

# Copy source code
COPY src ./src

# Set the environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["/bin/bash"]
