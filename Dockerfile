# Dockerfile using uv for podcaster
ARG UV_VERSION=0.9.2
ARG PYTHON_VERSION=3.11
ARG BASE_LAYER=bookworm

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-${BASE_LAYER}

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --locked --no-dev

# Copy source code
COPY src ./src
COPY README.md .

# Set the environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["/bin/bash"]
