# Dockerfile using uv for podcaster
ARG UV_VERSION=0.9.2
ARG PYTHON_VERSION=3.11
ARG BASE_LAYER=bookworm

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-${BASE_LAYER}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y cron ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev \
    && mkdir -p transcripts \
    && touch /var/log/cron.log

ENV PATH="/app/.venv/bin:$PATH"

ADD docker/entrypoint.sh /entrypoint.sh
ADD docker/run-cron.sh /usr/local/bin/run-podcaster-cron
ADD docker/crontab /etc/cron.d/podcaster-cron
RUN chmod 0644 /etc/cron.d/podcaster-cron && chmod +x /entrypoint.sh /usr/local/bin/run-podcaster-cron

ENTRYPOINT ["/entrypoint.sh"]
