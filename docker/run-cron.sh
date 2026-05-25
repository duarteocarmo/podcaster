#!/bin/bash
set -euo pipefail

while IFS= read -r -d '' env_var; do
    export "$env_var"
done < /proc/1/environ

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
cd /app
exec podcaster
