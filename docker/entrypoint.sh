#!/bin/bash
# Dump container env to a file cron can source
printenv | grep -E '^(OPENAI_API_KEY|REBUILD_TRIGGER_URL|MODAL_IMAGE_BUILDER_VERSION|VLLM_WORKER_MULTIPROC_METHOD|HF_XET_HIGH_PERFORMANCE)' > /etc/cron.env

cron && tail -f /var/log/cron.log
