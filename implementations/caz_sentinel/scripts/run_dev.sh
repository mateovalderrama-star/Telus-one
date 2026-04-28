#!/usr/bin/env bash
set -euo pipefail
uv run --group caz-sentinel --env-file .env uvicorn caz_sentinel.api:app --reload --host 0.0.0.0 --port 8000
