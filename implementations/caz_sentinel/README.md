# CAZ Sentinel

Runtime prompt guard for open-weight LLMs using CAZ (Concept Allocation Zones) probes. Wraps any HuggingFace causal model behind OpenAI-compatible endpoints, scoring each incoming prompt against a library of CAZ direction probes and suppressing requests that exceed concept thresholds.

## Overview

CAZ Sentinel performs a single pre-generation forward pass per prompt with `output_hidden_states=True`. For each registered probe, it computes a cosine similarity score between the last-token residual stream at a specific layer and a pre-fit unit direction vector. If any probe score exceeds its threshold, the request is suppressed before generation begins; otherwise the KV cache from the scoring pass is reused in `model.generate()` with no duplicate compute.

## Architecture

```
POST /v1/chat/completions
        │
        ▼
tokenize(messages)
        │
        ▼
forward(input_ids, output_hidden_states=True, use_cache=True)
        │  returns hidden_states[L] for each CAZ layer + past_key_values
        ▼
for c in probes: score_c = cosine(hidden_states[L_c][-1], d_c) → [0, 1]
        │
        ▼
any(score_c >= threshold_c) ?
        │
   ┌────┴────┐
   │ false   │ true
   ▼         ▼
generate() SUPPRESSED → refusal message + Chronicle UDM event
   │         │
   ▼         ▼
ChatCompletion (finish_reason=stop | content_filter)
```

Streaming requests that pass the guard do not benefit from KV cache reuse (limitation of `TextIteratorStreamer`); they are re-scored and re-tokenized.

## Quickstart

```bash
# Install dependencies
uv sync --group caz-sentinel

# Set environment variables
export CAZ_SENTINEL_PROBE_DIR=implementations/caz_sentinel/tests/fixtures/synthetic_probes
export CAZ_SENTINEL_MODEL_ID=EleutherAI/pythia-70m
export CAZ_SENTINEL_DEVICE=cuda

# Build a synthetic 9-probe fixture (uses pythia-70m — no GPU required)
uv run python implementations/caz_sentinel/scripts/build_synthetic_probes.py --out $CAZ_SENTINEL_PROBE_DIR

# Start the service
uv run uvicorn caz_sentinel.api:build_app --factory --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Liveness check. Returns model id, concept names, d_model. |
| POST | `/v1/chat/completions` | OpenAI-compatible chat endpoint. Supports `stream=true`. |
| POST | `/v1/audit` | Scoring-only (no generation). Returns full `AuditResult`. |
| GET | `/v1/audit` | List recent audit records (default limit 100). |
| GET | `/v1/audit/{request_id}` | Retrieve a specific audit record by id. |

### Request / Response shape

`POST /v1/chat/completions` matches the OpenAI v1 spec: `{model, messages, temperature, top_p, max_tokens, stream}`. Returns a `chat.completion` response. Response headers added: `x-sentinel-request-id`, `x-sentinel-decision` (`pass` | `suppressed` | `bypass`).

### Bypass mode

Set request header `x-sentinel-bypass: 1` to skip the guard entirely (reuses no KV cache). Useful for side-by-side testing. Returned with decision `bypass`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAZ_SENTINEL_PROBE_DIR` | — (required) | Path to directory of `.npz` probe files. |
| `CAZ_SENTINEL_MODEL_ID` | — (required) | HuggingFace model id (e.g. `EleutherAI/pythia-6.9b`). |
| `CAZ_SENTINEL_DEVICE` | `cuda` | `cuda` or `cpu`. Selects bfloat16 on CUDA, float32 on CPU. |
| `CAZ_SENTINEL_REFUSAL_MESSAGE` | `"This request was blocked by the CAZ Sentinel policy."` | Refusal text returned on suppression. |
| `CAZ_SENTINEL_AUDIT_DB` | `caz_sentinel_audit.db` | SQLite file path for audit history. |
| `CHRONICLE_ENDPOINT` | — (optional) | Google SecOps Chronicle ingestion URL. If unset, Chronicle emission is a no-op. |
| `CHRONICLE_CUSTOMER_ID` | `""` | Chronicle customer id included in UDM events. |
| `CAZ_SENTINEL_URL` | `http://localhost:8000` | URL used by the Streamlit demo UI. |
| `CAZ_SENTINEL_EAGER` | — | Set to `1` to build the app at import time (for direct `uvicorn caz_sentinel.api:app` without `--factory`). |

## Probe library format

Each probe is a `.npz` file with the following keys:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `concept` | scalar | str | Human-readable concept label. |
| `layer_idx` | scalar | int | Transformer layer index (0-indexed). |
| `direction` | `[d_model]` | float32 | Unit-normalized CAZ direction vector. |
| `threshold` | scalar | float | Decision threshold in [0, 1]. |
| `pool_method` | scalar | str | Pooling method (`"last"` by default). |
| `calibration_mu` | scalar | float | Calibration mean from offline fitting. |
| `calibration_sigma` | scalar | float | Calibration std dev from offline fitting. |
| `d_model` | scalar | int | Residual-stream dimensionality. |
| `model_fingerprint` | scalar | str | Hash of model config+weights used during fitting. |

Directions are defensively re-normalized on load. All probes in the directory must have identical `d_model` and `model_fingerprint`.

## Scoring formula

For each probe, the cosine score is remapped from [-1, 1] to [0, 1]:

```
cos = (hidden @ direction) / ||hidden||
score = 0.5 * (cos + 1.0)  # now in [0, 1]
```

- Score 1.0: hidden activation perfectly aligned with probe direction.
- Score 0.5: orthogonal (no alignment).
- Score 0.0: perfectly antiparallel.

If any probe score >= its threshold, the request is **suppressed** immediately.

## Audit records

Every scored request is logged to SQLite (or queried via POST `/v1/audit`) and includes:

- `request_id`: Unique identifier.
- `timestamp_ns`: Unix timestamp in nanoseconds.
- `input_text`: The prompt that was scored.
- `per_concept_scores`: Dict of concept name → cosine score.
- `alerts`: List of concepts that met or exceeded threshold.
- `decision`: `"pass"` or `"suppressed"`.
- `latency_ms`: Scoring latency in milliseconds.

## Chronicle integration (optional)

If `CHRONICLE_ENDPOINT` is set, suppressed requests emit a Google SecOps Chronicle UDM event containing:

- Request id and timestamp.
- Input text (for security review).
- Per-concept scores and alerts.
- Model id and customer id.

Events are sent asynchronously (non-blocking); a bounded queue (1000 events max) prevents memory bloat. Excess events are logged and dropped with a warning.

## Deploying with the real pythia-6.9b model

```bash
# Requires ~14 GB VRAM (single A10 or larger)
export CAZ_SENTINEL_MODEL_ID=EleutherAI/pythia-6.9b
export CAZ_SENTINEL_PROBE_DIR=/path/to/real/probes
export CAZ_SENTINEL_DEVICE=cuda
uvicorn caz_sentinel.api:app --factory --host 0.0.0.0 --port 8000
```

The offline probe library (produced by the CAZ Sentinel fitting pipeline on pythia-6.9b bfloat16) must be placed in `CAZ_SENTINEL_PROBE_DIR` before starting.

## Demo UI (A/B side-by-side)

```bash
CAZ_SENTINEL_URL=http://localhost:8000 \
uv run streamlit run implementations/caz_sentinel/ui/demo_app.py
```

Shows two panels: Sentinel OFF (raw generation via bypass route) and Sentinel ON (guarded generation with per-concept score bars).

## Known limits

- Single-GPU only (no tensor parallelism).
- Streaming pass path re-tokenizes the prompt (no KV cache reuse) — necessary limitation of `TextIteratorStreamer`.
- Probe library must be re-calibrated if the model weights change.
- Chronicle UDM events are fire-and-forget with a bounded queue; excess events are dropped with a warning log.

## Development

### Test fixtures

```bash
# Build synthetic probes for testing (9 concepts, pythia-70m)
uv run python implementations/caz_sentinel/scripts/build_synthetic_probes.py

# Outputs to: implementations/caz_sentinel/tests/fixtures/synthetic_probes/
```

### Run tests

```bash
uv run pytest implementations/caz_sentinel/tests/
```

### Development server

```bash
# Auto-reload on file changes (from repo root)
CAZ_SENTINEL_PROBE_DIR=implementations/caz_sentinel/tests/fixtures/synthetic_probes \
  CAZ_SENTINEL_MODEL_ID=EleutherAI/pythia-70m \
  CAZ_SENTINEL_DEVICE=cuda \
  uv run uvicorn caz_sentinel.api:build_app --factory --reload --host 0.0.0.0 --port 8000
```

## Implementation notes

- **Hooks for hidden state capture**: Registered on target transformer layers during forward pass; removed immediately after.
- **No model state mutation**: Scoring is pure; no parameters updated.
- **Defensive re-normalization**: Probe directions are re-normalized on load to handle floating-point drift.
- **Threshold comparison**: `score >= threshold` triggers suppression (inclusive).
- **KV cache handoff**: On pass, the KV cache from the scoring forward is passed to `model.generate()` to avoid duplicate compute.

## License

See `LICENSE.md`.
