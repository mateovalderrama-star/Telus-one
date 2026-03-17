"""Retroactive ingestion: convert existing MEP JSON files to Opik Traces.

This lets you visualise runs that completed before Opik was wired in.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.opik_integration.ingest \
        --mep_dir meps/openai_openai/chartqapro/test \
        [--metrics_file metrics.jsonl]
"""

import argparse
import contextlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .client import get_client


def _parse_ts(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def ingest_mep(
    mep: dict,
    client,
    metrics: Optional[dict] = None,
    project_name: str = "chartqapro-eval",
) -> None:
    """
    Convert a single MEP JSON record into a retroactive Opik trace.

    Parameters
    ----------
    mep : dict
        The raw MEP record.
    client : object
        The Opik client.
    metrics : dict, optional
        Pre-computed metrics for the sample.
    project_name : str, default 'chartqapro-eval'
        Target project.

    Returns
    -------
    None
    """
    sample = mep.get("sample", {})
    plan = mep.get("plan", {})
    vision = mep.get("vision", {})
    timestamps = mep.get("timestamps", {})
    config = mep.get("config", {})

    sample_id = sample.get("sample_id", "unknown")
    config_name = config.get("config_name", "unknown")
    question_type = sample.get("question_type", "standard")
    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    vision_parsed = vision.get("parsed", {})

    start_time = _parse_ts(timestamps.get("start"))
    end_time = _parse_ts(timestamps.get("end"))
    planner_ms = timestamps.get("planner_ms") or 0

    trace = client.trace(
        name=f"chartqapro/{sample_id}",
        start_time=start_time,
        end_time=end_time,
        input={"question": question, "expected_output": expected},
        output=vision_parsed if vision_parsed else None,
        tags=[config_name, question_type, "chartqapro", "retroactive"],
        metadata={
            "run_id": mep.get("run_id", ""),
            "config": config_name,
            "question_type": question_type,
            "schema_version": mep.get("schema_version", ""),
            "has_errors": bool(mep.get("errors")),
        },
        project_name=project_name,
    )

    # Planner span — estimate its time window from the start
    if plan.get("prompt"):
        p_start = start_time
        p_end = None
        if start_time and planner_ms:
            p_end = start_time + timedelta(milliseconds=planner_ms)
        planner_span = trace.span(
            name="planner",
            type="llm",
            start_time=p_start,
            end_time=p_end,
            input={"prompt": plan.get("prompt", "")},
            output={
                "plan": plan.get("parsed", {}),
                "parse_error": plan.get("parse_error", False),
            },
            model=config.get("planner_model", ""),
            metadata={"backend": config.get("planner_backend", "")},
        )
        planner_span.end()

    # Vision tool spans — one per ToolTrace entry
    for tt in vision.get("tool_trace", []):
        ts_start = _parse_ts(tt.get("start_ts"))
        ts_end = _parse_ts(tt.get("end_ts"))
        usage = tt.get("provider_metadata", {}).get("usage", {})
        tool_span = trace.span(
            name="vision_qa_tool",
            type="llm",
            start_time=ts_start,
            end_time=ts_end,
            input={
                "question": question,
                "plan_steps": plan.get("parsed", {}).get("steps", []),
            },
            output=vision_parsed if vision_parsed else None,
            model=tt.get("model", config.get("vision_model", "")),
            usage=usage if usage else None,
            metadata={
                "backend": tt.get("backend", config.get("vision_backend", "")),
                "elapsed_ms": tt.get("elapsed_ms"),
            },
        )
        tool_span.end()

    trace.end()

    # Log feedback scores from the matching metrics row
    if metrics:
        scores_to_log = {}
        for key in [
            "answer_accuracy",
            "judge_explanation_quality",
            "judge_hallucination_rate",
            "judge_plan_coverage",
            "judge_plan_adherence",
            "judge_faithfulness_alignment",
        ]:
            if key in metrics and isinstance(metrics[key], (int, float)):
                scores_to_log[key] = float(metrics[key])
        for name, value in scores_to_log.items():
            with contextlib.suppress(Exception):
                trace.log_feedback_score(name=name, value=value)


def ingest_dir(
    mep_dir: str,
    metrics_file: Optional[str] = None,
    project_name: str = "chartqapro-eval",
) -> int:
    """
    Bulk ingest all MEP files from a local directory into Opik.

    Parameters
    ----------
    mep_dir : str
        Path to the folder containing JSON results.
    metrics_file : str, optional
        Path to a .jsonl file with metrics data.
    project_name : str, default 'chartqapro-eval'
        Opik project identifier.

    Returns
    -------
    int
        The total number of successfully ingested records.
    """
    client = get_client()
    if client is None:
        print("[opik] No client — set OPIK_URL_OVERRIDE or OPIK_API_KEY")
        return 0

    # Build sample_id → metrics lookup if provided
    metrics_by_id: dict = {}
    if metrics_file and Path(metrics_file).exists():
        with open(metrics_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    row = json.loads(line)
                    metrics_by_id[row.get("sample_id", "")] = row

    mep_path = Path(mep_dir)
    mep_files = list(mep_path.glob("*.json"))
    if not mep_files:
        print(f"[opik] No MEP JSON files found in {mep_dir}")
        return 0

    count = 0
    for fpath in sorted(mep_files):
        try:
            mep = json.loads(fpath.read_text())
            sample_id = mep.get("sample", {}).get("sample_id", "")
            ingest_mep(
                mep,
                client,
                metrics=metrics_by_id.get(sample_id),
                project_name=project_name,
            )
            count += 1
            print(f"  ingested {sample_id}")
        except Exception as exc:
            print(f"  ERROR {fpath.name}: {exc}")

    print(f"[opik] Ingested {count}/{len(mep_files)} MEPs from {mep_dir}")
    with contextlib.suppress(Exception):
        client.flush()
    return count


def main() -> None:
    """
    Command-line interface for retroactive ingestion into Opik.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Ingest existing MEPs into Opik")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="Optional metrics.jsonl for feedback scores",
    )
    parser.add_argument("--project", default="chartqapro-eval", help="Opik project name")
    args = parser.parse_args()

    ingest_dir(args.mep_dir, args.metrics_file, project_name=args.project)


if __name__ == "__main__":
    main()
