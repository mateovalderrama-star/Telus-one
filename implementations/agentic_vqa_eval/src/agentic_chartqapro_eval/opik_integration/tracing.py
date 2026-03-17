"""Lightweight wrappers around opik Trace/Span for the MEP pipeline.

All helpers accept ``None`` as the client/trace and become no-ops, so the
rest of the codebase can call them unconditionally.
"""

import contextlib
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from opik.types import ErrorInfoDict


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def sample_trace(
    client,
    sample_id: str,
    question: str,
    expected_output: str,
    question_type: str,
    config_name: str,
    run_id: str,
    project_name: str = "chartqapro-eval",
):
    """
    Context manager to open and automatically close an Opik trace.

    Parameters
    ----------
    client : object
        The Opik client. If None, the context manager yields None.
    sample_id : str
        Unique identifier for the sample.
    question : str
        The input prompt text.
    expected_output : str
        The ground truth answer.
    question_type : str
        The category of the question.
    config_name : str
        The evaluation configuration used.
    run_id : str
        The unique ID of the pipeline run.
    project_name : str, default 'chartqapro-eval'
        Opik project identifier.

    Yields
    ------
    trace : object or None
        The initialized trace object.
    """
    if client is None:
        yield None
        return

    trace = client.trace(
        name=f"chartqapro/{sample_id}",
        input={"question": question, "expected_output": expected_output},
        tags=[config_name, question_type, "chartqapro"],
        metadata={
            "run_id": run_id,
            "config": config_name,
            "question_type": question_type,
        },
        project_name=project_name,
    )
    try:
        yield trace
    finally:
        trace.end()


def open_llm_span(
    trace,
    name: str,
    input_data: dict,
    model: str,
    metadata: Optional[dict] = None,
    parent_span_id: Optional[str] = None,
):
    """
    Begin a new LLM-type span within an active trace.

    Parameters
    ----------
    trace : object
        The parent trace or span.
    name : str
        Logical name for the operation.
    input_data : dict
        Model inputs.
    model : str
        Model identifier.
    metadata : dict, optional
        Additional context keys.
    parent_span_id : str, optional
        Explicit parent linkage.

    Returns
    -------
    object or None
        The active span object.
    """
    if trace is None:
        return None
    return trace.span(
        name=name,
        type="llm",
        input=input_data,
        model=model,
        metadata=metadata or {},
        parent_span_id=parent_span_id,
    )


def close_span(
    span,
    output: Optional[dict] = None,
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log results and terminate an active span.

    Parameters
    ----------
    span : object
        The span to close.
    output : dict, optional
        The result of the operation.
    usage : dict, optional
        Token usage statistics.
    error : str, optional
        Error message if the span failed.

    Returns
    -------
    None
    """
    if span is None:
        return
    kwargs: dict = {}
    if output is not None:
        kwargs["output"] = output
    if usage:
        kwargs["usage"] = usage
    if error:
        kwargs["error_info"] = ErrorInfoDict(message=error)
    span.end(**kwargs)


def log_trace_scores(trace, scores: dict) -> None:
    """
    Attach quantitative feedback scores to a trace.

    Parameters
    ----------
    trace : object
        The trace to update.
    scores : dict
        Mapping of metric names to numeric values.

    Returns
    -------
    None
    """
    if trace is None:
        return
    for name, value in scores.items():
        if isinstance(value, (int, float)):
            with contextlib.suppress(Exception):
                trace.log_feedback_score(name=name, value=float(value))
