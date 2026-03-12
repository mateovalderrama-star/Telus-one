"""PlannerAgent — text-only CrewAI agent that produces a strict JSON inspection plan.

The planner never sees the image. It creates a 2-4 step chart-reading procedure
that the VisionAgent will follow.
"""

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from crewai import LLM, Agent, Crew, Task

from ..datasets.perceived_sample import PerceivedSample
from ..opik_integration.tracing import close_span, open_llm_span
from ..utils.json_strict import parse_strict


PLANNER_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"

PLAN_REQUIRED_KEYS = [
    "steps",
    "expected_answer_type",
    "question_type",
    "answerability_check",
    "hints",
]


def _load_template() -> str:
    """Load the planner prompt template from a file."""
    return PLANNER_PROMPT_PATH.read_text()


def build_planner_prompt(sample: PerceivedSample) -> str:
    """Render the planner prompt for a given sample."""
    template = _load_template()

    choices_block = ""
    if sample.choices:
        choices_block = f"Choices: {', '.join(sample.choices)}"

    context_block = ""
    if sample.context:
        lines = ["Conversation context:"]
        for turn in sample.context:
            lines.append(f"  {turn.get('role', 'user')}: {turn.get('content', '')}")
        context_block = "\n".join(lines)

    return template.format(
        question=sample.question,
        question_type=sample.question_type.value,
        choices_block=choices_block,
        context_block=context_block,
    )


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    """Construct the LLM instance based on the specified backend and model.

    Reads API keys from environment variables if not provided directly.
    """
    if backend == "openai":
        return LLM(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
    if backend == "gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"Unknown planner backend: {backend!r}")


class PlannerAgent:
    """Wraps a CrewAI Agent/Crew to generate a strict JSON plan for chart inspection."""

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        """Parameters"""
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self._llm = _build_llm(backend, model, api_key)

    def run(
        self, sample: PerceivedSample, opik_trace: Any = None
    ) -> Tuple[str, dict, bool, str]:
        """
        Run the planner for one sample.

        Returns
        -------
            prompt       – the rendered prompt string
            parsed       – parsed plan dict (may be partial on error)
            parse_error  – True if JSON parsing needed repair or failed
            raw_text     – raw LLM output
        """
        prompt = build_planner_prompt(sample)

        span = open_llm_span(
            opik_trace,
            name="planner",
            input_data={"prompt": prompt},
            model=self.model,
            metadata={"backend": self.backend},
        )

        agent = Agent(
            role="Chart Reading Planner",
            goal=(
                "Produce a precise JSON inspection plan for answering a question about a chart. "
                "Output JSON only — no extra text."
            ),
            backstory=(
                "You are an expert chart analyst. You plan structured chart-reading procedures "
                "without seeing the image, so a vision agent can follow your steps precisely."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        task = Task(
            description=prompt,
            expected_output=(
                "A JSON object with keys: steps (list of 2-4 strings), "
                "expected_answer_type, question_type, answerability_check, hints"
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=PLAN_REQUIRED_KEYS)

        # Enforce step count bounds
        if parsed and "steps" in parsed:
            steps = list(parsed["steps"])
            if len(steps) < 2:
                steps += ["Check answerability from the chart data"] * (2 - len(steps))
            parsed["steps"] = steps[:4]

        close_span(
            span,
            output={"plan_steps": parsed.get("steps", []), "parse_error": not parse_ok},
        )

        return prompt, parsed, not parse_ok, raw_text
