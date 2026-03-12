"""VisionAgent — tool-using CrewAI agent that calls vision_qa_tool once.

The agent orchestrates the VLM call via vision_qa_tool and returns strict JSON
{answer, explanation}.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from crewai import LLM, Agent, Crew, Task

from ..datasets.perceived_sample import PerceivedSample
from ..opik_integration.tracing import close_span, open_llm_span
from ..tools.vision_qa_tool import VisionQATool
from ..utils.json_strict import parse_strict


VISION_PROMPT_PATH = Path(__file__).parent / "prompts" / "vision.txt"

VISION_REQUIRED_KEYS = ["answer", "explanation"]


def _load_template() -> str:
    """Load the vision agent prompt template from file."""
    return VISION_PROMPT_PATH.read_text()


def build_vision_task_description(
    sample: PerceivedSample, plan: dict, ocr_result: Optional[dict] = None
) -> str:
    """Build the task description prompt for the vision agent."""
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

    steps = plan.get("steps", [])
    plan_steps_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

    ocr_block = ""
    if ocr_result:
        lines = [
            "Pre-extracted text from chart (use as ground truth for visible text):"
        ]
        if ocr_result.get("chart_type"):
            lines.append(f"  Chart type : {ocr_result['chart_type']}")
        if ocr_result.get("title"):
            lines.append(f"  Title      : {ocr_result['title']}")
        x = ocr_result.get("x_axis", {})
        if x.get("label") or x.get("ticks"):
            lines.append(
                f"  X-axis     : label={x.get('label', '')!r}  ticks={x.get('ticks', [])}"
            )
        y = ocr_result.get("y_axis", {})
        if y.get("label") or y.get("ticks"):
            lines.append(
                f"  Y-axis     : label={y.get('label', '')!r}  ticks={y.get('ticks', [])}"
            )
        if ocr_result.get("legend"):
            lines.append(f"  Legend     : {ocr_result['legend']}")
        if ocr_result.get("data_labels"):
            lines.append(f"  Data labels: {ocr_result['data_labels']}")
        if ocr_result.get("annotations"):
            lines.append(f"  Annotations: {ocr_result['annotations']}")
        ocr_block = "\n".join(lines)

    return template.format(
        image_path=sample.image_path,
        question=sample.question,
        choices_block=choices_block,
        context_block=context_block,
        ocr_block=ocr_block,
        plan_steps_block=plan_steps_block,
    )


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    """Build an LLM instance for the given backend and model.

    Abstracts away differences in API key environment variables
    for different backends.
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
    raise ValueError(f"Unknown vision agent backend: {backend!r}")


class VisionAgent:
    """
    Two-layer vision agent.

      - Orchestrator LLM: decides to call vision_qa_tool (text-only reasoning)
      - vision_qa_tool:   calls the actual VLM backend (OpenAI/Gemini vision)
    """

    def __init__(
        self,
        agent_backend: str = "gemini",
        agent_model: str = "gemini-2.5-flash-lite",
        vision_backend: str = "gemini",
        vision_model: str = "gemini-2.5-flash-lite",
        agent_api_key: Optional[str] = None,
        vision_api_key: Optional[str] = None,
    ):
        """Initialize the VisionAgent with specified backends, models, and API keys."""
        self.agent_backend = agent_backend
        self.agent_model = agent_model
        self.vision_backend = vision_backend
        self.vision_model = vision_model
        self.agent_api_key = agent_api_key
        self.vision_api_key = vision_api_key

    def _build_tool(self, opik_trace: Any = None) -> VisionQATool:
        """Build the vision_qa_tool instance with appropriate backend and model."""
        key = self.vision_api_key or (
            os.environ.get("OPENAI_API_KEY", "")
            if self.vision_backend == "openai"
            else os.environ.get("GEMINI_API_KEY", "")
        )
        return VisionQATool(
            backend=self.vision_backend,
            model=self.vision_model,
            api_key=key,
            opik_trace=opik_trace,
        )

    def run(
        self,
        sample: PerceivedSample,
        plan: dict,
        opik_trace: Any = None,
        ocr_result: Optional[dict] = None,
    ) -> Tuple[str, dict, bool, str, List[dict]]:
        """
        Run the vision agent for one sample.

        Args:
            ocr_result: Optional pre-extracted text dict from OcrReaderTool. When
                        provided it is injected into the task prompt as grounding
                        context. Pass None to skip OCR grounding.

        Returns
        -------
            task_description – rendered task prompt
            parsed           – {answer, explanation} dict
            parse_error      – True if JSON parsing needed repair or failed
            raw_text         – raw agent output
            tool_traces      – list of ToolTrace dicts from vision_qa_tool
        """
        tool = self._build_tool(opik_trace=opik_trace)
        llm = _build_llm(self.agent_backend, self.agent_model, self.agent_api_key)
        task_description = build_vision_task_description(
            sample, plan, ocr_result=ocr_result
        )

        vision_span = open_llm_span(
            opik_trace,
            name="vision_agent",
            input_data={"task_description": task_description},
            model=self.agent_model,
            metadata={"backend": self.agent_backend},
        )

        agent = Agent(
            role="Chart Reading Vision Agent",
            goal=(
                "Answer chart questions by calling vision_qa_tool exactly once, "
                "then output strict JSON with 'answer' and 'explanation'."
            ),
            backstory=(
                "You are a precise chart analysis agent. You use vision_qa_tool to "
                "inspect chart images and produce grounded, evidence-based answers. "
                "You follow inspection plans step by step and never hallucinate."
            ),
            llm=llm,
            tools=[tool],
            verbose=False,
            allow_delegation=False,
            max_iter=3,  # limit iterations to prevent runaway tool calls
        )

        task = Task(
            description=task_description,
            expected_output='JSON object: {"answer": "...", "explanation": "..."}',
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=VISION_REQUIRED_KEYS)

        tool_traces = tool.pop_traces()

        close_span(vision_span, output=parsed if parsed else {"parse_error": True})

        return task_description, parsed, not parse_ok, raw_text, tool_traces
