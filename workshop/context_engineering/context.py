import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from workshop import config
from workshop.context_engineering.prompts import QDRANT_PROMPT, format_summary_prompt
from workshop.context_engineering.search_engine_query import QdrantQuery
from workshop.context_engineering.tools import QDRANT_TOOLS

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


@dataclass
class ContextEngineeringResult:
    """Result from the context engineering pipeline."""
    qdrant_results: list[dict]
    agent_summary: str
    tool_used: str
    tool_args: dict[str, Any] = field(default_factory=dict)


def get_context(question: str, limit: int = 5) -> ContextEngineeringResult:
    """Run a Qdrant-backed context engineering with LLM-based tool routing.

    1. LLM (Agent) selects the right Qdrant tool (hybrid retrieval or recommendations with constraints)
    2. Executes the selected tool
    3. Summarizes results

    Args:
        question: The user's natural language question.
        limit: Number of papers Qdrant returns.

    Returns:
        ContextEngineeringResult.
    """
    qdrant = QdrantQuery()
    tool_name = "unknown"
    tool_args: dict[str, Any] = {}

    try:
        # Phase 1: LLM selects the tool
        response = openai_client.responses.create(
            model=config.OPENAI_MODEL,
            instructions=QDRANT_PROMPT,
            input=question,
            tools=QDRANT_TOOLS,
            tool_choice="required",  # LLM must call a tool, no plain-text replies allowed
            parallel_tool_calls=False,  # only one tool call per response
        )

        results = []
        # Extract the single tool call from the response
        tool_call = next(
            (item for item in response.output if item.type == "function_call"),
            None,
        )
        if tool_call:
            tool_name = tool_call.name
            # arguments arrive as a JSON string — parse into a dict
            args = (
                json.loads(tool_call.arguments)
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )
            args["limit"] = limit  # limit is user-defined, not chosen by the LLM
            tool_args = args.copy()

            func = getattr(qdrant, tool_name, None)
            if func:
                logger.info(f"Executing: {tool_name} with args: {args}")
                results = func(**args)
                logger.info(f"Got {len(results)} results")

        # Phase 2: Summarize
        agent_summary = ""
        if results:
            system_prompt, user_message = format_summary_prompt(question, results, limit=limit)
            summary_resp = openai_client.responses.create(
                model=config.OPENAI_MODEL,
                instructions=system_prompt,
                input=user_message,
            )
            agent_summary = summary_resp.output_text.strip()

        return ContextEngineeringResult(
            qdrant_results=results,
            agent_summary=agent_summary,
            tool_used=tool_name,
            tool_args=tool_args,
        )
    finally:
        qdrant.close()
