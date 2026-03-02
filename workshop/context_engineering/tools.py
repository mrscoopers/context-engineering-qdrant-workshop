"""Qdrant tool definitions for paper retrieval and recommendation."""

QDRANT_TOOLS = [
    {
        "type": "function",
        "name": "retrieve_papers_based_on_query",
        "description": (
            "Search for PubMed papers using a single natural-language query. "
            "Use this tool when the user's request can be captured as one coherent search intent "
            "with NO negative constraints (no exclusions, no 'but not about', no 'excluding')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single, focused search query that faithfully represents the user's intent. "
                        "Preserve all biomedical terms, gene names, drug names, and conditions from the user's question. "
                        "Remove conversational filler (e.g., 'Can you find me...') "
                        "but do NOT drop substantive concepts or add topics the user did not mention. "
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "recommend_papers_based_on_constraints",
        "description": (
            "Recommend PubMed papers by combining positive and negative constraints. "
            "Use this tool INSTEAD of retrieve_papers_based_on_query when the user's request "
            "contains AT LEAST ONE negative constraint — something they explicitly want to EXCLUDE "
            "from results (signaled by phrases like 'but not about', 'excluding', 'not like', "
            "'nothing about', 'shouldn't include', 'avoid')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Descriptions of what the user WANTS papers to be about. "
                        "Each item should be a self-contained descriptive phrase capturing one distinct topic. "
                        "Split into separate items ONLY when the user's intents are clearly independent concepts "
                        "that would lose meaning if merged into one sentence. "
                        "Keep as a single item when concepts naturally form one coherent research topic."
                    ),
                },
                "negative_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Descriptions of what the user does NOT want papers to be about. "
                        "Each item should be a self-contained descriptive phrase capturing one distinct exclusion. "
                        "Split into separate items when exclusions are clearly independent concepts "
                        "— do NOT merge independent exclusions, "
                        "as merging narrows the exclusion space and misrepresents the user's intent."
                    ),
                },
            },
            "required": ["negative_examples"],
        },
    },
]
