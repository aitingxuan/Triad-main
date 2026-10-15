"""TraceMAD: structured long-form CoT evaluation for industrial anomaly answers.

Stage A (rules, offline): schema / decision / think consistency / type match.
Stage B (optional LLM judge): logic / completeness / grounding via OpenAI-compatible API.
"""

from .parse import parse_gpt_output
from .metrics_stage_a import evaluate_sample_stage_a, aggregate_stage_a

__all__ = [
    "parse_gpt_output",
    "evaluate_sample_stage_a",
    "aggregate_stage_a",
]
