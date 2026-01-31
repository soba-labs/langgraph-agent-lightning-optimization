"""Custom adapter for LangGraph agents to work with AgentLightning APO."""

from typing import List, Sequence
from agentlightning.adapter import TraceToMessages
from agentlightning.adapter.messages import OpenAIMessages
from agentlightning.types import Span


class LangGraphAdapter(TraceToMessages):
    """
    Adapter for LangGraph agents that handles missing gen_ai attributes.

    LangGraph doesn't automatically produce gen_ai.* attributes like OpenAI does,
    so this adapter returns an empty list for traces without those attributes.

    Note: APO will use reward signals for optimization when messages aren't available.
    This is less efficient than full message-based optimization but still functional.
    """

    def adapt(self, source: Sequence[Span], /) -> List[OpenAIMessages]:
        """
        Transform trace spans into OpenAI chat payloads, handling LangGraph's format.

        LangGraph traces don't contain gen_ai.* attributes, so we check for their
        presence and return an empty list if they're missing. This allows APO to
        continue using reward-based optimization without message analysis.

        Args:
            source: Spans that may or may not contain gen_ai.* attributes.

        Returns:
            List of OpenAI message dicts, or empty list if spans lack gen_ai attributes.
        """
        # Check if any span has gen_ai attributes
        has_genai_attrs = False
        for span in source:
            if any(k.startswith("gen_ai.") for k in span.attributes.keys()):
                has_genai_attrs = True
                break

        # If no gen_ai attributes found (typical for LangGraph), return empty
        if not has_genai_attrs:
            return []

        # Otherwise, use parent's implementation for OpenAI-style traces
        try:
            return super().adapt(source)
        except (ValueError, KeyError):
            # If parent adapter fails, return empty list as fallback
            return []
