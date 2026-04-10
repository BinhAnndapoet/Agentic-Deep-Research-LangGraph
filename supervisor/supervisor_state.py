"""
State Definitions and Pydantic Schemas for the Supervisor and Global Graph.

This module defines:
1. The global state shared across the entire research system (master graph).
2. Structured output schemas used by the Supervisor to decompose a research task
   into parallelizable sub-topics.
"""

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GlobalResearchState(TypedDict):
    """
    Represents the global state of the entire research workflow.

    This state is used as the shared data structure for the master graph,
    enabling coordination between different components such as the Supervisor,
    parallel Researchers, and the final report generator.

    Attributes:
        messages (List[BaseMessage]):
            The full conversation history across the system.
            This field uses `add_messages` to automatically append new messages
            as the workflow progresses.

        research_brief (str):
            The finalized research brief generated during the scoping phase.
            This serves as the primary input for downstream decomposition and research.

        sub_topics (List[str]):
            A list of independent sub-topics derived from the research brief.
            These topics are intended to be processed in parallel by multiple
            researcher nodes.

        aggregated_notes (List[str]):
            A collection of notes or findings returned by parallel researcher nodes.
            This field is annotated with `operator.add` to support fan-in behavior,
            meaning results from multiple parallel branches are merged into a single list.

            [IMPORTANT]
            This enables scalable parallel research by safely combining outputs
            from multiple concurrent executions.

        final_report (Optional[str]):
            The synthesized final report that combines all research findings.
            This is typically generated after all sub-topics have been processed.
            It may be None until the final stage of the workflow is completed.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    sub_topics: List[str]
    
    aggregated_notes: Annotated[List[str], operator.add]
    
    final_report: Optional[str]


class SubTopicExtraction(BaseModel):
    """
    Schema enforcing structured output from the Supervisor LLM
    for research task decomposition.

    This model ensures that the Supervisor:
    1. Explicitly explains its reasoning for breaking down the task.
    2. Produces a set of independent sub-topics suitable for parallel execution.

    Attributes:
        reasoning (str):
            A concise explanation of how the research brief was analyzed and
            decomposed into sub-topics. This helps with transparency and debugging.

        topics (List[str]):
            A list of clearly defined, non-overlapping sub-topics.
            Each topic should be:
                - Independent (can be researched in isolation)
                - Specific (not vague or overly broad)
                - Collectively covering the full scope of the research brief
    """
    reasoning: str = Field(
        description="Explanation of how the research brief is analyzed and decomposed into sub-topics."
    )
    topics: List[str] = Field(
        description="A list of independent and well-defined sub-topics that can be researched in parallel."
    )