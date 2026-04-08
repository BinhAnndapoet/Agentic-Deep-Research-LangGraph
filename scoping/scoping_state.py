"""
State Definitions and Pydantic Schemas for the Scoping Sub-graph.

This module defines:
1. The internal state structure used during the scoping phase.
2. Structured output schemas that enforce consistent responses from the LLM,
   particularly for clarification decisions and research question formulation.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ScopingState(TypedDict):
    """
    Represents the internal state passed between nodes within the Scoping sub-graph.

    Attributes:
        messages (Sequence[BaseMessage]):
            The conversation history, including system, user, and assistant messages.
            This field is annotated with `add_messages` to automatically append new
            messages as the graph executes.

        research_brief (str):
            A progressively refined description of the user's research request.
            This may start as a rough idea and become more specific after clarification.

        need_clarification (bool):
            A flag indicating whether additional input from the user is required
            before proceeding to the research phase.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    need_clarification: bool


class ClarifyWithUser(BaseModel):
    """
    Schema enforcing structured LLM output for the clarification decision step.

    This model ensures the LLM explicitly decides whether it has enough information
    to proceed, or if it must ask the user a follow-up question.

    Attributes:
        need_clarification (bool):
            Indicates whether further clarification from the user is required.
            - True: The system should ask a follow-up question.
            - False: The system can proceed with the research phase.

        question (str):
            A clear and concise clarifying question to ask the user when
            `need_clarification` is True. Should directly address missing or
            ambiguous information.

        verification (str):
            A short acknowledgement message confirming that the system has
            sufficient information and will proceed with the research.
            This is used only when `need_clarification` is False.
    """
    need_clarification: bool = Field(
        description="True if additional clarification from the user is required; otherwise False."
    )
    question: str = Field(
        description="A specific clarifying question to ask the user if more information is needed."
    )
    verification: str = Field(
        description="A confirmation message indicating that research will proceed without further clarification."
    )


class ResearchQuestion(BaseModel):
    """
    Schema enforcing structured LLM output for the final research brief.

    This model ensures that the LLM produces a well-defined, actionable
    research question that can be used downstream in retrieval or analysis steps.

    Attributes:
        research_topic (str):
            A precise, detailed, and unambiguous research question or topic.
            It should clearly define the scope, intent, and key aspects to investigate.
    """
    research_topic: str = Field(
        description="A detailed and concrete research question that clearly defines the scope and objective of the research."
    )