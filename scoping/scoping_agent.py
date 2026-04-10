"""
Scoping Agent Sub-graph

This module defines a LangGraph sub-graph responsible for the *scoping phase*
of a research workflow.

Responsibilities:
1. clarify_node:
   Analyze the conversation history to determine whether the user's request
   is sufficiently clear. If not, generate a clarifying question.

2. write_brief_node:
   If the request is clear, synthesize the conversation into a well-defined
   research brief that can guide downstream research steps.
"""

from datetime import datetime
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from config import llm_logic
from .scoping_state import ScopingState, ClarifyWithUser, ResearchQuestion
from .scoping_prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt


def get_today_str() -> str:
    """
    Returns the current date formatted as a human-readable string.

    Returns:
        str: Current date in the format "Month DD, YYYY"
             (e.g., "April 09, 2026").
    """
    return datetime.now().strftime("%B %d, %Y")


def format_messages_as_text(messages) -> str:
    """
    Convert a sequence of chat messages into a plain text format suitable for prompting.

    Each message is represented as:
        "<role>: <content>"

    Args:
        messages (Sequence[BaseMessage]):
            The conversation history.

    Returns:
        str: A newline-separated string representation of the conversation.
    """
    return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# ==== GRAPH NODES ====


def clarify_node(state: ScopingState):
    """
    Determine whether the user's request is sufficiently clear to proceed.

    This node:
    1. Converts the conversation history into a text prompt.
    2. Invokes the LLM with a structured output schema (`ClarifyWithUser`).
    3. Decides whether clarification is needed.

    Behavior:
        - If clarification is required:
            Returns a follow-up question and sets `need_clarification = True`.
        - If no clarification is required:
            Returns a confirmation message and sets `need_clarification = False`.

    Args:
        state (ScopingState):
            The current graph state containing conversation messages and metadata.

    Returns:
        dict:
            Partial state update including:
            - messages: A new AI message (question or confirmation).
            - need_clarification: Boolean flag indicating next step.
    """
    messages_text = format_messages_as_text(state["messages"])
    
    formatted_prompt = clarify_with_user_instructions.format(
        messages=messages_text,
        date=get_today_str()
    )
    
    # Enforce structured JSON output from the LLM
    structured_clarifier = llm_logic.with_structured_output(ClarifyWithUser)
    response: ClarifyWithUser = structured_clarifier.invoke([HumanMessage(content=formatted_prompt)])
    
    if response.need_clarification:
        return {
            "messages": [AIMessage(content=response.question)],
            "need_clarification": True
        }
    else:
        return {
            "messages": [AIMessage(content=response.verification)],
            "need_clarification": False
        }


def write_brief_node(state: ScopingState):
    """
    Generate a finalized research brief from the conversation history.

    This node:
    1. Formats the conversation into a prompt.
    2. Invokes the LLM with a structured schema (`ResearchQuestion`).
    3. Produces a clear, actionable research topic.

    Args:
        state (ScopingState):
            The current graph state containing the full conversation history.

    Returns:
        dict:
            Partial state update including:
            - research_brief: A well-defined research topic string.
    """
    messages_text = format_messages_as_text(state["messages"])
    
    formatted_prompt = transform_messages_into_research_topic_prompt.format(
        messages=messages_text,
        date=get_today_str()
    )
    
    structured_brief_writer = llm_logic.with_structured_output(ResearchQuestion)
    response: ResearchQuestion = structured_brief_writer.invoke([HumanMessage(content=formatted_prompt)])
    
    print("\n[Scoping] Research brief successfully generated.")
    return {
        "research_brief": response.research_topic
    }


# ==== GRAPH EDGES ====


def route_after_clarify(state: ScopingState) -> Literal["write_brief_node", "__end__"]:
    """
    Route execution flow after the clarification step.

    Logic:
        - If clarification is still needed:
            Terminate the graph early so the system can return the question to the user.
        - If sufficient information is available:
            Proceed to the research brief generation step.

    Args:
        state (ScopingState):
            The current graph state.

    Returns:
        Literal["write_brief_node", "__end__"]:
            The next node to execute in the graph.
    """
    if state.get("need_clarification", False):
        return "__end__"
    return "write_brief_node"


# ==== GRAPH BUILDER ====

scoping_builder = StateGraph(ScopingState)

scoping_builder.add_node("clarify_node", clarify_node)
scoping_builder.add_node("write_brief_node", write_brief_node)

scoping_builder.add_edge(START, "clarify_node")

scoping_builder.add_conditional_edges(
    "clarify_node",
    route_after_clarify,
    {
        "write_brief_node": "write_brief_node",
        "__end__": END
    }
)

scoping_builder.add_edge("write_brief_node", END)


"""
Compiled LangGraph sub-graph for the scoping phase.

This graph:
- Starts at `clarify_node`
- Either:
    • Ends early if clarification is needed, or
    • Continues to `write_brief_node` to produce a research brief
- Always terminates after completing its responsibility
"""
scoping_subgraph = scoping_builder.compile()