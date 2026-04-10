"""
Supervisor Node Logic

This module defines the core logic of the Supervisor node.

Responsibilities:
- Read the finalized research brief.
- Use an LLM to decompose the brief into a set of independent sub-topics.
- Prepare the system for parallel execution (fan-out) across multiple researcher nodes.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from config import llm_logic, MAX_CONCURRENT_RESEARCHERS
from .supervisor_state import GlobalResearchState, SubTopicExtraction
from .supervisor_prompts import supervisor_planner_prompt


def extract_sub_topics_node(state: GlobalResearchState):
    """
    Decompose the research brief into parallelizable sub-topics.

    This node:
    1. Retrieves the `research_brief` from the global state.
    2. Formats a prompt instructing the LLM to break the brief into sub-topics.
    3. Enforces structured output using the `SubTopicExtraction` schema.
    4. Logs the resulting topics for observability and debugging.
    5. Returns the list of sub-topics for downstream fan-out execution.

    The resulting sub-topics should:
        - Be independent from each other
        - Be specific and actionable
        - Collectively cover the full scope of the research brief
        - Respect the maximum concurrency limit

    Args:
        state (GlobalResearchState):
            The global graph state containing the finalized research brief.

    Returns:
        dict:
            Partial state update containing:
            - sub_topics (List[str]):
                A list of extracted sub-topics. This will be used by the
                main graph to trigger parallel execution (e.g., via Send()).
    """
    research_brief = state.get("research_brief", "")
    
    formatted_prompt = supervisor_planner_prompt.format(
        research_brief=research_brief,
        max_topics=MAX_CONCURRENT_RESEARCHERS
    )
    
    # Enforce structured JSON output from the LLM
    structured_supervisor = llm_logic.with_structured_output(SubTopicExtraction)
    response: SubTopicExtraction = structured_supervisor.invoke([HumanMessage(content=formatted_prompt)])
    
    # Log results for observability
    print(f"\n[Supervisor] Decomposed into {len(response.topics)} parallel research streams:")
    for idx, topic in enumerate(response.topics):
        print(f"  {idx+1}. {topic}")
        
    return {
        "sub_topics": response.topics
    }