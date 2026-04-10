from datetime import datetime
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config import llm_researcher, llm_logic, MAX_RESEARCH_ITERATIONS
from tools import researcher_tools
from .researcher_state import ResearcherState
from .researcher_prompts import (
    research_agent_prompt, 
    compress_research_system_prompt,
    compress_research_human_message
)

def get_today_str() -> str:
    return datetime.now().strftime("%B %d, %Y")

llm_researcher_with_tools = llm_researcher.bind_tools(researcher_tools)

def research_agent_node(state: ResearcherState):
    current_topic = state["current_topic"]
    
    messages = state.get("messages", [])
    if not messages:
        sys_prompt = research_agent_prompt.format(date=get_today_str())
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Please research this specific topic: {current_topic}")
        ]
    
    response = llm_researcher_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def compress_research_node(state: ResearcherState):
    print(f"\n[Researcher] Đang nén dữ liệu cho topic: '{state['current_topic']}'...")
    current_topic = state["current_topic"]
    raw_messages = state["messages"]
    
    sys_prompt = compress_research_system_prompt.format(date=get_today_str())
    human_prompt = compress_research_human_message.format(research_topic=current_topic)
    
    compress_messages = [SystemMessage(content=sys_prompt)] + raw_messages + [HumanMessage(content=human_prompt)]
    
    response = llm_logic.invoke(compress_messages)
    
    print(f"[Researcher] ✅ Hoàn tất thu thập cho topic: '{current_topic}'")
    return {"aggregated_notes": [f"### Topic: {current_topic}\n\n{response.content}"]}

def route_research(state: ResearcherState) -> Literal["research_tools_node", "compress_research_node"]:
    last_message = state["messages"][-1]
    iteration_count = state.get("iteration_count", 0)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if iteration_count < MAX_RESEARCH_ITERATIONS:
            return "research_tools_node"
    return "compress_research_node"

researcher_builder = StateGraph(ResearcherState)
researcher_builder.add_node("research_agent_node", research_agent_node)
researcher_builder.add_node("research_tools_node", ToolNode(researcher_tools))
researcher_builder.add_node("compress_research_node", compress_research_node)

researcher_builder.add_edge(START, "research_agent_node")
researcher_builder.add_conditional_edges(
    "research_agent_node", route_research,
    {"research_tools_node": "research_tools_node", "compress_research_node": "compress_research_node"}
)
researcher_builder.add_edge("research_tools_node", "research_agent_node")
researcher_builder.add_edge("compress_research_node", END)

researcher_subgraph = researcher_builder.compile()