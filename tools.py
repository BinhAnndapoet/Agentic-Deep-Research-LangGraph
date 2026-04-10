from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def think_tool(reflection: str) -> str:
    """Use this tool for reflection and strategic planning during research."""
    return f"Thinking recorded: {reflection}"

researcher_tools = [think_tool]

try:
    from langchain_tavily import TavilySearch
    web_search_tool = TavilySearch(max_results=3, include_raw_content=False)
    researcher_tools.append(web_search_tool)
    print("[Tools] ✅ Đã tải Tavily Web Search.")
except ImportError:
    print("[Tools] ⚠️ Cảnh báo: langchain-tavily chưa được cài đặt.")