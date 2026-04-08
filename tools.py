from langchain_tavily import TavilySearch
    
web_search_tool = TavilySearch(max_results=3, include_raw_content=False)
    
