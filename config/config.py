
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# LLM dành cho các tác vụ Logic, Trích xuất, Giám sát (Cần độ chính xác cao)
llm_logic = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_retries=3
)

# LLM dành cho Agent tìm kiếm (ReAct) - Cân bằng giữa logic và linh hoạt
llm_researcher = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_retries=3
)

# LLM dành cho viết báo cáo cuối cùng (Văn phong tự nhiên hơn)
llm_writer = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    max_retries=3
)


# Số vòng lặp tối đa cho mỗi Researcher Agent (Tránh infinite loop)
MAX_RESEARCH_ITERATIONS = 4

# Số lượng tác nhân nghiên cứu con (worker) được chạy song song tối đa
MAX_CONCURRENT_RESEARCHERS = 3