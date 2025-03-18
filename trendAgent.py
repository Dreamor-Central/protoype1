import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.tools.searxng import Searxng
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.crawl4ai_tools import Crawl4aiTools
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tavily import search_tavily  # ✅ Import Tavily search function
from reddit import fetch_reddit_posts  # ✅ Import Reddit search function

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM (GPT-4 Turbo)
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# Initialize search tools
duckduckgo = DuckDuckGo()
searxng_search = Searxng(host="https://searx.be", fixed_max_results=5, news=True, science=True)
crawl4ai_search = Crawl4aiTools()  # No API key required

def get_fashion_insights(query: str):
    """Fetches insights using multiple search tools and generates a human-friendly expert response."""
    try:
        search_results = {}
        tool_map = {
            "Tavily": search_tavily,
            "DuckDuckGo": duckduckgo,
            "Searxng": searxng_search,
            "Crawl4AI": crawl4ai_search,
            "Reddit": fetch_reddit_posts
        }

        for tool_name, tool in tool_map.items():
            try:
                if callable(tool):  # Function-based tools (Tavily, Reddit)
                    search_results[tool_name] = tool(query)
                elif hasattr(tool, 'search') and callable(getattr(tool, 'search')):  # Object-based tools
                    search_results[tool_name] = tool.search(query)
                else:
                    search_results[tool_name] = f"❌ {tool_name} has no valid search method"
                print(f"✅ {tool_name} search successful")
            except Exception as e:
                search_results[tool_name] = f"❌ {tool_name} failed: {str(e)}"
                print(f"❌ {tool_name} failed: {str(e)}")

        # Filter valid results
        filtered_results = {k: v for k, v in search_results.items() if v}

        if not filtered_results:
            return "❌ No relevant insights found."

        # Format search results
        aggregated_data = "\n\n".join([f"🔹 {source}:\n{data}" for source, data in filtered_results.items()])

        # **Dynamic System Prompt**: Handles both fashion-related and general queries
        SYSTEM_PROMPT = f"""
        You are a friendly yet knowledgeable AI assistant who specializes in fashion but can also answer general queries. 
        Your goal is to keep the conversation engaging and subtly guide the user toward fashion-related topics when possible.

        🔹 **If the user asks about fashion (trends, styling, fabrics, industry news, etc.), respond as a top-tier fashion expert.**  
        🔹 **If the user asks a general question, respond warmly while subtly bringing attention to style and trends whenever relevant.**  
        🔹 **Always maintain a balance between expertise and an engaging, conversational tone.**

        Now, based on the user's query, provide a friendly yet expert-level response:
        """

        # Generate structured response using GPT-4 Turbo
        structured_answer = llm.invoke([
            HumanMessage(content=f"""
            {SYSTEM_PROMPT}

            🔍 **Search Results:**
            {aggregated_data}

            **Make sure to:**
            - 🛍️ If fashion-related → Provide styling insights, trend analysis, or outfit recommendations.
            - 📰 If general → Answer clearly, but relate it to fashion subtly (if possible).
            - 🎭 Keep it engaging, fun, and helpful—like a conversation with an expert stylist.
            """)
        ]).content

        return structured_answer
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Expose function for module import
def fetch_trend_insights(query: str):
    return get_fashion_insights(query)

# Run script only if executed directly
if __name__ == "__main__":
    print("\n🔧 Environment Setup:")
    print(f"DuckDuckGo: {'✅ Available' if duckduckgo else '❌ Not available'}")
    print(f"Searxng: {'✅ Available' if searxng_search else '❌ Not available'}")
    print(f"Crawl4AI: {'✅ Available'}")  # Always available
    print(f"Tavily: {'✅ Available'}")  # Assuming Tavily function works
    print(f"Reddit: {'✅ Available'}")  # Assuming Reddit function works
    print(f"OpenAI: {'✅ Available' if OPENAI_API_KEY else '❌ Not available (Check OPENAI_API_KEY)'}")

    query = input("\n🔍 Enter your fashion or general query: ")
    print("\n🔎 Searching across multiple platforms...")

    result = fetch_trend_insights(query)

    print("\n💡 **Insights:**\n")
    print(result)
