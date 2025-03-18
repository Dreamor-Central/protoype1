import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv()

# ✅ Initialize Tavily Search Tool
def search_tavily(query: str, num_results: int = 3):
    """Uses Tavily API to perform a web search and return relevant results."""
    try:
        tavily_tool = TavilySearchResults(max_results=num_results)
        results = tavily_tool.invoke(query)

        if not results:
            print("⚠️ No relevant search results found from Tavily.")
            return []

        return [{"title": res["title"], "url": res["url"]} for res in results]

    except Exception as e:
        print(f"❌ Error with Tavily API: {e}")
        return []

# ✅ Run as standalone tool for testing
if __name__ == "__main__":
    query = input("🔍 Enter a topic to search: ").strip()
    if query:
        results = search_tavily(query)
        if results:
            print(f"\n🌍 Top {len(results)} Tavily Search Results:")
            for i, res in enumerate(results, 1):
                print(f"{i}. 📌 {res['title']}")
                print(f"   🔗 {res['url']}\n")
        else:
            print("⚠️ No relevant search results found from Tavily.")
    else:
        print("⚠️ Please enter a search query.")
