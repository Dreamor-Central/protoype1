from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.tavily import TavilyTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
import os
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY. Please set it in your environment variables.")

# Initialize model
groq_model = Groq(id="deepseek-r1-distill-llama-70b")

# AWS Clients
personalize_client = boto3.client('personalize-runtime', region_name='us-east-1')
comprehend_client = boto3.client("comprehend", region_name="us-east-1")

# Function for sentiment analysis
def analyze_sentiment(text):
    response = comprehend_client.detect_sentiment(Text=text, LanguageCode="en")
    return response["Sentiment"]

# Scraping Agent
scraping_agent = Agent(
    name="scraping agent",
    role="Extracts product details from Vasavi's website, including category, price, and other details",
    tools=[Crawl4aiTools()],
    model=groq_model,
    instructions=[
        "Scrape ONLY the website 'https://vasavi.co/' for the latest fashion product details.",
        "Extract categories, products, prices, and descriptions, and display them in a table format.",
        "Ensure data is structured and properly formatted for easy use.",
        "Do not scrape any external sources, only Vasavi's website."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Web Search Agent
web_search_agent = Agent(
    name="web search agent",
    role="Answers fashion-related & general queries to keep customers engaged",
    model=groq_model,
    tools=[TavilyTools()],
    instructions=[
        "Search the web for real-time fashion news, trends, and celebrity styles.",
        "Provide informative and engaging answers to general user queries.",
        "Ensure responses are up-to-date and relevant to the customer."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Sentiment Agent
sentiment_agent = Agent(
    name="sentiment agent",
    role="Adjusts responses based on customer mood to make the chatbot feel human",
    model=groq_model,
    instructions=[
        "Analyze customer messages to detect their sentiment (happy, confused, frustrated, excited).",
        "Adjust response tone to match their emotions and keep them engaged.",
        "If the customer seems confused, provide extra styling tips or discounts."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Function to get combined response
def get_combined_response(user_input):
    # Sentiment Analysis
    sentiment = analyze_sentiment(user_input)

    # Web Search Result
    web_search_result = web_search_agent.run(user_input)

    # Scraping Vasavi's Website (Corrected)
    scraped_data = agent.print_response("Tell me about https://vasavi.co.")

    # Formatting the final response
    response = f"""
**User Query:** {user_input}

🛍 **Fashion Trends & Insights:**
{web_search_result}

💡 **Suggested Products from Vasavi:**
{scraped_data}

😊 **Sentiment Analysis Result:** {sentiment}
_(Adjusting response tone accordingly)_
    """

    return response



if __name__ == "__main__":
    user_query = "I want to try a new fashion style, but I’m unsure. Can you help?"
    
    # Get the response from all agents combined
    final_response = get_combined_response(user_query)

    # Display the results
    print("Final Response:\n", final_response)
