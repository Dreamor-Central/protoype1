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

# Initialize Crawling Tool
crawl_tool = Crawl4aiTools()

# Scraping Agent
scraping_agent = Agent(
    name="scraping agent",
    role="Extracts product details from Vasavi's website, including category, price, and other details",
    tools=[crawl_tool],
    model=groq_model,
    instructions=[
        "Scrape ONLY the website 'https://vasavi.co/' for the latest fashion product details when needed.",
        "Extract categories, products, prices, and descriptions, and format them in a structured table.",
        "Only scrape when fashion-related product details are requested.",
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

# Multi-Agent System
multi_agent = Agent(
    team=[web_search_agent, scraping_agent, sentiment_agent],
    model=groq_model,
    instructions=[
        "Each agent should contribute based on its expertise.",
        "Ensure responses are clear, engaging, and well-structured.",
        "The scraping agent should fetch fashion product details only when needed.",
        "The sentiment agent should adjust the tone based on user mood.",
        "The web search agent should provide fashion trends and latest news when required."
    ],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    # Testing Multi-Agent System (without explicitly triggering scraping)
    user_query = "theres this college fest i am going to attend, suggest me some trendy plus good clothing from vasavi?"
    
    response = multi_agent.run(user_query)
    
    print("Chatbot Response:", response)

