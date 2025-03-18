import sys
from phi.agent import Agent
from trendAgent import fetch_trend_insights
from textRecom import get_fashion_recommendations
from imageRecom import image_recommendation
from customercare import handle_customer_query
from dotenv import load_dotenv
import re

load_dotenv()

# System prompt for AI assistant
SYSTEM_PROMPT = """
You are Vasavi's Virtual Stylist – an expert in Vasavi's fashion collections, trends, and customer support.
Your role is to be **friendly, professional, and engaging** while assisting customers with Vasavi products.

**Your Responsibilities:**
- Greet users warmly and introduce Vasavi's styling services.
- Always recommend Vasavi products exclusively.
- Compliment users' fashion choices and suggest how Vasavi items can enhance their style.
- Use friendly phrases like *"Great choice from Vasavi!"*, *"That's an amazing Vasavi style!"*, and *"Happy to help you find the perfect Vasavi outfit!"*.
- Maintain smooth conversation flow and provide structured responses about Vasavi products.
- **Remember previous conversations** to offer a personalized Vasavi shopping experience.
- If a user's query is unrelated to fashion, redirect it to fashion and Vasavi products.
- Conclude conversations politely, such as *"Thanks for chatting with Vasavi AI! Stay stylish with Vasavi!"*.
- Share **Vasavi's website** [https://vasavi.co/](https://vasavi.co/) and **Instagram (@vasavi.co)** for brand inquiries.
- Provide customer support contact: **support@vasavi.co**.
"""

# Define individual agents
customer_care_agent = Agent(
    name="CustomerCareAgent",
    role="Handles customer inquiries and support issues.",
    functions=handle_customer_query
)

fashion_trend_agent = Agent(
    name="FashionTrendAgent",
    role="Provides insights on fashion trends and general queries.",
    functions=fetch_trend_insights
)

text_recommendation_agent = Agent(
    name="TextRecommendationAgent",
    role="Suggests outfits based on user queries.",
    functions=get_fashion_recommendations
)

image_recommendation_agent = Agent(
    name="ImageRecommendationAgent",
    role="Finds similar fashion items based on uploaded images.",
    functions=image_recommendation
)

# Create a multi-agent system
multi_agent = Agent(
    name="Vasavi AI Assistant",
    role="Your go-to virtual stylist and fashion expert!",
    team=[
        customer_care_agent,
        fashion_trend_agent,
        text_recommendation_agent,
        image_recommendation_agent
    ],
    instructions=[
        "Direct user queries to the most suitable agent.",
        "Ensure structured and engaging responses.",
        "When unsure, prioritize the fashion trend agent before falling back to GPT-4."
    ],
    show_tool_calls=False
)

# Store chat history
chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]

def clean_response(response):
    """Thoroughly cleans the agent response to extract only the human-readable content."""
    if response is None:
        return "I'm sorry, I couldn't process that request."
    
    if isinstance(response, str) and not any(marker in response for marker in ["content=", "metrics=", "tokens=", "event="]):
        return response.replace("\\n", "\n").replace('\\"', '"')
    
    if not isinstance(response, str):
        try:
            if isinstance(response, dict) and "content" in response:
                return response["content"]
            response = str(response)
        except:
            return "I'm sorry, I couldn't process that request."
    
    content_match = re.search(r'content=["\']+(.*?)["\']+', response, re.DOTALL)
    if content_match:
        return content_match.group(1).replace("\\n", "\n").replace('\\"', '"')
    
    content_match = re.search(r'content=(.*?)(?:metrics=|tokens=|$)', response, re.DOTALL)
    if content_match:
        extracted = content_match.group(1).strip()
        if (extracted.startswith('"') and extracted.endswith('"')) or (extracted.startswith("'") and extracted.endswith("'")):
            extracted = extracted[1:-1]
        return extracted.replace("\\n", "\n").replace('\\"', '"')
    
    for marker in ["metrics=", "tokens=", "event=", "content_type="]:
        if marker in response:
            cleaned = response.split(marker)[0].strip()
            return cleaned.replace("\\n", "\n").replace('\\"', '"')
    
    cleaned = re.sub(r'metrics=.*?($|\n)|tokens=.*?($|\n)|event=.*?($|\n)|content_type=.*?($|\n)', '', response)
    cleaned = re.sub(r'\s*[\}\]]\s*$', '', cleaned)
    
    return cleaned.replace("\\n", "\n").replace('\\"', '"').strip()

def chat_with_multi_agent(user_input):
    """Routes user queries based on keywords to the appropriate agents."""
    keywords = {
        "customer support": customer_care_agent,
        "help": customer_care_agent,
        "refund": customer_care_agent,
        "return": customer_care_agent,
        "trend": fashion_trend_agent,
        "fashion": fashion_trend_agent,
        "style": fashion_trend_agent,
        "recommend": text_recommendation_agent,
        "suggest": text_recommendation_agent,
        "image": image_recommendation_agent,
        "photo": image_recommendation_agent
    }
    
    for key, agent in keywords.items():
        if key in user_input.lower():
            raw_response = agent.run(user_input)
            return clean_response(raw_response)
    
    raw_response = multi_agent.run(user_input)
    return clean_response(raw_response)

def process_image(image_path):
    """Handles image-based queries using the ImageRecommendationAgent."""
    try:
        raw_response = image_recommendation(image_path)
        return clean_response(raw_response)
    except Exception as e:
        return f"I'm sorry, I couldn't process that image. Error: {str(e)}"

def chat():
    """Interactive CLI chat for testing (does not interfere with Streamlit)."""
    print("\n👋 Welcome to Vasavi AI! I'm your personal stylist. How can I assist you today?\n")

    while True:
        user_input = input("🗣️ You: ")

        if any(keyword in user_input.lower() for keyword in ["exit", "quit", "bye", "goodbye", "see you", "thanks"]):
            print("\n👋 Vasavi AI: Thanks for chatting with Vasavi AI! Stay stylish! 💃✨")
            break

        response = chat_with_multi_agent(user_input)
        print("\n🤖 Vasavi AI:")
        print(response, "\n")

if __name__ == "__main__":
    chat()