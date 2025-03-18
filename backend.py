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
    
    # If response is already a clean string, return it
    if isinstance(response, str) and not any(marker in response for marker in ["content=", "metrics=", "tokens=", "event="]):
        # Just handle escape sequences
        return response.replace("\\n", "\n").replace('\\"', '"')
    
    # Convert to string if it's another type
    if not isinstance(response, str):
        try:
            # If it's a dictionary with a content field, extract that
            if isinstance(response, dict) and "content" in response:
                return response["content"]
            response = str(response)
        except:
            return "I'm sorry, I couldn't process that request."
    
    # Try to extract content with regex patterns
    # Pattern 1: Look for content="..." or content="""..."""
    content_match = re.search(r'content=["\']+(.*?)["\']+', response, re.DOTALL)
    if content_match:
        return content_match.group(1).replace("\\n", "\n").replace('\\"', '"')
    
    # Pattern 2: Look for content following a specific format
    content_match = re.search(r'content=(.*?)(?:metrics=|tokens=|$)', response, re.DOTALL)
    if content_match:
        extracted = content_match.group(1).strip()
        # Remove enclosing quotes if present
        if (extracted.startswith('"') and extracted.endswith('"')) or (extracted.startswith("'") and extracted.endswith("'")):
            extracted = extracted[1:-1]
        return extracted.replace("\\n", "\n").replace('\\"', '"')
    
    # Pattern 3: Just return the first part before any metadata markers
    for marker in ["metrics=", "tokens=", "event=", "content_type="]:
        if marker in response:
            cleaned = response.split(marker)[0].strip()
            return cleaned.replace("\\n", "\n").replace('\\"', '"')
    
    # If all else fails, do some basic cleanup
    # Remove common metadata patterns
    cleaned = re.sub(r'metrics=.*?($|\n)|tokens=.*?($|\n)|event=.*?($|\n)|content_type=.*?($|\n)', '', response)
    # Remove trailing metadata indicators
    cleaned = re.sub(r'\s*[,\}\]]\s*$', '', cleaned)
    
    return cleaned.replace("\\n", "\n").replace('\\"', '"').strip()

def chat_with_multi_agent(user_input):
    """Routes user queries to the appropriate agents while maintaining conversation context."""
    
    # Keep last 5 messages for continuity
    context = " ".join(msg["content"] for msg in chat_memory[-5:] if msg["role"] != "system")
    user_input_with_context = f"{context}\nUser: {user_input}" if context else user_input

    try:
        raw_response = multi_agent.run(user_input_with_context)
        response = clean_response(raw_response)
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        response = "I'm sorry, I encountered an issue while processing your request."

    # Store conversation history
    chat_memory.append({"role": "user", "content": user_input})
    chat_memory.append({"role": "assistant", "content": response})

    return response

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

        # Exit conditions
        exit_keywords = ["exit", "quit", "bye", "goodbye", "see you", "thanks"]
        if any(keyword in user_input.lower() for keyword in exit_keywords):
            print("\n👋 Vasavi AI: Thanks for chatting with Vasavi AI! Stay stylish! 💃✨")
            break

        response = chat_with_multi_agent(user_input)

        print("\n🤖 Vasavi AI:")
        print(response, "\n")

# Run CLI chat only if executed directly
if __name__ == "__main__":
    chat()