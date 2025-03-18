import sys
from phi.agent import Agent
from textRecom import get_fashion_recommendations  # Only Vasavi products
from imageRecom import image_recommendation
from customercare import handle_customer_query
from dotenv import load_dotenv
import re

load_dotenv()

# System prompt for AI assistant
SYSTEM_PROMPT = """
You are Vasavi's Virtual Stylist – an expert in Vasavi's fashion collections and customer support.
Your role is to be **friendly, professional, and engaging** while assisting customers with Vasavi products.

**Your Responsibilities:**
- ONLY recommend Vasavi products.
- Compliment users' fashion choices and suggest Vasavi items to enhance their style.
- Use friendly phrases like *"Great choice from Vasavi!"*, *"That's an amazing Vasavi style!"*, and *"Happy to help you find the perfect Vasavi outfit!"*.
- **Never suggest non-Vasavi products** and always refer to Vasavi’s official store.
- Provide customer support contact: **support@vasavi.co**.
"""

# Define individual agents
customer_care_agent = Agent(
    name="CustomerCareAgent",
    role="Handles customer inquiries and support issues.",
    functions=handle_customer_query
)

text_recommendation_agent = Agent(
    name="TextRecommendationAgent",
    role="Suggests Vasavi outfits based on user queries.",
    functions=get_fashion_recommendations
)

image_recommendation_agent = Agent(
    name="ImageRecommendationAgent",
    role="Finds Vasavi products based on uploaded images.",
    functions=image_recommendation
)

# Create a multi-agent system, ensuring only Vasavi recommendations
multi_agent = Agent(
    name="Vasavi AI Assistant",
    role="Your go-to Vasavi stylist and shopping expert!",
    team=[
        customer_care_agent,
        text_recommendation_agent,
        image_recommendation_agent
    ],
    instructions=[
        "Direct user queries to the most suitable agent.",
        "Ensure structured and engaging responses.",
        "NEVER provide recommendations that are not from Vasavi's collection."
    ],
    show_tool_calls=False
)

# Store chat history
chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]

def chat_with_multi_agent(user_input):
    """Routes user queries to the appropriate agents while maintaining conversation context."""
    
    context = " ".join(msg["content"] for msg in chat_memory[-5:] if msg["role"] != "system")
    user_input_with_context = f"{context}\nUser: {user_input}" if context else user_input

    fashion_keywords = ["recommend", "outfit", "style", "wear", "suggest", "buy", "pair with"]
    
    if any(word in user_input.lower() for word in fashion_keywords):
        response = text_recommendation_agent.run(user_input_with_context)
    elif "image" in user_input.lower() or "upload" in user_input.lower():
        response = image_recommendation_agent.run(user_input_with_context)
    elif "return" in user_input.lower() or "refund" in user_input.lower() or "customer care" in user_input.lower():
        response = customer_care_agent.run(user_input_with_context)
    else:
        response = "I can only help with Vasavi fashion recommendations. Let me know what you're looking for!"

    chat_memory.append({"role": "user", "content": user_input})
    chat_memory.append({"role": "assistant", "content": response})

    return response

def process_image(image_path):
    """Handles image-based queries using the ImageRecommendationAgent."""
    try:
        raw_response = image_recommendation(image_path)
        return raw_response
    except Exception as e:
        return f"I'm sorry, I couldn't process that image. Error: {str(e)}"

def chat():
    """Interactive CLI chat for testing (does not interfere with Streamlit)."""
    print("\n👋 Welcome to Vasavi AI! I'm your personal stylist. How can I assist you today?\n")

    while True:
        user_input = input("🗣️ You: ")

        exit_keywords = ["exit", "quit", "bye", "goodbye", "see you", "thanks"]
        if any(keyword in user_input.lower() for keyword in exit_keywords):
            print("\n👋 Vasavi AI: Thanks for chatting with Vasavi AI! Stay stylish! 💃✨")
            break

        response = chat_with_multi_agent(user_input)

        print("\n🤖 Vasavi AI:")
        print(response, "\n")

if __name__ == "__main__":
    chat()