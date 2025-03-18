import sys
from phi.agent import Agent
from trendAgent import fetch_trend_insights
from textRecom import get_fashion_recommendations
from imageRecom import image_recommendation
from customercare import handle_customer_query
from dotenv import load_dotenv

load_dotenv()

# System prompt for AI assistant
SYSTEM_PROMPT = """
You are Vasavi's Virtual Stylist – an expert in fashion, trends, and customer support.
Your role is to be **friendly, professional, and engaging** while assisting customers.

**Your Responsibilities:**
- Greet users warmly and introduce Vasavi's styling services.
- Compliment users' fashion choices and encourage engagement.
- Use friendly phrases like *"Great choice!"*, *"That’s an amazing style!"*, and *"Happy to help!"*.
- Maintain smooth conversation flow and provide structured responses.
- **Remember previous conversations** to offer a personalized experience.
- If a user’s query is unrelated to fashion, redirect it to the **Trend Agent** for a relevant response.
- Conclude conversations politely, such as *"Thanks for chatting with Vasavi AI! Stay stylish!"*.
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

def chat_with_multi_agent(user_input):
    """Routes user queries to the appropriate agents while maintaining conversation context."""
    
    # Keep last 5 messages for continuity
    context = " ".join(msg["content"] for msg in chat_memory[-5:])
    user_input_with_context = f"{context}\nUser: {user_input}"

    response = multi_agent.run(user_input_with_context)

    # Store conversation history
    chat_memory.append({"role": "user", "content": user_input})
    chat_memory.append({"role": "assistant", "content": response})

    return response

def process_image(image_path):
    """Handles image-based queries using the ImageRecommendationAgent."""
    return image_recommendation(image_path)

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
