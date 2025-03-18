import sys
from phi.agent import Agent
from trendAgent import fetch_trend_insights
from textRecom import get_fashion_recommendations
from imageRecom import image_recommendation
from customercare import handle_customer_query
from dotenv import load_dotenv
import openai

load_dotenv()

# System prompt for AI assistant
SYSTEM_PROMPT = """
You are Vasavi's Virtual Salesman – an expert in fashion, trends, and customer support. 
Your role is to be **friendly, professional, and engaging** while assisting customers.

**What You Do:**
- Introduce yourself and Vasavi's services warmly.
- Compliment users' fashion choices and encourage engagement.
- Say things like *"Great question!"*, *"That's an excellent choice!"*, and *"I'm happy to assist!"*.
- Ensure a smooth conversation flow and provide well-structured responses.
- **Remember previous conversations** to maintain context and improve user experience.
- If a user’s query is unrelated to fashion, forward it to the **Trend Agent** for a relevant response.
- End conversations gracefully when the user’s doubts are resolved, saying something like *"Thank you for chatting with Vasavi AI! Stay stylish!"*.
- Provide **Vasavi's website** [https://vasavi.co/](https://vasavi.co/) and **Instagram handle (@vasavi.co)** when users inquire about the brand.
- For customer support, direct users to **support@vasavi.co**.
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

# Create an agent team
multi_agent = Agent(
    name="Vasavi AI Assistant",
    role="I am your ready-to-go virtual salesman/stylist who will make you ready for every special occasion!",
    team=[
        customer_care_agent,
        fashion_trend_agent,
        text_recommendation_agent,
        image_recommendation_agent
    ],
    instructions=[
        "Route user queries to the most relevant agent.",
        "Ensure structured responses and friendly engagement.",
        "If uncertain, prioritize the fashion trend agent before falling back to GPT-4."
    ],
    show_tool_calls=False 
)

# Store chat history
chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]

def chat_with_multi_agent(user_input):
    """Handles user queries by routing them to appropriate agents while maintaining context."""
    
    # Add previous chat context for better conversation flow
    context = " ".join(msg["content"] for msg in chat_memory[-5:])
    user_input_with_context = f"{context}\nUser: {user_input}"

    # ✅ FIX: Use `.run()` instead of `.invoke()`
    response = multi_agent.run(user_input_with_context)

    # Store the conversation in chat memory
    chat_memory.append({"role": "user", "content": user_input})
    chat_memory.append({"role": "assistant", "content": response})

    return response

def chat():
    """Handles interactive chat with the Vasavi AI Assistant."""
    print("\n👋 **Welcome to Vasavi AI Assistant!** I'm your virtual fashion expert. How can I assist you today?\n")

    while True:
        user_input = input("🗣️ **You:** ")

        # Exit conditions with multiple keywords
        exit_keywords = ["exit", "quit", "bye", "goodbye", "see you", "thanks"]
        if any(keyword in user_input.lower() for keyword in exit_keywords):
            print("\n👋 **Vasavi AI:** Thank you for chatting with Vasavi AI! Stay stylish! 💃✨")
            break  # Exit loop

        response = chat_with_multi_agent(user_input)

        print("\n🤖 **Vasavi AI Response:**\n")
        print(response, "\n")

if __name__ == "__main__":
    chat()
