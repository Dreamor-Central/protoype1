import os
import openai
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone properly
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Access Pinecone index
pinecone_index = pc.Index(name="vasavi")

# Load return & refund policy
def load_return_policy():
    try: 
        with open("Return Refund Policy.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Return policy is currently unavailable. Please contact support."

return_policy = load_return_policy()

# Function to query policy using LLM
def query_policy_llm(user_query):
    system_prompt = f"""
    Heyy, fashion queen/kings! 👑✨ You're Vasavi's in-house fashion guru and customer care expert.  
    Talk like a super friendly Indian salesman + stylish fashion expert.  
    Keep it **warm, energetic, and engaging**—but no flirting! 😆 

    --- [Return & Refund Policy] ---
    {return_policy}
    --------------------------------

    💖 **Golden Rules:**  
    1️⃣ **Always start with a warm greeting & ask how they’re doing!** 🥰  
       ("Heyy, fabulous! How's your day going? 😍")  
    2️⃣ **Keep it short & helpful—no extra gyaan!**  
    3️⃣ **Use fun emojis & casual desi slang, like a pro stylist!**  
    4️⃣ **For products, search Pinecone. For refunds, check policy file.**  
    5️⃣ **If confused, suggest human support—no fake answers!**  

    Answer like a stylish, friendly expert!  
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.8
    )

    return response.choices[0].message.content

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to get product info from Pinecone
def get_product_info(user_query):
    try:
        query_vector = embedding_model.encode(user_query).tolist()
        search_result = pinecone_index.query(vector=query_vector, top_k=3, include_metadata=True)

        if search_result and "matches" in search_result:
            products = search_result["matches"]
            if products:
                return "\n".join(
                    [
                        f"✨ **{p['metadata'].get('name', 'Unknown Product')}** 🛍️\n"
                        f"💖 {p['metadata'].get('description', 'No description available')}\n"
                        f"💰 **Price:** ₹{p['metadata'].get('price', 'N/A')} 💸\n"
                        f"🧵 **Fabric:** {p['metadata'].get('fabric', 'Unknown')}\n"
                        for p in products
                    ]
                )  
        return "Oopsie! 😭 No matching products found, babe. Try searching for something else! ✨"
    except Exception as e:
        return f"Uff, technical glitch! 🤯 Error: {str(e)}"

# Handle customer queries
def handle_customer_query(user_query):
    query_lower = user_query.lower().strip()

    # Start response with a friendly greeting 😍
    greeting = "Heyy, fashionista! 😍 How’s your day going? 🌸✨\n\n"

    if any(word in query_lower for word in ["refund", "return policy", "return", "exchange"]):
        return greeting + query_policy_llm(user_query)
    
    elif any(word in query_lower for word in ["do you have", "is this in stock", "size", "price", "fabric", "available"]):
        return greeting + get_product_info(user_query)
    
    elif any(word in query_lower for word in ["contact", "customer care", "support", "helpline"]):
        return greeting + "📞 **Contact Vasavi Support:** Call us at +91 98765-43210 or email support@vasavi.com 💌"
    
    elif any(word in query_lower for word in ["where is my order", "track my order", "order status", "delivery"]):
        return greeting + "🚚 **Track your order:** Visit [Vasavi Order Tracking](https://vasavi.com/track-order) and enter your order ID! 🛍️"
    
    return greeting + "Babe, I'm not too sure about that! 😕 Try rephrasing or contact support, I got your back! 💌"

# Expose functions for import
__all__ = ["handle_customer_query"]
