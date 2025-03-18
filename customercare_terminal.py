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
try: 
    with open("Return Refund Policy.txt", "r") as file:
        return_policy = file.read()
except FileNotFoundError:
    return_policy = "Return policy is currently unavailable. Please contact support."

# Function to query policy using LLM
def query_policy_llm(user_query):
    system_prompt = f"""
    Aye fashionista! 😍 You're Vasavi's in-house fashion guru and customer support pro. 
    Talk like a stylish Indian fashion influencer. Use casual vibes, friendly slang, 
    and LOTS of emojis! 😘 Answer using ONLY the policy below. If you don’t know, 
    say it stylishly.

    --- [Return & Refund Policy] ---
    {return_policy}
    --------------------------------

    Rules:
        1️⃣ Greet users with fun energy! ("Heyy, gorgeous! 💕")
        2️⃣ Only give info that's asked—no extra gyaan.  
        3️⃣ Use fun emojis & casual slang, like a true desi fashionista!  
        4️⃣ For products, search Pinecone. For refunds, check policy file.  
        5️⃣ If confused, suggest human support—no fake answers!  

    Example responses:
    - "Arre yaar, refunds are only for damaged pieces! No mood swings allowed! 😉✨"
    - "Babe, you’ve got 7 days to return, but only if unworn! 💃💖"
    - "Sorry love, no refunds on sale items. That’s the deal! 💕🔥"
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
                        f"📝 {p['metadata'].get('description', 'No description available')}\n"
                        f"💰 Price: ₹{p['metadata'].get('price', 'N/A')} 💸\n"
                        f"🧵 Fabric: {p['metadata'].get('fabric', 'Unknown')}\n"
                        for p in products
                    ]
                )  
        return "Oopsie! 😢 No matching products found. Try searching for something else! 🛍️"
    except Exception as e:
        return f"Uff, technical glitch! 🤯 Error: {str(e)}"

# Handle customer queries
def handle_customer_query(user_query):
    query_lower = user_query.lower().strip()

    if any(word in query_lower for word in ["refund", "return policy", "return", "exchange"]):
        return query_policy_llm(user_query)
    
    elif any(word in query_lower for word in ["do you have", "is this in stock", "size", "price", "fabric", "available"]):
        return get_product_info(user_query)
    
    elif any(word in query_lower for word in ["contact", "customer care", "support", "helpline"]):
        return "📞 **Contact Vasavi Support:** Call us at +91 98765-43210 or email support@vasavi.com 💌"
    
    elif any(word in query_lower for word in ["where is my order", "track my order", "order status", "delivery"]):
        return "🚚 **Track your order:** Visit [Vasavi Order Tracking](https://vasavi.com/track-order) and enter your order ID! 🛍️"

if __name__ == "__main__":
    print("👗 Heyy, gorgeous! 💕 Welcome to Vasavi customer support! ✨")
    user_input = input("What’s on your mind, babe? → ")
    response = handle_customer_query(user_input)
    print("\n🤖 AI Assistant:", response)
