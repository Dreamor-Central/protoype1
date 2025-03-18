import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "vasavi"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists, otherwise create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536,  # Adjust if needed
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Your specified region
        )
    )

# Connect to existing Pinecone index
index = pc.Index(INDEX_NAME)

# System prompt for fashion recommendations
SYSTEM_PROMPT = """
You are a fashion stylist working exclusively for Vasavi, a premium clothing brand. 
Your goal is to provide **recommendations that only feature Vasavi products**. 
Be warm, friendly, and professional in your responses.

### **Your Styling Approach:**
1. **Brand Focus**
   - ONLY recommend Vasavi products from the provided recommendations.
   - Always mention "Vasavi" brand name when referring to products.
   - Highlight that these are exclusive Vasavi designs.

2. **Context Awareness**  
   - If the user asks for **tops**, recommend **only Vasavi tops from the Pinecone index**.  
   - If they need **bottoms**, suggest **only Vasavi bottoms from the Pinecone index**.  
   - For **full outfits**, ensure they include only Vasavi items.

3. **Expert-Level Insights:**  
   - Mention which **body types the Vasavi outfit suits best**.  
   - Suggest **fabric textures, seasonal wear, and trendy elements** of Vasavi products.  
   - Offer **accessorizing tips** (Vasavi shoes, jewelry, bags if available).  

4. **Engaging, Human-like Tone:**  
   - Speak like a **Vasavi personal stylist** helping a customer.  
   - Ensure the user feels confident in their Vasavi outfit choices.
   - Always close with an invitation to explore more Vasavi products.
"""

# Function to get text embeddings
def get_embedding(text: str):
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return embedding_response.data[0].embedding

# Function to fetch recommendations from Pinecone
def fetch_recommendation(query: str, top_k: int = 5):
    query_embedding = get_embedding(query)
    
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    recommendations = []
    for match in response.matches:
        metadata = match.metadata

        recommendations.append({
            "style_name": metadata.get("STYLE NAME", "Unknown"),
            "description": metadata.get("DESCRIPTION", "No description available"),
            "price": metadata.get("PRICES", "N/A"),
            "fabric": metadata.get("FABRIC DESCRIPTION", "Unknown"),
        })
    
    return recommendations

# Function to generate AI-powered fashion recommendations
def generate_response(user_query: str):
    recommendations = fetch_recommendation(user_query)
    
    if not recommendations:
        return "I couldn't find any Vasavi products matching your query. Would you like to explore our latest collection? Visit [Vasavi.co](https://vasavi.co/)"

    # Format recommendations for response
    recommendation_text = "\n".join([
        f"- **{r['style_name']}**\n"
        f"  {r['description']}\n"
        f"  **Price:** {r['price']} | **Fabric:** {r['fabric']}\n"
        f"  🔗 [Buy Now](https://vasavi.co/)"
        for r in recommendations
    ])

    return f"Here are some Vasavi products matching your request:\n\n{recommendation_text}\n\nVisit [Vasavi.co](https://vasavi.co/) for more options!"

# Function to get fashion recommendations
def get_fashion_recommendations(user_query: str):
    return generate_response(user_query)
