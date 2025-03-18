import openai
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "vasavi"

# Validate API Keys
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys. Check .env file!")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Function to generate query embedding
def get_query_embedding(query):
    try:
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding  # Extract vector
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

# Function to search Pinecone for recommendations
def search_fashion_products(query, top_k=5):
    embedding = get_query_embedding(query)
    if embedding is None:
        print("❌ Error: Could not generate embedding for query.")
        return []
    
    # Perform search in Pinecone
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    
    if not results.matches:
        print("⚠️ No relevant matches found.")
        return []
    
    # Print results
    recommendations = []
    print("\n🔍 **Search Results:**")
    for match in results.matches:
        metadata = match.metadata
        recommendations.append({
            "Style Name": metadata.get("name", "Unknown"),
            "Description": metadata.get("description", "No description"),
            "Price": metadata.get("price", "N/A"),
            "Fabric": metadata.get("fabric", "Unknown"),
            "Score": round(match.score, 4)
        })
    
    return recommendations

# Example usage
if __name__ == "__main__":
    user_query = input("Enter a description of the style you're looking for: ")
    results = search_fashion_products(user_query)
    
    if results:
        for idx, item in enumerate(results, 1):
            print(f"\n🔹 **Result {idx}:**")
            print(f"👗 Style Name: {item['Style Name']}")
            print(f"📖 Description: {item['Description']}")
            print(f"💰 Price: {item['Price']}")
            print(f"🧵 Fabric: {item['Fabric']}")
            print(f"⭐ Score: {item['Score']}")
