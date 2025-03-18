import openai
from pinecone import Pinecone
import os
import json
from dotenv import load_dotenv
from PIL import Image
import IPython.display as display

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "vasavi"
IMAGE_FOLDER = "converted_images"

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
    """Generates an embedding for the query using OpenAI embeddings."""
    try:
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding  # Extract vector
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

# Function to fetch image path
def get_image_path(image_name):
    """Returns the full path of the image if it exists in the converted_images folder."""
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    return image_path if os.path.exists(image_path) else None

# Function to search Pinecone for fashion recommendations
def search_fashion_products(query, top_k=3):
    """Finds the best-matching products from Pinecone and retrieves their details."""
    embedding = get_query_embedding(query)
    if embedding is None:
        print("❌ Error: Could not generate embedding for query.")
        return []
    
    # Perform search in Pinecone
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    
    if not results.matches:
        print("⚠️ No relevant matches found.")
        return []
    
    recommendations = []
    for match in results.matches:
        metadata = match.metadata
        image_name = metadata.get("image_name", "Unknown.jpg")  # Image name in metadata
        
        recommendations.append({
            "Style Name": metadata.get("name", "Unknown"),
            "Description": metadata.get("description", "No description available"),
            "Price": metadata.get("price", "N/A"),
            "Fabric": metadata.get("fabric", "Unknown"),
            "Image Path": get_image_path(image_name),  # Get actual image path
            "Score": round(match.score, 4)
        })
    
    return recommendations

# Function to generate expert fashion advice
def generate_fashion_advice(product):
    """Uses OpenAI to generate detailed styling recommendations for the given product."""
    system_prompt = """
    You are a friendly and knowledgeable fashion stylist. 
    When a user asks for a recommendation, provide them with:
    - A warm and engaging response.
    - A detailed fashion description of the recommended outfit.
    - Styling tips, including what accessories, shoes, or colors to pair it with.
    - Situations or events where this outfit would be perfect.
    - Alternative outfit options for different styles (e.g., casual vs formal).
    Keep the tone fun, stylish, and full of personality!
    """
    
    user_prompt = f"""
    I have a product recommendation:
    - Name: {product['Style Name']}
    - Description: {product['Description']}
    - Price: {product['Price']}
    - Fabric: {product['Fabric']}
    
    Please provide a stylish, expert fashion recommendation!
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating fashion advice: {e}"

# Main function
if __name__ == "__main__":
    user_query = input("🛍️ What kind of outfit are you looking for? ")
    results = search_fashion_products(user_query)

    if results:
        print("\n🎉 **Top Fashion Recommendations:**\n")
        for idx, item in enumerate(results, 1):
            print(f"🔹 **Result {idx}:**")
            print(f"👗 Style Name: {item['Style Name']}")
            print(f"📖 Description: {item['Description']}")
            print(f"💰 Price: {item['Price']}")
            print(f"🧵 Fabric: {item['Fabric']}")
            print(f"⭐ Score: {item['Score']}")

            # Generate expert fashion advice
            advice = generate_fashion_advice(item)
            print(f"\n✨ **Fashion Expert's Advice:**\n{advice}\n")

            # Display image if found
            if item["Image Path"]:
                print(f"🖼️ Displaying image: {item['Image Path']}")
                image = Image.open(item["Image Path"])
                display.display(image)  # Works in Jupyter Notebook / IPython
            
            print("\n" + "-"*60 + "\n")
    else:
        print("❌ No suitable recommendations found.")
