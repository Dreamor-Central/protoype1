import os
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import openai

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load ViT Model
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)  
model = ViTModel.from_pretrained(model_name)
model.eval()

# Load FAISS Index
faiss_index_path = "cloth_faiss.index"
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    raise FileNotFoundError(f"❌ FAISS index file '{faiss_index_path}' not found!")

# Path to converted images directory
converted_folder = "converted_images"
if not os.path.exists(converted_folder):
    raise FileNotFoundError(f"❌ Converted images directory '{converted_folder}' not found!")

# System Prompt for OpenAI Agent
SYSTEM_PROMPT = """
You are a dedicated Vasavi brand representative and fashion assistant. 
When a user uploads an image, your job is to find and recommend ONLY visually similar products from Vasavi's collection.
You then generate a natural, engaging response that clearly positions these as Vasavi-exclusive products.

Make sure to:
- Compliment the user's style while mentioning they'd look great in Vasavi products.
- Highlight unique features of the recommended Vasavi products.
- Always mention "Vasavi" by name multiple times in your response.
- Emphasize that these are exclusive Vasavi designs not available elsewhere.
- Encourage them to explore more Vasavi products.
- Maintain a warm, conversational tone while promoting Vasavi brand identity.
- NEVER suggest products from other brands or generic items.
"""

def get_image_embedding(image_path):
    """Extracts an embedding from an image using ViT."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return embedding.numpy()
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def find_similar_images(input_image_path, top_k=5):
    """Finds similar images in the FAISS index."""
    embedding = get_image_embedding(input_image_path)
    if embedding is None:
        return []

    embedding = np.expand_dims(embedding, axis=0)  # Reshape for FAISS
    distances, indices = index.search(embedding, top_k)

    # Fetch corresponding image paths
    image_files = sorted([os.path.join(converted_folder, f) for f in os.listdir(converted_folder) if f.endswith(".jpg")])
    similar_images = [image_files[i] for i in indices[0] if i < len(image_files)]

    return similar_images

def generate_friendly_response(image_path, similar_images):
    """Generates a friendly text response using OpenAI's LLM."""
    if not similar_images:
        return "I couldn't find any similar products right now. Maybe try a different image?"

    prompt = f"""
    The user uploaded an image of a clothing item. Based on the visual similarity, 
    I found {len(similar_images)} similar products from Vasavi’s collection.

    - These products match the user’s style.
    - Highlight unique selling points (fabric, design, or trending factors).
    - Keep the tone friendly, warm, and engaging.

    Image recommendations:
    {', '.join(similar_images)}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "Sorry, I couldn't generate a recommendation message at the moment."

def image_recommendation(input_image_path):
    """Main function to get image recommendations and generate a text response."""
    similar_images = find_similar_images(input_image_path)
    text_response = generate_friendly_response(input_image_path, similar_images)
    return {"text": text_response, "images": similar_images}

if __name__ == "__main__":
    test_image = "test_img.png"  
    if not os.path.exists(test_image):
        print(f"❌ Test image '{test_image}' not found!")
    else:
        results = image_recommendation(test_image)
        print("💬 Recommendation:", results["text"])
        print("📸 Recommended Images:", results["images"])
