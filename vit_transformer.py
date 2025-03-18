import os
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, ViTModel

# ✅ Check and report on image directory contents
image_folder = "vasavi_images"
converted_folder = "converted_images"

# First, verify the image folder exists
if not os.path.exists(image_folder):
    print(f"🚨 Error: Folder '{image_folder}' does not exist!")
    print(f"💡 Creating '{image_folder}' folder. Please add your images there.")
    os.makedirs(image_folder, exist_ok=True)
    exit()

# ✅ Create converted folder
os.makedirs(converted_folder, exist_ok=True)

# ✅ Recursive function to find and process all images in any subfolder
def process_images_recursively(folder_path, output_folder, prefix=""):
    processed_count = 0
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Path does not exist: {folder_path}")
        return 0
        
    # Check if this is a directory
    if not os.path.isdir(folder_path):
        print(f"⚠️ Not a directory: {folder_path}")
        return 0
        
    print(f"📂 Scanning directory: {folder_path}")
    
    try:
        files = os.listdir(folder_path)
    except Exception as e:
        print(f"❌ Error reading directory {folder_path}: {e}")
        return 0
        
    print(f"📁 Found {len(files)} items in {folder_path}")
    
    for item in files:
        full_path = os.path.join(folder_path, item)
        
        # If it's a directory, recursively process it
        if os.path.isdir(full_path):
            print(f"📂 Found subdirectory: {item}")
            sub_prefix = f"{prefix}{item}_" if prefix else f"{item}_"
            processed_count += process_images_recursively(full_path, output_folder, sub_prefix)
            continue
            
        # Try to process as an image
        try:
            # First attempt: use PIL to open the file
            with Image.open(full_path) as img:
                # It opened successfully, so it's an image! Convert and save it
                img = img.convert("RGB")
                # Clean the filename and add prefix to avoid name collisions
                base_name = os.path.splitext(os.path.basename(item))[0]
                base_name = ''.join(c for c in base_name if c.isalnum() or c in '._- ')
                new_filename = f"{prefix}{base_name}.jpg"
                new_path = os.path.join(output_folder, new_filename)
                img.save(new_path, "JPEG")
                processed_count += 1
                print(f"✅ Processed {full_path} -> {new_filename}")
        except Exception as e:
            print(f"❌ Cannot process {full_path}: {e}")
    
    return processed_count

# Process all images recursively
print(f"\n🔄 Recursively processing images from '{image_folder}' to '{converted_folder}'")
converted_count = process_images_recursively(image_folder, converted_folder)
print(f"📸 {converted_count} images processed to JPG format.")

# Stop early if no images were processed
if converted_count == 0:
    print("\n🚨 No images could be processed. Please check your input files.")
    exit()

# ✅ Load Model & Feature Extractor
print("\n🧠 Loading ViT model...")
model_name = "google/vit-base-patch16-224-in21k"
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ✅ Function to Extract Image Embeddings
def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        print(f"✅ Successfully processed: {os.path.basename(image_path)}")
        return embedding.numpy()
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return None

# ✅ Generate Embeddings
image_files = [os.path.join(converted_folder, f) for f in os.listdir(converted_folder) if f.endswith(".jpg")]

# 🔹 Print number of images found
print(f"\n🖼️ Found {len(image_files)} images in {converted_folder}")

if len(image_files) == 0:
    print("🚨 No images found! Please check your input folder and try again.")
    exit()

print("\n🔄 Generating embeddings...")
embeddings = []
for img in image_files:
    emb = get_image_embedding(img)
    if emb is not None:
        embeddings.append(emb)

# ✅ Check if any images failed
if not embeddings:
    print("🚨 No valid embeddings! Check your images.")
    exit()

# 🔹 Print how many embeddings were created
print(f"\n🧩 Successfully created embeddings for {len(embeddings)} images.")

# ✅ Save FAISS Index
print("\n💾 Creating and saving FAISS index...")
embeddings_array = np.array(embeddings)
d = embeddings_array.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)
faiss.write_index(index, "cloth_faiss.index")

print(f"🎉 FAISS index saved as 'cloth_faiss.index' with {len(embeddings)} images!")