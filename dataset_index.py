import pandas as pd
from pinecone import Pinecone
import openai
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
import numpy as np

# Load API Keys from .env
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

# Check if Pinecone index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    time.sleep(10)  # Wait for the index to initialize

index = pc.Index(PINECONE_INDEX_NAME)

# Load dataset
file_path = "vasavi_quantities_sheet.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File '{file_path}' not found!")

df = pd.read_excel(file_path)

# Rename columns properly
expected_columns = [
    "S.NO", "STYLE IMAGE", "NaN1", "STYLE NAME", "NaN2", "STYLE NUMBER",
    "DESCRIPTION", "PRICES", "FABRIC DESCRIPTION", "XS", "S", "M", "L", "XL", "TOTAL"
]
if len(df.columns) < len(expected_columns):
    raise ValueError("❌ Excel file format mismatch. Check column names!")

df.columns = expected_columns

# Drop unnecessary columns
df = df.drop(columns=["NaN1", "NaN2", "STYLE IMAGE"]).iloc[1:].reset_index(drop=True)

# Handle missing values
df.fillna("", inplace=True)
df["STYLE NUMBER"] = df["STYLE NUMBER"].astype(str).str.strip()
df["PRICES"] = df["PRICES"].astype(str).str.strip()
df = df[df["STYLE NUMBER"] != ""]  # Remove rows without style number

# Function to get embeddings using text-embedding-3-small
def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding  # Extract vector
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

# Upload data to Pinecone
failed_entries = []
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="🚀 Uploading to Pinecone"):
    try:
        product_id = row["STYLE NUMBER"]
        product_text = f"{row['STYLE NAME']} - {row['DESCRIPTION']} - {row['FABRIC DESCRIPTION']} - {row['PRICES']}"

        embedding = get_embedding(product_text)
        if embedding:
            metadata = {
                "name": row["STYLE NAME"] or "Unknown",
                "description": row["DESCRIPTION"] or "No description",
                "price": row["PRICES"] or "N/A",
                "fabric": row["FABRIC DESCRIPTION"] or "Unknown"
            }
            index.upsert([(product_id, embedding, metadata)])
        else:
            failed_entries.append(product_id)

    except Exception as e:
        print(f"❌ Error processing {row['STYLE NAME']} ({product_id}): {e}")
        failed_entries.append(product_id)

if failed_entries:
    print(f"⚠️ {len(failed_entries)} entries failed to upload: {failed_entries}")

print("✅ Task completed successfully!")
