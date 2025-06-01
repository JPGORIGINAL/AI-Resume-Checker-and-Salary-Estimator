import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# --- Configuration ---
# Path to the cleaned DataFrame saved in the previous step
CLEANED_DATA_PATH = 'cleaned_resume_data.pkl'
# Model to use for generating embeddings. 'all-MiniLM-L6-v2' is a good balance of size and performance.
# Other options: 'all-mpnet-base-v2' (larger, better performance), 'paraphrase-MiniLM-L3-v2' (smaller)
MODEL_NAME = 'all-MiniLM-L6-v2'
# Path to save the generated embeddings
EMBEDDINGS_PATH = 'resume_embeddings.pkl'

print(f"Starting embedding generation with model: {MODEL_NAME}")

# --- Load Cleaned Data ---
if not os.path.exists(CLEANED_DATA_PATH):
    print(f"Error: Cleaned data not found at {CLEANED_DATA_PATH}.")
    print("Please run 'data_loader.py' first to generate this file.")
    exit() # Exit the script if data is not found

df = pd.read_pickle(CLEANED_DATA_PATH)
print(f"Cleaned data loaded successfully from: {CLEANED_DATA_PATH}. Shape: {df.shape}")

# --- Load Sentence Transformer Model ---
try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"SentenceTransformer model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please check your internet connection or if the model name is correct.")
    exit()

# --- Generate Embeddings ---
print("Generating embeddings for all resumes... This may take a while.")
# The .tolist() is important if you want to store them in a DataFrame column
# If you have a very large dataset and memory becomes an issue, you might process in batches.
# For 13k resumes, this should be fine.
resume_embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
# Convert to a list of numpy arrays if you plan to store them directly in a DataFrame column
# For 'normalize_embeddings=True', cosine similarity is directly dot product
# We will store them as a NumPy array for efficiency
print(f"Embeddings generated. Shape: {resume_embeddings.shape}") # Should be (13389, embedding_dimension)

# --- Save Embeddings ---
# Saving embeddings as a NumPy array directly is more efficient for this purpose
np.save('resume_embeddings.npy', resume_embeddings)
print(f"Resume embeddings saved to 'resume_embeddings.npy'.")

# Optionally, you can also add embeddings to the DataFrame and save it as PKL,
# but for large datasets, saving separately is better for specific operations.
# df['Embeddings'] = list(resume_embeddings) # Convert to list of arrays to store in DataFrame column
# df.to_pickle(EMBEDDINGS_PATH)
# print(f"Cleaned DataFrame with embeddings saved to '{EMBEDDINGS_PATH}'.")

print("\nEmbedding generation complete!")