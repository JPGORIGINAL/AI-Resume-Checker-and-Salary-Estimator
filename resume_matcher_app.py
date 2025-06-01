import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Configuration ---
CLEANED_DATA_PATH = 'cleaned_resume_data.pkl'
EMBEDDINGS_PATH = 'resume_embeddings.npy'
MODEL_NAME = 'all-MiniLM-L6-v2' # Must be the same model used for generating existing embeddings

# --- Load Data and Model ---
print("Loading necessary components...")
try:
    df = pd.read_pickle(CLEANED_DATA_PATH)
    resume_embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)
    print("Components loaded successfully.")
except FileNotFoundError:
    print(f"Error: Required files '{CLEANED_DATA_PATH}' or '{EMBEDDINGS_PATH}' not found.")
    print("Please ensure 'data_loader.py' and 'embedding_generator.py' were run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# Get a list of all unique categories for user input validation
available_categories = df['Category'].unique().tolist()
available_categories.sort() # Sort for better display

def get_user_input():
    """Gets user's resume text and desired job category."""
    print("\n--- Enter Your Resume Details ---")
    user_resume_text = input("Please paste your resume text here (or type 'exit' to quit):\n")
    if user_resume_text.lower() == 'exit':
        return None, None

    print("\nAvailable Job Categories:")
    for i, category in enumerate(available_categories):
        print(f"{i+1}. {category}")

    while True:
        try:
            category_input = input("Enter the number corresponding to your desired job category, or type the category name (e.g., 'Accountant' or '1'): ").strip()

            # Try to convert to int if it's a number
            if category_input.isdigit():
                category_index = int(category_input) - 1
                if 0 <= category_index < len(available_categories):
                    user_job_category = available_categories[category_index]
                    break
                else:
                    print("Invalid number. Please enter a valid number from the list.")
            # If not a number, treat as a string category name
            else:
                # Normalize input category to match dataset for comparison
                normalized_input_category = category_input.strip().lower()
                found_category = None
                for cat in available_categories:
                    if cat.lower() == normalized_input_category:
                        found_category = cat
                        break

                if found_category:
                    user_job_category = found_category
                    break
                else:
                    print(f"Category '{category_input}' not found. Please choose from the available categories.")
        except ValueError:
            print("Invalid input. Please enter a number or the category name.")

    return user_resume_text, user_job_category

def find_most_similar_resumes(user_resume_text, user_job_category, df, resume_embeddings, model, top_n=5):
    """
    Filters resumes by category, calculates similarity with user's resume,
    and returns the top N most similar.
    """
    print(f"\nSearching for resumes in the '{user_job_category}' category...")

    # 1. Filter dataset by category
    category_df = df[df['Category'] == user_job_category].copy()
    if category_df.empty:
        print(f"No resumes found for the category '{user_job_category}'.")
        return []

    # Get indices of filtered resumes from the original DataFrame
    # This is crucial to select correct embeddings from the full array
    filtered_indices = category_df.index.tolist()
    filtered_embeddings = resume_embeddings[filtered_indices]

    print(f"Found {len(category_df)} resumes in '{user_job_category}' category.")

    # 2. Generate embedding for user's resume
    print("Generating embedding for your resume...")
    user_resume_embedding = model.encode([user_resume_text], normalize_embeddings=True)[0]

    # 3. Calculate cosine similarity
    # Since embeddings are normalized, dot product is equivalent to cosine similarity
    similarities = cosine_similarity([user_resume_embedding], filtered_embeddings)[0]

    # 4. Get top N similar resumes
    # Sort indices by similarity in descending order
    top_similar_indices_in_filtered = similarities.argsort()[::-1]

    results = []
    for i in top_similar_indices_in_filtered[:top_n]:
        original_df_index = filtered_indices[i] # Map back to original DataFrame index
        similarity_score = similarities[i]
        # Retrieve category (should be the target category) and the actual text from the original DF
        category = df.loc[original_df_index, 'Category']
        text_snippet = df.loc[original_df_index, 'Text'][:200] + "..." # Get a snippet
        results.append({
            "category": category,
            "similarity_score": similarity_score,
            "text_snippet": text_snippet,
            "original_index": original_df_index # Useful for debugging/reference
        })

    return results

# --- Main Application Loop ---
if __name__ == "__main__":
    print("Resume Comparison System is ready!")
    print("This system will compare your resume to similar resumes in a chosen job category.")

    while True:
        user_resume, user_category = get_user_input()

        if user_resume is None: # User typed 'exit'
            print("Exiting application. Goodbye!")
            break

        if user_resume and user_category:
            top_matches = find_most_similar_resumes(user_resume, user_category, df, resume_embeddings, model, top_n=5)

            if top_matches:
                print(f"\n--- Top 5 Most Similar Resumes in '{user_category}' Category ---")
                for i, match in enumerate(top_matches):
                    print(f"\n{i+1}. Similarity Score: {match['similarity_score']:.4f}")
                    print(f"   Category: {match['category']}")
                    print(f"   Snippet: {match['text_snippet']}")
            else:
                print(f"\nCould not find similar resumes for your input in category '{user_category}'.")