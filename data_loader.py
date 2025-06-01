import pandas as pd
import spacy
import os # Import os module to check for file existence

# --- Configuration ---
# Since your CSV is in the same folder as this script, you can just use its name.
DATASET_PATH = 'resume_dataset.csv'

# --- Load the Dataset ---
df = pd.DataFrame() # Initialize an empty DataFrame
if not os.path.exists(DATASET_PATH):
    print(f"Error: The file was not found at {DATASET_PATH}")
    print("Please ensure 'resume_dataset.csv' is in the same directory as this script (ResumeMatcherAI folder).")
else:
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded successfully from: {DATASET_PATH}")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        df.info()
        print("\nNumber of unique categories and their counts:")
        print(df['Category'].value_counts())
        print("\nExample of 'Text' column content (first entry):")
        if not df.empty and 'Text' in df.columns:
            # Print first 500 characters of the first text
            print(df['Text'].iloc[0][:500])
        else:
            print("Dataset is empty or 'Text' column not found.")
    except KeyError:
        print("Error: 'Category' or 'Text' column not found in the CSV. Please ensure your columns are named exactly 'Category' and 'Text' (case-sensitive).")
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")

# --- Basic Data Cleaning (only if DataFrame loaded successfully) ---
if not df.empty and 'Category' in df.columns and 'Text' in df.columns:
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Drop rows where 'Text' or 'Category' is missing (if any)
    initial_rows = len(df)
    df.dropna(subset=['Text', 'Category'], inplace=True)
    if len(df) < initial_rows:
        print(f"\nDropped {initial_rows - len(df)} rows with missing 'Text' or 'Category' values.")

    print(f"\nDataset shape after initial loading and cleaning: {df.shape}")

    # --- Save the cleaned DataFrame for future steps ---
    # This is important so you don't reload and clean every time
    df.to_pickle('cleaned_resume_data.pkl')
    print("\nCleaned DataFrame saved to 'cleaned_resume_data.pkl'")
elif not df.empty:
    print("Could not perform cleaning: 'Category' or 'Text' column missing after loading.")

# --- Initialize spaCy for NLP (Optional but good to have ready) ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("\nspaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you've run 'python -m spacy download en_core_web_sm' in your terminal.")