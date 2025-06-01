import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import string
import joblib # For loading the salary prediction model and transformers

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Career & Resume Tools")

# --- Configuration Paths ---
CLEANED_DATA_PATH = 'cleaned_resume_data.pkl'
EMBEDDINGS_PATH = 'resume_embeddings.npy'
RESUME_MODEL_NAME = 'all-MiniLM-L6-v2' # Model for resume embeddings

SALARY_MODEL_DIR = 'salary_model_assets'
SALARY_REGRESSOR_PATH = os.path.join(SALARY_MODEL_DIR, 'salary_regressor_model.pkl')
# UPDATED: Corrected encoder and scaler paths to match what train_salary_model.py saves
JOB_ROLE_ENCODER_PATH = os.path.join(SALARY_MODEL_DIR, 'job_title_encoder.pkl') # Was job_role_encoder.pkl
RATING_SCALER_PATH = os.path.join(SALARY_MODEL_DIR, 'rating_scaler.pkl')       # Was years_experience_scaler.pkl


# --- Global Pre-processing for Resume Text ---
def preprocess_resume_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespaces
    return text

# --- Caching for Efficiency (Resume Matcher Components) ---
@st.cache_resource
def load_resume_resources():
    """Loads the DataFrame, embeddings, and SentenceTransformer model for resume matching."""
    try:
        df = pd.read_pickle(CLEANED_DATA_PATH)
        resume_embeddings = np.load(EMBEDDINGS_PATH)
        model = SentenceTransformer(RESUME_MODEL_NAME)
        return df, resume_embeddings, model
    except FileNotFoundError:
        st.error(f"Error: Required resume files not found. Ensure '{CLEANED_DATA_PATH}' and '{EMBEDDINGS_PATH}' exist.")
        st.error("Please ensure 'data_loader.py' and 'embedding_generator.py' were run successfully.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during loading resume resources: {e}")
        st.stop()

# --- Caching for Efficiency (Salary Estimator Components) ---
@st.cache_resource
def load_salary_resources():
    """Loads the trained salary model and pre-processing tools."""
    try:
        regressor = joblib.load(SALARY_REGRESSOR_PATH)
        job_role_enc = joblib.load(JOB_ROLE_ENCODER_PATH)
        rating_scaler = joblib.load(RATING_SCALER_PATH) # Changed to rating_scaler
        return regressor, job_role_enc, rating_scaler
    except FileNotFoundError:
        st.error(f"Error: Required salary model files not found in '{SALARY_MODEL_DIR}'.")
        st.error("Please ensure 'train_salary_model.py' was run successfully and created these files.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during loading salary resources: {e}")
        st.stop()

# Load all resources at startup
resume_df, resume_embeddings, resume_model = load_resume_resources()
salary_regressor, job_role_encoder, rating_scaler = load_salary_resources() # Changed variable name


# --- Resume Matcher Functions ---
available_categories = resume_df['Category'].unique().tolist()
available_categories.sort()

def find_most_similar_resumes(user_resume_text_raw, user_job_category, df, embeddings, model, top_n=5):
    st.write(f"Searching for resumes in the '{user_job_category}' category...")
    category_df = df[df['Category'] == user_job_category].copy()
    if category_df.empty:
        st.warning(f"No resumes found for the category '{user_job_category}'. Please choose another category.")
        return []

    filtered_indices = category_df.index.tolist()
    filtered_embeddings = embeddings[filtered_indices]

    st.info(f"Found {len(category_df)} resumes in '{user_job_category}' category.")

    processed_user_resume_text = preprocess_resume_text(user_resume_text_raw)
    st.write(f"Pre-processed user resume snippet: {processed_user_resume_text[:200]}...")

    st.write("Generating embedding for your resume...")
    user_resume_embedding = model.encode([processed_user_resume_text], normalize_embeddings=True)[0]

    similarities = cosine_similarity([user_resume_embedding], filtered_embeddings)[0]
    top_similar_indices_in_filtered = similarities.argsort()[::-1]

    results = []
    for i in top_similar_indices_in_filtered[:top_n]:
        original_df_index = filtered_indices[i]
        similarity_score = similarities[i]
        category = df.loc[original_df_index, 'Category']
        text_snippet = df.loc[original_df_index, 'Text'][:500] + "..."
        results.append({
            "category": category,
            "similarity_score": similarity_score,
            "text_snippet": text_snippet,
            "original_index": original_df_index
        })
    return results

# --- Salary Estimator Function ---
# UPDATED: Function now takes 'rating_input' instead of 'years_experience' and 'education_level'
def estimate_salary(job_role_input, rating_input):
    try:
        # Pre-process inputs using the *saved* encoders and scaler
        
        # Job Role Encoding: Handle unseen labels
        # Find the exact casing from the encoder's classes_ if available, otherwise fallback
        try:
            # Try to find the exact match from the encoder's classes (case-sensitive)
            if job_role_input in job_role_encoder.classes_:
                encoded_job_role = job_role_encoder.transform([job_role_input])[0]
            else:
                # Try finding a case-insensitive match
                matched_class = None
                for cls in job_role_encoder.classes_:
                    if job_role_input.lower() == cls.lower():
                        matched_class = cls
                        break
                if matched_class:
                    encoded_job_role = job_role_encoder.transform([matched_class])[0]
                else:
                    # If still not found, handle as unseen (e.g., assign a default/median job role code)
                    # For a robust solution, you might consider OneHotEncoder with handle_unknown='ignore'
                    # Or a more sophisticated fuzzy matching.
                    # For now, if unseen, we'll give a warning and use the first job role found in the encoder.
                    st.warning(f"Job role '{job_role_input}' not found in training data. Using a default encoding, which may affect accuracy.")
                    encoded_job_role = job_role_encoder.transform([job_role_encoder.classes_[0]])[0]

        except Exception as e:
            st.error(f"Error encoding job role: {e}. Please ensure the job role is valid.")
            return None


        # Rating Scaling
        scaled_rating = rating_scaler.transform([[rating_input]])[0][0]

        # Create DataFrame for prediction (must match training features order and names)
        # Features were: ['Rating_Scaled', 'Job Title_Encoded']
        input_data = pd.DataFrame([[scaled_rating, encoded_job_role]],
                                  columns=['Rating_Scaled', 'Job Title_Encoded'])
        
        predicted_salary = salary_regressor.predict(input_data)[0]
        return predicted_salary
    except Exception as e:
        st.error(f"Error during salary estimation: {e}")
        st.error("Please ensure your inputs are valid and the model assets are correctly loaded.")
        return None

# --- Streamlit UI Layout ---
st.title("👨‍💼 AI Career & Resume Tools 📈")

st.markdown("""
This integrated system helps you with two key career tasks:
1.  **Resume Matcher:** Find similar resumes in a specific job category based on your input.
2.  **Career Income Estimator:** Get an estimated annual income based on job role and company rating.
""")

# Use tabs for different functionalities
tab1, tab2 = st.tabs(["Resume Matcher", "Career Income Estimator"])

with tab1:
    st.header("1. Resume Matcher")
    st.subheader("Find Similar Resumes in Your Desired Category")
    
    user_resume_text_input = st.text_area(
        "Paste your resume text below:",
        placeholder="e.g., 'Experienced financial analyst skilled in budgeting and forecasting...'",
        height=200,
        key="resume_input"
    )

    selected_category = st.selectbox(
        "Choose the job category you are interested in:",
        options=["-- Select a Category --"] + available_categories,
        key="category_select"
    )

    st.markdown("""
    ---
    **Note on Resume Pre-processing:** Your input resume will be automatically pre-processed (converted to lowercase, punctuation and numbers removed, extra spaces cleaned) to ensure optimal comparison with the dataset.
    ---
    """)

    if st.button("Find Similar Resumes", key="find_resume_button"):
        if not user_resume_text_input or selected_category == "-- Select a Category --":
            st.warning("Please enter your resume text and select a job category to proceed.")
        else:
            with st.spinner("Processing your request..."):
                top_matches = find_most_similar_resumes(user_resume_text_input, selected_category, resume_df, resume_embeddings, resume_model, top_n=5)

                if top_matches:
                    st.success("Comparison complete! Here are the top matches:")
                    for i, match in enumerate(top_matches):
                        st.subheader(f"Match {i+1} (Category: {match['category']})")
                        st.metric(label="Similarity Score", value=f"{match['similarity_score']:.4f}")
                        st.write("---")
                        st.text_area(f"Resume Snippet {i+1}:", value=match['text_snippet'], height=150, disabled=True, key=f"snippet_{i}")
                        st.write("---")
                else:
                    st.info("No similar resumes found for your input in the selected category.")

with tab2:
    st.header("2. Career Income Estimator")
    st.subheader("Estimate Your Potential Annual Income")

    # Get unique Job Roles from the trained encoder for dropdowns
    available_job_roles = sorted(job_role_encoder.classes_.tolist())

    job_role_input = st.selectbox(
        "Select Job Role:",
        options=["-- Select Job Role --"] + available_job_roles,
        key="job_role_select"
    )
    
    # UPDATED: Input for Rating instead of Years of Experience
    rating_input = st.slider(
        "Company/Job Rating (e.g., Glassdoor rating):",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
        key="rating_input"
    )
    st.markdown(f"*(Note: Rating refers to the company's Glassdoor rating, which correlated with salary in the training data.)*")


    if st.button("Estimate Income", key="estimate_income_button"):
        if job_role_input == "-- Select Job Role --":
            st.warning("Please select a Job Role.")
        else:
            with st.spinner("Estimating income..."):
                estimated_income = estimate_salary(
                    job_role_input,
                    rating_input
                )
                if estimated_income is not None:
                    st.success(f"Estimated Annual Income: ${estimated_income:,.2f}")
                else:
                    st.error("Could not estimate income. Please check your inputs.")

st.markdown("---")
st.markdown("Developed for AI Bootcamp Project")