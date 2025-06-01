# AI Career & Resume Tools

## Project Overview

This project is an AI-powered application designed to assist users with two key career-related tasks:
1.  **Resume Matching:** It allows users to input their resume text and find similar resumes from a pre-defined dataset within a specific job category. This helps in assessing resume relevance for a particular role or exploring similar profiles.
2.  **Career Income Estimator:** It provides an estimated annual income based on a specified job role and a company's rating (e.g., Glassdoor rating). This tool offers quick salary insights for various positions.

The application leverages advanced Natural Language Processing (NLP) techniques and machine learning models to provide these insights, all presented through an interactive web interface built with Streamlit.

## Features

* **Semantic Resume Search:** Utilizes Sentence-Transformers to generate embeddings for resumes, enabling highly accurate similarity comparisons beyond keyword matching.
* **Category-Specific Matching:** Allows users to filter resume searches by job categories (e.g., "HR", "Sales", "Data Science").
* **Dynamic Salary Estimation:** Predicts annual income based on `Job Title` and `Company Rating` (as a proxy for company quality/experience correlation) using a trained Regression model.
* **User-Friendly Interface:** Built with Streamlit for easy interaction and visualization of results.

## Technologies Used

* **Python 3.8+**
* **Streamlit**: For creating the interactive web application.
* **pandas**: For data manipulation and analysis.
* **numpy**: For numerical operations, especially with embeddings.
* **scikit-learn**: For machine learning models (RandomForestRegressor, LabelEncoder, StandardScaler) and utility functions (train_test_split, cosine_similarity).
* **sentence-transformers**: For generating high-quality semantic embeddings from text.
* **joblib**: For efficient saving and loading of trained models and pre-processing objects.
* **re (Regular Expressions)** & **string**: For text pre-processing.

## Installation and Setup

Follow these steps to set up and run the project locally:

### Prerequisites

* Python 3.8 or higher installed on your system.

### Steps

1.  **Clone the Repository (if applicable):**
    If this project is hosted on Git, clone it to your local machine.
    ```bash
    git clone <your-repository-url-here> # Replace with your actual repository URL
    cd ResumeMatcherAI
    ```
    If you received the project as a `.zip` file, extract it and navigate into the `ResumeMatcherAI` folder in your terminal.

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv .venv
    ```
    **Activate the virtual environment:**
    * **On Windows (PowerShell):**
        ```bash
        .venv\Scripts\Activate.ps1
        ```
    * **On macOS/Linux (Bash/Zsh):**
        ```bash
        source .venv/bin/activate
        ```
    Your terminal prompt should show `(.venv)` at the beginning, indicating the environment is active.

3.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Data Setup

You need to download and place two datasets in your project directory:

1.  **Resume Dataset (`resume_dataset.csv`):**
    * Ensure you have the `resume_dataset.csv` file. This dataset contains various resume texts categorized by job roles.
    * **Place this file directly in the `ResumeMatcherAI/` project root directory.**

2.  **Glassdoor Salary Dataset (`glassdoor_salaries.csv`):**
    * Download the dataset from Kaggle: [Glassdoor Salary Prediction Dataset](https://www.kaggle.com/datasets/sureshmecad/salary-prediction-dataset-glassdoor)
    * **Rename** the downloaded file to `glassdoor_salaries.csv`.
    * **Place this file directly in the `ResumeMatcherAI/` project root directory.**

## Model Training and Embedding Generation

Before running the main application, you need to generate resume embeddings and train the salary prediction model.

1.  **Generate Resume Embeddings:**
    This script processes `resume_dataset.csv` to create cleaned text and numerical embeddings, saving them as `cleaned_resume_data.pkl` and `resume_embeddings.npy`.
    ```bash
    python embedding_generator.py
    ```

2.  **Train Salary Prediction Model:**
    This script processes `glassdoor_salaries.csv`, trains a machine learning model, and saves the trained model and its pre-processing components (encoders, scalers) into the `salary_model_assets/` directory.
    ```bash
    python train_salary_model.py
    ```

## How to Run the Application

Once all data and models are set up:

1.  **Ensure your virtual environment is active.**
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

## Project Structure

ResumeMatcherAI/

├── app.py                      # Main Streamlit application
├── data_loader.py              # Script to load and initially clean resume data
├── embedding_generator.py      # Script to generate and save resume embeddings
├── train_salary_model.py       # Script to train and save the salary prediction model
├── requirements.txt            # List of Python dependencies
├── README.md                   # This instruction file
├── resume_dataset.csv          # Raw resume dataset
├── glassdoor_salaries.csv      # Raw Glassdoor salary dataset
├── cleaned_resume_data.pkl     # Processed resume data (generated by embedding_generator.py)
├── resume_embeddings.npy       # Numerical embeddings for resumes (generated by embedding_generator.py)
└── salary_model_assets/        # Directory for trained salary model assets
├── salary_regressor_model.pkl  # The trained salary prediction model
├── job_title_encoder.pkl       # LabelEncoder for job titles
└── rating_scaler.pkl           # StandardScaler for company ratings


## Credits and Acknowledgements

* **Resume Dataset:** (If you have a specific source, cite it here. Otherwise, you can state "Internal mock dataset" or similar if it's not from a public source).
* **Glassdoor Salary Dataset:** Dataset sourced from Kaggle: [Salary Prediction Dataset (Glassdoor)](https://www.kaggle.com/datasets/sureshmecad/salary-prediction-dataset-glassdoor)
* **Sentence-Transformers Library:** Hugging Face and their open-source community.
