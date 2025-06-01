import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import re # For cleaning salary estimates

# --- Configuration ---
DATASET_PATH = 'glassdoor_salaries.csv' # IMPORTANT: Ensure this matches your renamed file!
MODEL_OUTPUT_DIR = 'salary_model_assets'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
print(f"Loading data from {DATASET_PATH}...")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Data loaded successfully. Head of the dataset:")
    print(df.head())
    print("\nData Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure '{DATASET_PATH}' is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Data Pre-processing ---
print("\nStarting data pre-processing...")

# Handle missing values (if any) before specific column operations
# Drop rows where essential columns are missing
df.dropna(subset=['Job Title', 'Rating', 'Salary Estimate'], inplace=True)
print(f"Dataset shape after dropping NaNs in essential columns: {df.shape}")

# --- NEW: Parse 'Salary Estimate' into a numerical 'Salary' column ---
# Example: "$80K - $120K (Glassdoor)" -> 80000, 120000 -> 100000
# Example: "$40-$60 per hour" -> convert to annual
# We'll aim for a simple average of the range, converted to annual salary.

def parse_salary_estimate(salary_str):
    salary_str = salary_str.lower().replace('$', '').replace('k', '000').replace(',', '').strip()

    # Handle "per hour" estimates by multiplying by typical working hours (2080 hours/year)
    if 'per hour' in salary_str:
        numbers = re.findall(r'\d+', salary_str)
        if len(numbers) >= 1:
            hourly_rate = (int(numbers[0]) + int(numbers[1])) / 2 if len(numbers) >= 2 else int(numbers[0])
            return hourly_rate * 2080 # 40 hours/week * 52 weeks/year
        else:
            return None
    
    # Handle annual ranges
    if '-' in salary_str:
        parts = salary_str.split('-')
        if len(parts) == 2:
            try:
                min_salary = float(parts[0].strip())
                max_salary = float(parts[1].split('(')[0].strip()) # Remove any text like (Glassdoor)
                return (min_salary + max_salary) / 2
            except ValueError:
                return None
    
    # Handle single values if any (though ranges are more common in this dataset)
    try:
        return float(re.sub(r'[^0-9.]', '', salary_str.split('(')[0]))
    except ValueError:
        return None

df['Avg Salary'] = df['Salary Estimate'].apply(parse_salary_estimate)
df.dropna(subset=['Avg Salary'], inplace=True) # Drop rows where salary parsing failed
print("Parsed 'Salary Estimate' into numerical 'Avg Salary'.")
print(f"Dataset shape after parsing salaries: {df.shape}")


# Encode 'Job Title'
job_title_encoder = LabelEncoder()
df['Job Title_Encoded'] = job_title_encoder.fit_transform(df['Job Title'])
print("Encoded 'Job Title'.")

# Scale 'Rating' (which is already numerical)
scaler = StandardScaler()
df['Rating_Scaled'] = scaler.fit_transform(df[['Rating']])
print("Scaled 'Rating'.")

print("Pre-processing complete. Transformed data head (selected columns):")
print(df[['Job Title', 'Rating', 'Salary Estimate', 'Avg Salary', 'Job Title_Encoded', 'Rating_Scaled']].head())


# --- Define Features (X) and Target (y) ---
# Features for the model: using scaled Rating and encoded Job Title
X = df[['Rating_Scaled', 'Job Title_Encoded']]
y = df['Avg Salary'] # Target variable is now the parsed numerical salary

print(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")

# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

# --- Model Training ---
print("\nTraining the RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R2): {r2:.4f}")
print("MAE represents the average absolute difference between predicted and actual salaries.")
print("R-squared indicates how well the model explains the variance in the target variable (closer to 1 is better).")


# --- Save Model and Pre-processing Tools ---
print("\nSaving the trained model and pre-processing tools...")
joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, 'salary_regressor_model.pkl'))
joblib.dump(job_title_encoder, os.path.join(MODEL_OUTPUT_DIR, 'job_title_encoder.pkl'))
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'rating_scaler.pkl'))

print(f"All assets saved to '{MODEL_OUTPUT_DIR}' directory.")
print("\nSalary model training complete!")