import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import os # Import the os module to construct file paths

st.set_page_config(layout="wide")

st.title("Restaurant Rating Prediction App")

# Define the path to your dataset
DATASET_PATH = 'Dataset .csv' # Make sure this matches your file name exactly

# --- Data Loading and Preprocessing ---
st.header("1. Data Loading and Preprocessing")

@st.cache_data
def load_and_preprocess_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path) # Directly read the file from the specified path
        st.write("Original shape:", df.shape)

        df_clean = df.dropna()
        st.write("After dropping missing values:", df_clean.shape)
        st.subheader("First 5 rows of cleaned data:")
        st.dataframe(df_clean.head())

        categorical_cols = df_clean.select_dtypes(include='object').columns
        le = LabelEncoder()
        df_encoded = df_clean.copy() # Create a copy to avoid SettingWithCopyWarning
        for col in categorical_cols:
            df_encoded.loc[:, col] = le.fit_transform(df_clean[col].astype(str))
        
        st.subheader("First 5 rows of encoded data:")
        st.dataframe(df_encoded.head())
        return df, df_encoded
    else:
        st.error(f"Dataset not found at: {file_path}. Please ensure 'Dataset .csv' is in the same directory as the app.")
        return None, None

# Call the function with the internal file path
df_original, df_processed = load_and_preprocess_data(DATASET_PATH)

if df_processed is not None:
    # --- Data Splitting ---
    st.header("2. Data Splitting (Training and Testing Sets)")

    X = df_processed.drop('Aggregate rating', axis=1)
    y = df_processed['Aggregate rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Training set shape:", X_train.shape)
    st.write("Testing set shape:", X_test.shape)

    # --- Model Training ---
    st.header("3. Model Training (Random Forest Classifier)")

    y_train_class = y_train.round().astype(int)
    y_test_class = y_test.round().astype(int)

    clf = RandomForestClassifier(random_state=42)
    with st.spinner("Training the Random Forest Classifier..."):
        clf.fit(X_train, y_train_class)
    st.success("Model training complete!")

    # --- Model Evaluation ---
    st.header("4. Model Evaluation")

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_class, y_pred, average='weighted', zero_division=0)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    st.subheader("Classification Report:")
    st.code(classification_report(y_test_class, y_pred, zero_division=0))

    # --- Cuisine-wise Accuracy Analysis ---
    st.header("5. Cuisine-wise Accuracy Analysis")

    X_test_with_cuisine = X_test.copy()
    
    # Ensure indices align correctly.
    X_test_with_cuisine['Cuisines'] = df_original.loc[X_test.index, 'Cuisines']
    X_test_with_cuisine['True_Rating'] = y_test_class
    X_test_with_cuisine['Predicted_Rating'] = y_pred

    cuisine_groups = X_test_with_cuisine.groupby('Cuisines')
    cuisine_accuracy = {}

    for cuisine, group in cuisine_groups:
        acc = np.mean(group['True_Rating'] == group['Predicted_Rating'])
        cuisine_accuracy[cuisine] = acc

    sorted_acc = sorted(cuisine_accuracy.items(), key=lambda x: x[1], reverse=True)

    st.subheader("Cuisine-wise accuracy (top 5):")
    for cuisine, acc in sorted_acc[:5]:
        st.write(f"- {cuisine}: {acc:.2f}")

    st.subheader("Cuisine-wise accuracy (bottom 5):")
    for cuisine, acc in sorted_acc[-5:]:
        st.write(f"- {cuisine}: {acc:.2f}")

    st.subheader("Analysis of Challenges and Biases:")
    st.write("- Cuisines with fewer samples may have lower accuracy due to insufficient data.")
    st.write("- If certain cuisines always get the same predicted rating, the model may be biased or underfitting.")
    st.write("- Check class distribution per cuisine for imbalance.")
else:
    st.warning("Please ensure 'Dataset .csv' is present in the same directory as this Streamlit app.")