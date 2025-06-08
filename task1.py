import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Predict Restaurant Ratings")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset .csv')
    df_clean = df.dropna()
    categorical_cols = df_clean.select_dtypes(include='object').columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean.loc[:, col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le
    return df_clean, le_dict, categorical_cols

df_clean, le_dict, categorical_cols = load_data()

# Split data
X = df_clean.drop('Aggregate rating', axis=1)
y = df_clean['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Restaurant Features")

def user_input_features():
    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = list(le_dict[col].classes_)
            input_data[col] = st.sidebar.selectbox(f"{col}", options)
        else:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            input_data[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
    # Encode categorical
    for col in categorical_cols:
        input_data[col] = le_dict[col].transform([input_data[col]])[0]
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Prediction
if st.button("Predict Rating"):
    prediction = reg.predict(input_df)[0]
    st.success(f"Predicted Aggregate Rating: {prediction:.2f}")

# Show feature importance
if st.checkbox("Show Most Influential Features"):
    feature_importance = reg.coef_
    features = X.columns
    indices = np.argsort(np.abs(feature_importance))[::-1]
    st.write("Top 5 most influential features:")
    for i in range(5):
        st.write(f"{features[indices[i]]}: {feature_importance[indices[i]]:.4f}")