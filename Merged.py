import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Restaurant Data Analysis App")
DATASET_PATH = 'Dataset .csv'

# --- Global Data Loading and Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'Dataset .csv' is in the same directory as this app.")
        return None, None, None, None, None

    df = pd.read_csv(file_path)
    df_clean = df.dropna()

    # Preprocessing for prediction model
    df_encoded = df_clean.copy()
    categorical_cols = df_clean.select_dtypes(include='object').columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded.loc[:, col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le

    # Return original (cleaned for recommendations), encoded, and label encoders
    return df, df_clean, df_encoded, le_dict, categorical_cols

df_original, df_cleaned_for_recommender, df_encoded_for_prediction, le_dict, categorical_cols = load_and_preprocess_data(DATASET_PATH)

if df_encoded_for_prediction is None:
    st.stop() # Stop the app if data loading failed

# --- Model Training (Cached) ---
@st.cache_data
def train_prediction_model(df_processed):
    X = df_processed.drop('Aggregate rating', axis=1)
    y = df_processed['Aggregate rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression for direct rating prediction
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Random Forest for classification (rounded ratings)
    y_train_class = y_train.round().astype(int)
    y_test_class = y_test.round().astype(int)
    clf_model = RandomForestClassifier(random_state=42)
    clf_model.fit(X_train, y_train_class)

    return reg_model, clf_model, X, X_train, X_test, y_train_class, y_test_class

reg_model, clf_model, X_features, X_train_global, X_test_global, y_train_class_global, y_test_class_global = train_prediction_model(df_encoded_for_prediction)

# --- Navigation Sidebar ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a feature",
    ["Predict Restaurant Ratings", "Restaurant Recommendation System", "Model Evaluation", "Geographical Analysis"]
)

# --- Feature 1: Predict Restaurant Ratings (from task1.py) ---
def predict_restaurant_ratings_app():
    st.header("Predict Restaurant Ratings")
    st.markdown("Use the sidebar on the left to input features and predict the aggregate rating of a restaurant.")

    def user_input_features():
        input_data = {}
        for col in X_features.columns:
            if col in categorical_cols:
                options = list(le_dict[col].classes_)
                input_data[col] = st.sidebar.selectbox(f"Select {col}", options, key=f"pred_input_{col}")
            else:
                min_val = float(X_features[col].min())
                max_val = float(X_features[col].max())
                mean_val = float(X_features[col].mean())
                input_data[col] = st.sidebar.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val, key=f"pred_input_{col}")
        
        # Encode categorical features for prediction
        for col in categorical_cols:
            if col in input_data and input_data[col] is not None:
                # Ensure the value is in a list for transform
                try:
                    input_data[col] = le_dict[col].transform([str(input_data[col])])[0]
                except ValueError as e:
                    st.warning(f"Could not transform '{input_data[col]}' for column '{col}'. Please check input. Error: {e}")
                    # Assign a default or handle appropriately, e.g., using the mode if the value is not found
                    # For simplicity, we'll just show the warning here.
                    input_data[col] = -1 # Indicate an error in transformation

        return pd.DataFrame([input_data])

    st.sidebar.header("Input Restaurant Features")
    input_df = user_input_features()

    if st.sidebar.button("Predict Rating", key="predict_rating_button"):
        if -1 in input_df.values: # Check if any transformation failed
            st.error("Cannot predict due to invalid input for categorical features. Please select valid options.")
        else:
            try:
                prediction = reg_model.predict(input_df)[0]
                st.success(f"Predicted Aggregate Rating: {prediction:.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    if st.checkbox("Show Most Influential Features (Linear Regression)", key="show_importance_pred"):
        if reg_model is not None and X_features is not None:
            feature_importance = reg_model.coef_
            features = X_features.columns
            indices = np.argsort(np.abs(feature_importance))[::-1]
            st.write("Top 5 most influential features:")
            for i in range(5):
                st.write(f"- **{features[indices[i]]}**: {feature_importance[indices[i]]:.4f}")
        else:
            st.info("Model or features not available for importance display.")

# --- Feature 2: Restaurant Recommendation System (from task2.py) ---
def restaurant_recommendation_system_app():
    st.header("Restaurant Recommendation System")
    st.markdown("Adjust your preferences in the sidebar to find the perfect restaurant.")

    # Sidebar for user preferences
    st.sidebar.header("Set Your Preferences")

    # Cuisine options
    cuisine_options = sorted(set(sum([str(c).split(', ') for c in df_cleaned_for_recommender['Cuisines'].unique()], [])))
    selected_cuisine = st.sidebar.selectbox("Cuisine Preference", options=["Any"] + cuisine_options, key="rec_cuisine")

    # Price range
    min_price_df = int(df_cleaned_for_recommender['Average Cost for two'].min())
    max_price_df = int(df_cleaned_for_recommender['Average Cost for two'].max())
    price_range = st.sidebar.slider("Price Range (for two)", min_price_df, max_price_df, (min_price_df, max_price_df), key="rec_price")

    # Location options
    location_options = sorted(df_cleaned_for_recommender['City'].unique())
    selected_location = st.sidebar.selectbox("City", options=["Any"] + location_options, key="rec_city")

    # Minimum rating
    min_rating_df = float(df_cleaned_for_recommender['Aggregate rating'].min())
    max_rating_df = float(df_cleaned_for_recommender['Aggregate rating'].max())
    selected_rating = st.sidebar.slider("Minimum Rating", min_rating_df, max_rating_df, min_rating_df, key="rec_min_rating")

    # Minimum votes
    min_votes_df = int(df_cleaned_for_recommender['Votes'].min())
    max_votes_df = int(df_cleaned_for_recommender['Votes'].max())
    selected_votes = st.sidebar.slider("Minimum Votes", min_votes_df, max_votes_df, min_votes_df, key="rec_min_votes")

    # Recommendation function
    def recommend_restaurants(df, cuisine=None, price_min=None, price_max=None, location=None, min_rating=None, min_votes=None):
        filtered = df.copy()
        if cuisine and cuisine != "Any":
            filtered = filtered[filtered['Cuisines'].str.contains(cuisine, case=False, na=False)]
        if price_min is not None:
            filtered = filtered[filtered['Average Cost for two'] >= price_min]
        if price_max is not None:
            filtered = filtered[filtered['Average Cost for two'] <= price_max]
        if location and location != "Any":
            filtered = filtered[filtered['City'].str.contains(location, case=False, na=False)]
        if min_rating is not None:
            filtered = filtered[filtered['Aggregate rating'] >= min_rating]
        if min_votes is not None:
            filtered = filtered[filtered['Votes'] >= min_votes]
        filtered = filtered.sort_values(['Aggregate rating', 'Votes'], ascending=[False, False])
        return filtered.head(10)

    # Button to get recommendations
    if st.sidebar.button("Get Recommendations", key="get_rec_button"):
        results = recommend_restaurants(
            df_cleaned_for_recommender,
            cuisine=selected_cuisine,
            price_min=price_range[0],
            price_max=price_range[1],
            location=selected_location,
            min_rating=selected_rating,
            min_votes=selected_votes
        )
        if not results.empty:
            st.subheader("Top Restaurant Recommendations")
            st.dataframe(results[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'City', 'Aggregate rating', 'Votes']])
        else:
            st.warning("No restaurants found matching your criteria. Try relaxing your filters.")

# --- Feature 3: Model Evaluation (from task3.py) ---
def model_evaluation_app():
    st.header("Restaurant Rating Prediction Model Evaluation")

    st.subheader("Model Training and Evaluation Metrics (Random Forest Classifier)")

    if clf_model is None or X_train_global is None or X_test_global is None or y_train_class_global is None or y_test_class_global is None:
        st.error("Model or data not available for evaluation. Please ensure data loading and training were successful.")
        return

    st.write("Training set shape:", X_train_global.shape)
    st.write("Testing set shape:", X_test_global.shape)

    y_pred = clf_model.predict(X_test_global)

    accuracy = accuracy_score(y_test_class_global, y_pred)
    precision = precision_score(y_test_class_global, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_class_global, y_pred, average='weighted', zero_division=0)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    st.subheader("Classification Report:")
    st.code(classification_report(y_test_class_global, y_pred, zero_division=0))

    st.subheader("Cuisine-wise Accuracy Analysis")

    # Re-align original cuisine names for analysis
    X_test_with_cuisine = X_test_global.copy()
    # Need to map the encoded 'Cuisines' back to original names for reporting.
    # This assumes 'Cuisines' is one of the categorical columns encoded.
    # If original DF row indices match X_test_global, then we can use original 'Cuisines' column.
    
    # Check if 'Cuisines' is in df_original, and align using original index
    if 'Cuisines' in df_original.columns:
        # Get original cuisines for the test set
        original_test_cuisines = df_original.loc[X_test_global.index, 'Cuisines']
        X_test_with_cuisine['Cuisines'] = original_test_cuisines
        X_test_with_cuisine['True_Rating'] = y_test_class_global
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
        st.warning("Cannot perform cuisine-wise analysis as 'Cuisines' column is not found in the original dataset for mapping.")

# --- Feature 4: Geographical Analysis (from task4.py) ---
def geographical_analysis_app():
    st.header("Restaurant Geographical Analysis")
    st.markdown("---")

    if df_cleaned_for_recommender is None: # Using df_cleaned_for_recommender as it's already dropped NaNs relevant for geo analysis
        st.error("Data not loaded for geographical analysis. Please check dataset path.")
        return

    st.header("1. Geographical Distribution of Restaurants")
    st.markdown("This interactive map shows the distribution of restaurants based on their latitude and longitude. Zoom in to see individual restaurants and click on markers for details.")

    # Center the map around the mean latitude and longitude of the cleaned data
    mean_latitude = df_cleaned_for_recommender['Latitude'].mean()
    mean_longitude = df_cleaned_for_recommender['Longitude'].mean()

    m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=6, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in df_cleaned_for_recommender.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            popup_html = f"""
            <b>{row['Restaurant Name']}</b><br>
            City: {row['City']}<br>
            Locality: {row['Locality']}<br>
            Cuisines: {row['Cuisines']}<br>
            Rating: {row['Aggregate rating']}<br>
            Price Range: {'$' * int(row['Price range'])} ({row['Price range']}/4)
            """
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row['Restaurant Name']
            ).add_to(marker_cluster)

    try:
        from streamlit_folium import st_folium
        st_folium(m, width=900, height=600)
    except ImportError:
        st.warning("`streamlit_folium` not found. Displaying map as static HTML. For interactive maps, run: `pip install streamlit-folium`")
        map_html = m._repr_html_()
        st.components.v1.html(map_html, width=900, height=600, scrolling=True)

    st.subheader("Geographical Insights:")
    st.write("- **Clusters:** Observe dense clusters of markers, indicating major urban centers or popular dining districts.")
    st.write("- **Spread:** The overall spread of points illustrates the geographical coverage of your dataset.")
    st.write("- **Outliers:** Isolated markers might represent restaurants in less common locations or potentially data anomalies.")

    st.markdown("---")
    st.header("2. Restaurant Concentration Analysis")
    st.markdown("This section analyzes how restaurants are distributed across different cities and localities.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Concentration by City")
        restaurants_by_city = df_cleaned_for_recommender['City'].value_counts().reset_index()
        restaurants_by_city.columns = ['City', 'Number of Restaurants']
        st.dataframe(restaurants_by_city.head(10))
        st.markdown("""
        **Insights (City):**
        - Cities with very high counts are major restaurant markets.
        - This helps identify key urban centers driving the dataset's volume.
        """)

    with col2:
        st.subheader("Concentration by Locality")
        restaurants_by_locality = df_cleaned_for_recommender['Locality'].value_counts().reset_index()
        restaurants_by_locality.columns = ['Locality', 'Number of Restaurants']
        st.dataframe(restaurants_by_locality.head(10))
        st.markdown("""
        **Insights (Locality):**
        - Localities with high counts are specific dining hubs or commercial areas within cities.
        - Provides a granular view of restaurant density, useful for localized market analysis.
        """)

    st.markdown("---")
    st.header("3. Statistics by City and Locality")
    st.markdown("Detailed statistics on average ratings, price ranges, and popular cuisines for different areas.")

    # --- Average Ratings by City and Locality ---
    st.subheader("Average Restaurant Ratings")
    col3, col4 = st.columns(2)
    with col3:
        st.write("By City:")
        avg_rating_by_city = df_cleaned_for_recommender.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
        avg_rating_by_city.columns = ['City', 'Average Rating']
        st.dataframe(avg_rating_by_city.head(10))
        st.markdown("""
        **Insights:**
        - High average ratings suggest a strong culinary scene or high customer satisfaction.
        - Helps identify cities/localities with a good reputation for food quality.
        """)
    with col4:
        st.write("By Locality:")
        avg_rating_by_locality = df_cleaned_for_recommender.groupby('Locality')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
        avg_rating_by_locality.columns = ['Locality', 'Average Rating']
        st.dataframe(avg_rating_by_locality.head(10))

    # --- Average Price Ranges by City and Locality ---
    st.subheader("Average Price Ranges (1=cheap, 4=expensive)")
    col5, col6 = st.columns(2)
    with col5:
        st.write("By City:")
        avg_price_by_city = df_cleaned_for_recommender.groupby('City')['Price range'].mean().sort_values(ascending=False).reset_index()
        avg_price_by_city.columns = ['City', 'Average Price Range']
        st.dataframe(avg_price_by_city.head(10))
        st.markdown("""
        **Insights:**
        - Higher average price ranges (closer to 4) indicate a prevalence of more expensive dining.
        - Lower average price ranges suggest an abundance of budget-friendly eateries.
        """)
    with col6:
        st.write("By Locality:")
        avg_price_by_locality = df_cleaned_for_recommender.groupby('Locality')['Price range'].mean().sort_values(ascending=False).reset_index()
        avg_price_by_locality.columns = ['Locality', 'Average Price Range']
        st.dataframe(avg_price_by_locality.head(10))

    # --- Analyze Cuisines by City and Locality ---
    st.subheader("Popular Cuisines and Culinary Diversity")

    # Explode cuisines for city-wise analysis
    df_cuisines_exploded = df_cleaned_for_recommender.assign(Cuisine=df_cleaned_for_recommender['Cuisines'].str.split(', ')).explode('Cuisine')

    col7, col8 = st.columns(2)
    with col7:
        st.write("Top 5 Cuisines in Top 3 Cities:")
        top_cities_for_cuisine = df_cleaned_for_recommender['City'].value_counts().head(3).index.tolist()
        for city in top_cities_for_cuisine:
            st.write(f"**{city}:**")
            city_cuisines = df_cuisines_exploded[df_cuisines_exploded['City'] == city].groupby('Cuisine').size().reset_index(name='Count')
            city_cuisines = city_cuisines.sort_values(by='Count', ascending=False).head(5)
            st.dataframe(city_cuisines.set_index('Cuisine'))
        st.markdown("""
        **Insights:**
        - Reveals dominant food preferences in specific cities.
        - Vital for restauranteurs planning new ventures or tailoring services.
        """)
    with col8:
        st.write("Top 5 Cuisines in Top 3 Localities:")
        top_localities_for_cuisine = df_cleaned_for_recommender['Locality'].value_counts().head(3).index.tolist()
        for locality in top_localities_for_cuisine:
            st.write(f"**{locality}:**")
            locality_cuisines = df_cuisines_exploded[df_cuisines_exploded['Locality'] == locality].groupby('Cuisine').size().reset_index(name='Count')
            locality_cuisines = locality_cuisines.sort_values(by='Count', ascending=False).head(5)
            st.dataframe(locality_cuisines.set_index('Cuisine'))

    col9, col10 = st.columns(2)
    with col9:
        st.write("Number of Unique Cuisines by City:")
        unique_cuisines_city = df_cuisines_exploded.groupby('City')['Cuisine'].nunique().sort_values(ascending=False).reset_index()
        unique_cuisines_city.columns = ['City', 'Unique Cuisines Count']
        st.dataframe(unique_cuisines_city.head(10))
        st.markdown("""
        **Insights:**
        - Higher count signifies greater culinary diversity.
        - Indicates mature and diverse food markets.
        """)
    with col10:
        st.write("Number of Unique Cuisines by Locality:")
        unique_cuisines_locality = df_cuisines_exploded.groupby('Locality')['Cuisine'].nunique().sort_values(ascending=False).reset_index()
        unique_cuisines_locality.columns = ['Locality', 'Unique Cuisines Count']
        st.dataframe(unique_cuisines_locality.head(10))


# --- Main App Logic ---
if app_mode == "Predict Restaurant Ratings":
    predict_restaurant_ratings_app()
elif app_mode == "Restaurant Recommendation System":
    restaurant_recommendation_system_app()
elif app_mode == "Model Evaluation":
    model_evaluation_app()
elif app_mode == "Geographical Analysis":
    geographical_analysis_app()