import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os # For checking file existence

# Ensure these libraries are installed:
# pip install pandas streamlit folium
# (Optional for direct folium rendering in newer Streamlit versions: pip install streamlit-folium)

st.set_page_config(layout="wide")

st.title("Restaurant Geographical Analysis")
st.markdown("---")

# Define the path to your dataset
DATASET_PATH = 'Dataset .csv'

# --- Data Loading and Cleaning ---
st.header("1. Data Loading and Preprocessing")

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        st.success("Dataset loaded successfully.")
        st.write(f"Original shape: {df.shape}")

        # Drop rows with any missing values crucial for geographical analysis
        df_clean = df.dropna(subset=['City', 'Locality', 'Latitude', 'Longitude', 'Restaurant Name',
                                      'Aggregate rating', 'Cuisines', 'Price range'])
        st.write(f"Shape after dropping rows with missing essential values: {df_clean.shape}")
        st.subheader("First 5 rows of cleaned data:")
        st.dataframe(df_clean.head())
        return df_clean
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'Dataset .csv' is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading or cleaning: {e}")
        return None

df_cleaned = load_and_clean_data(DATASET_PATH)

if df_cleaned is not None:
    st.markdown("---")
    st.header("2. Geographical Distribution of Restaurants")
    st.markdown("This interactive map shows the distribution of restaurants based on their latitude and longitude. Zoom in to see individual restaurants and click on markers for details.")

    # Create a base map using folium
    # Center the map around the mean latitude and longitude of the cleaned data
    mean_latitude = df_cleaned['Latitude'].mean()
    mean_longitude = df_cleaned['Longitude'].mean()

    m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=6, tiles='OpenStreetMap')

    # Add a MarkerCluster to group nearby markers
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each restaurant
    for idx, row in df_cleaned.iterrows():
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

    # Display the map in Streamlit
    # Streamlit has built-in support for folium maps via st_folium, but if not installed,
    # embedding HTML is an alternative. For simplicity and broad compatibility,
    # we'll generate the HTML and embed it. If streamlit_folium is installed,
    # st_folium(m, width=900, height=600) is preferred.
    
    # Check if streamlit_folium is available
    try:
        from streamlit_folium import st_folium
        st_folium(m, width=900, height=600)
    except ImportError:
        # Fallback to embedding raw HTML if streamlit_folium is not installed
        st.warning("`streamlit_folium` not found. Displaying map as static HTML. For interactive maps, run: `pip install streamlit-folium`")
        map_html = m._repr_html_()
        st.components.v1.html(map_html, width=900, height=600, scrolling=True)


    st.subheader("Geographical Insights:")
    st.write("- **Clusters:** Observe dense clusters of markers, indicating major urban centers or popular dining districts (e.g., New Delhi NCR, parts of the Philippines or UAE from previous runs).")
    st.write("- **Spread:** The overall spread of points illustrates the geographical coverage of your dataset.")
    st.write("- **Outliers:** Isolated markers might represent restaurants in less common locations or potentially data anomalies.")

    st.markdown("---")
    st.header("3. Restaurant Concentration Analysis")
    st.markdown("This section analyzes how restaurants are distributed across different cities and localities.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Concentration by City")
        restaurants_by_city = df_cleaned['City'].value_counts().reset_index()
        restaurants_by_city.columns = ['City', 'Number of Restaurants']
        st.dataframe(restaurants_by_city.head(10))
        st.markdown("""
        **Insights (City):**
        - Cities with very high counts are major restaurant markets.
        - This helps identify key urban centers driving the dataset's volume.
        """)

    with col2:
        st.subheader("Concentration by Locality")
        restaurants_by_locality = df_cleaned['Locality'].value_counts().reset_index()
        restaurants_by_locality.columns = ['Locality', 'Number of Restaurants']
        st.dataframe(restaurants_by_locality.head(10))
        st.markdown("""
        **Insights (Locality):**
        - Localities with high counts are specific dining hubs or commercial areas within cities.
        - Provides a granular view of restaurant density, useful for localized market analysis.
        """)

    st.markdown("---")
    st.header("4. Statistics by City and Locality")
    st.markdown("Detailed statistics on average ratings, price ranges, and popular cuisines for different areas.")

    # --- Average Ratings by City and Locality ---
    st.subheader("Average Restaurant Ratings")
    col3, col4 = st.columns(2)
    with col3:
        st.write("By City:")
        avg_rating_by_city = df_cleaned.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
        avg_rating_by_city.columns = ['City', 'Average Rating']
        st.dataframe(avg_rating_by_city.head(10))
        st.markdown("""
        **Insights:**
        - High average ratings suggest a strong culinary scene or high customer satisfaction.
        - Helps identify cities/localities with a good reputation for food quality.
        """)
    with col4:
        st.write("By Locality:")
        avg_rating_by_locality = df_cleaned.groupby('Locality')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
        avg_rating_by_locality.columns = ['Locality', 'Average Rating']
        st.dataframe(avg_rating_by_locality.head(10))

    # --- Average Price Ranges by City and Locality ---
    st.subheader("Average Price Ranges (1=cheap, 4=expensive)")
    col5, col6 = st.columns(2)
    with col5:
        st.write("By City:")
        avg_price_by_city = df_cleaned.groupby('City')['Price range'].mean().sort_values(ascending=False).reset_index()
        avg_price_by_city.columns = ['City', 'Average Price Range']
        st.dataframe(avg_price_by_city.head(10))
        st.markdown("""
        **Insights:**
        - Higher average price ranges (closer to 4) indicate a prevalence of more expensive dining.
        - Lower average price ranges suggest an abundance of budget-friendly eateries.
        """)
    with col6:
        st.write("By Locality:")
        avg_price_by_locality = df_cleaned.groupby('Locality')['Price range'].mean().sort_values(ascending=False).reset_index()
        avg_price_by_locality.columns = ['Locality', 'Average Price Range']
        st.dataframe(avg_price_by_locality.head(10))

    # --- Analyze Cuisines by City and Locality ---
    st.subheader("Popular Cuisines and Culinary Diversity")

    # Explode cuisines for city-wise analysis
    df_cuisines_exploded = df_cleaned.assign(Cuisine=df_cleaned['Cuisines'].str.split(', ')).explode('Cuisine')

    col7, col8 = st.columns(2)
    with col7:
        st.write("Top 5 Cuisines in Top 3 Cities:")
        top_cities_for_cuisine = df_cleaned['City'].value_counts().head(3).index.tolist()
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
        top_localities_for_cuisine = df_cleaned['Locality'].value_counts().head(3).index.tolist()
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

else:
    st.warning("Please ensure 'Dataset .csv' is present in the same directory as this Streamlit app to run the analysis.")