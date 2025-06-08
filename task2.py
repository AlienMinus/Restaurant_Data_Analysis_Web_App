import streamlit as st
import pandas as pd

st.title("Restaurant Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv('Dataset .csv')
    df_clean = df.dropna()
    return df_clean

df = load_data()

# Sidebar for user preferences
st.sidebar.header("Set Your Preferences")

# Cuisine options
cuisine_options = sorted(set(sum([str(c).split(', ') for c in df['Cuisines'].unique()], [])))
selected_cuisine = st.sidebar.selectbox("Cuisine Preference", options=["Any"] + cuisine_options)

# Price range
min_price = int(df['Average Cost for two'].min())
max_price = int(df['Average Cost for two'].max())
price_range = st.sidebar.slider("Price Range (for two)", min_price, max_price, (min_price, max_price))

# Location options
location_options = sorted(df['City'].unique())
selected_location = st.sidebar.selectbox("City", options=["Any"] + location_options)

# Minimum rating
min_rating = float(df['Aggregate rating'].min())
max_rating = float(df['Aggregate rating'].max())
selected_rating = st.sidebar.slider("Minimum Rating", min_rating, max_rating, min_rating)

# Minimum votes
min_votes = int(df['Votes'].min())
max_votes = int(df['Votes'].max())
selected_votes = st.sidebar.slider("Minimum Votes", min_votes, max_votes, min_votes)

# Recommendation function
def recommend_restaurants(
    df, 
    cuisine=None, 
    price_min=None, 
    price_max=None, 
    location=None, 
    min_rating=None, 
    min_votes=None
):
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
if st.sidebar.button("Get Recommendations"):
    results = recommend_restaurants(
        df,
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