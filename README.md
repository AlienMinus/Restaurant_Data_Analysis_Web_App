# 🍽️ Restaurant Data Analysis App

An interactive Streamlit web application that allows users to explore restaurant data, predict restaurant ratings using machine learning models, visualize restaurant locations on a map, and evaluate classification performance.

---

## 📌 Features

- 🧹 **Data Cleaning & Encoding**: Automatically handles missing values and encodes categorical variables using `LabelEncoder`.
- 🔮 **Rating Prediction**: Predict restaurant ratings using a **Linear Regression** model.
- 🧠 **Classification Model**: Classify rounded ratings using a **Random Forest Classifier**.
- 🗺️ **Geospatial Visualization**: Visualize restaurants on an interactive **Folium Map** with clustering.
- 📈 **Model Evaluation**: View precision, recall, and classification report for the classifier.
- ⚡ **Streamlit Caching**: Uses `@st.cache_data` for optimized performance and faster reloads.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/restaurant-data-analysis-app.git
cd restaurant-data-analysis-app
```

### 2. Install Dependencies
Make sure you are using Python 3.8 or above.

```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Place your dataset file named Dataset .csv in the root directory. Ensure it includes features like:
- Aggregate rating (target variable)
- Categorical variables (e.g., Cuisines, City, Restaurant Name)
- Geolocation data (optional for map)

## 📊 Usage
Run the Streamlit App
```bash
streamlit run Merged.py
```
Then open http://localhost:8501 in your browser.

## 🗂️ App Structure
```bash
restaurant-data-analysis-app/
├── Merged.py                 # Main Streamlit app
├── Dataset .csv           # Dataset file
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## 🧠 Machine Learning Details
### Preprocessing
- Missing values are dropped using dropna().
- Categorical features are label encoded.
- Target: Aggregate rating

### Models Used
<table>
<tr>
<th>Task</th>
<th>Model</th>
</tr>
<tr>
<td>Rating Prediction</td>
<td>Linear Regression</td>
</tr>
<tr>
<td>Rating Classification</td>
<td>Random Forest Classifier</td>
</tr>
</table>
	
### Metrics Evaluated
- Accuracy
- Precision
- Recall
- Classification Report

## 🗺️ Map Visualization
- Uses folium and MarkerCluster to display restaurant locations on a map.
- Can be extended to filter by city, rating, or cuisine.

## 🧪 Example Use Cases
- Predict how a new restaurant might be rated based on location, cuisine, and other features.
- Analyze trends in restaurant ratings across different cities.
- Visualize where highly-rated restaurants are concentrated.

## 🛠️ Requirements
- streamlit
- pandas
- numpy
- scikit-learn
- folium
<br/> (Install using pip install -r requirements.txt)

## 📌 Notes
- Ensure your dataset contains a column named Aggregate rating.
- You may need to adjust LabelEncoder application if you include unseen categorical values during prediction.
- Rename Dataset .csv if needed or allow file uploads via Streamlit widgets.

## 📄 License
This project is open-source and available under the MIT License.

## 👨‍💻 Author
- AlienMinus
- Electrical & Computer Engineer
- GitHub: AlienMinus
- Email: dasmanasranjan2005@gmail.com

## 🌟 Acknowledgments
- Streamlit
- scikit-learn
- Folium




