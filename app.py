import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# ==============================
# 🎯 Load Saved Models & Objects
# ==============================
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("dbscan_model.pkl", "rb") as f:
    dbscan = pickle.load(f)

with open("hierarchical_model.pkl", "rb") as f:
    hier = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ==============================
# 🧩 Helper Functions
# ==============================

def clip_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

def apply_skew(df, cols):
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

def encode_features(df):
    df = pd.get_dummies(
        df,
        columns=["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"],
        drop_first=True
    )
    return df

def preprocess_input(df):
    """Apply preprocessing exactly as in training."""
    df = clip_outliers(df, ["Trip_Price", "Trip_Distance_km"])
    df = apply_skew(df, ["Trip_Price", "Trip_Distance_km"])
    df = encode_features(df)

    # Align columns with training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    return X_pca


# ==============================
# 🌸 Streamlit UI Setup (Pink Blossom)
# ==============================
st.set_page_config(page_title="🚖 Taxi Ride Clustering App", layout="centered")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to bottom right, #ffe6f0, #fff8fc);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.stTextInput, .stNumberInput, .stSelectbox {
    border-radius: 10px;
}
div.stButton > button:first-child {
    background-color: #ff99c8;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #ff80b5;
    transform: scale(1.05);
}
h1, h2, h3 {
    color: #cc007a;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# 🏙️ App Layout
# ==============================
st.title("🚖 Taxi Ride Clustering Prediction")
st.markdown("### 🔍 Predict which cluster a taxi ride belongs to using unsupervised models.")

# ------------------------------
# 🧾 Input Section
# ------------------------------
st.subheader("📥 Enter Ride Details")

col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("📏 Trip Distance (km)", min_value=0.1, value=5.0)
    trip_duration = st.number_input("⏱️ Trip Duration (minutes)", min_value=1.0, value=25.0)
    passenger_count = st.number_input("👥 Passenger Count", min_value=1, max_value=6, value=2)
    base_fare = st.number_input("💵 Base Fare (USD)", min_value=1.0, value=3.0)
    per_km_rate = st.number_input("🚗 Per Km Rate", min_value=0.5, value=1.2)

with col2:
    per_minute_rate = st.number_input("🕐 Per Minute Rate", min_value=0.1, value=0.3)
    trip_price = st.number_input("💰 Trip Price (USD)", min_value=1.0, value=15.0)
    time_of_day = st.selectbox("🌞 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_of_week = st.selectbox("📅 Day of Week", ["Weekday", "Weekend"])
    traffic = st.selectbox("🚦 Traffic Condition", ["Low", "Medium", "High"])
    weather = st.selectbox("🌦️ Weather Condition", ["Clear", "Rain", "Snow"])

# ------------------------------
# 🤖 Model Selection
# ------------------------------
st.subheader("🤖 Choose Clustering Model")
model_choice = st.radio("Select Model:", ("KMeans", "Hierarchical", "DBSCAN"), horizontal=True)

# ------------------------------
# 🚀 Predict Button
# ------------------------------
if st.button("🔮 Predict Cluster"):
    # Prepare input DataFrame
    input_data = pd.DataFrame([{
        "Trip_Distance_km": trip_distance,
        "Trip_Duration_Minutes": trip_duration,
        "Passenger_Count": passenger_count,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_minute_rate,
        "Trip_Price": trip_price,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Traffic_Conditions": traffic,
        "Weather": weather
    }])

    # Preprocess
    X_pca = preprocess_input(input_data)

    # Handle Hierarchical separately
    if model_choice == "Hierarchical":
        st.warning("⚠️ Hierarchical clustering doesn’t support single-point predictions. It is mainly used for visualization and grouping on full datasets.")
    else:
        if model_choice == "KMeans":
            cluster = int(kmeans.predict(X_pca)[0])
            descriptions = {
                0: "Morning commuters with calm traffic and lower fares.",
                1: "Typical city rides under normal conditions.",
                2: "Evening/night rides with faster movement and slightly higher prices.",
                3: "Longer trips with moderate fares, possibly suburban routes."
            }

        elif model_choice == "DBSCAN":
            # Use pre-trained DBSCAN model (no refitting)
            labels = dbscan.fit_predict(X_pca)
            cluster = int(labels[0]) if len(labels) > 0 else -1
            descriptions = {
                0: "Regular city rides with average distance and fare.",
                1: "Short, dense rides — possibly in heavy traffic zones.",
                2: "Late-night or long-duration rides with higher fares.",
                3: "Moderate city rides, consistent pricing patterns.",
                4: "Frequent commuter trips with stable fares and short distances.",
                5: "High-end or intercity rides — rare and expensive.",
                6: "Compact trips during off-peak hours with low fares.",
                -1: "Noise points — irregular or outlier rides not fitting clusters."
            }

        # Display Result
        st.success(f"✅ Predicted Cluster: **{cluster}** ({model_choice})")
        st.info(f"🧭 Description: {descriptions.get(cluster, 'No description available.')}")

st.markdown("---")
st.caption("Developed with 💖 by **Sunmathi** | Models: KMeans • Hierarchical • DBSCAN")
