# 🚖 Taxi Ride Clustering Prediction

### Predict which cluster a taxi ride belongs to using unsupervised machine learning models.

This project uses **KMeans**, **Hierarchical Clustering**, and **DBSCAN** to group taxi rides based on ride distance, duration, traffic, and fare patterns.  
It helps identify common ride types — from regular city rides to long intercity trips or unusual outlier rides.

---

## 🌐 Live Demo

👉 **Streamlit App Link:** [https://taxirideclustering.streamlit.app/](https://taxirideclustering.streamlit.app/)

---

## 📘 Project Overview

**Goal:**  
To analyze taxi trip data and group rides into meaningful clusters based on trip characteristics like fare, distance, duration, traffic, and weather.

**Approach:**
1. Performed **Exploratory Data Analysis (EDA)** and preprocessing.  
2. Applied **Log Transform (Skew Correction)** to reduce data skewness.  
3. Used **IQR-based Outlier Clipping** to handle extreme values.  
4. Scaled features with **StandardScaler** for uniformity.  
5. Reduced dimensionality with **Principal Component Analysis (PCA)**.  
6. Trained three unsupervised models:
   - 🟢 **KMeans**
   - 🟣 **Hierarchical Clustering**
   - 🔵 **DBSCAN**
7. Built a **Streamlit web app** for real-time cluster prediction.

---

## 📊 Dataset Details

**Dataset Name:** `taxi_trip_pricing.csv`  
**Dataset Link:** [https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction)

This dataset contains synthetic or real-world taxi trip data with the following columns:

| **Feature Name** | **Description** |
|------------------|-----------------|
| `Trip_Distance_km` | Total distance of the ride in kilometers |
| `Trip_Duration_Minutes` | Total trip duration in minutes |
| `Passenger_Count` | Number of passengers in the ride |
| `Base_Fare` | Starting fare for the trip (USD) |
| `Per_Km_Rate` | Rate charged per kilometer (USD) |
| `Per_Minute_Rate` | Rate charged per minute (USD) |
| `Trip_Price` | Final trip fare (USD) |
| `Time_of_Day` | Time period (Morning / Afternoon / Evening / Night) |
| `Day_of_Week` | Weekday or Weekend |
| `Traffic_Conditions` | Low / Medium / High traffic intensity |
| `Weather` | Clear / Rain / Snow weather condition |

---

## 🧠 Models & Clusters

### **KMeans Clusters**
| Cluster | Description |
|----------|--------------|
| 0 | 🌅 Morning commuters with calm traffic and lower fares. |
| 1 | 🔵 Typical city rides under normal conditions. |
| 2 | 🌙 Evening/night rides with faster movement and slightly higher prices. |
| 3 | 🛣️ Longer trips with moderate fares, possibly suburban routes. |

### **Hierarchical Clusters**
| Cluster | Description |
|----------|--------------|
| 0 | Short-distance rides with lower fares, mostly during calm traffic. |
| 1 | Medium-length city trips under typical traffic conditions. |
| 2 | Longer trips with higher fares, possibly evening or intercity rides. |

### **DBSCAN Clusters**
| Cluster | Description |
|----------|--------------|
| 0 | 🚕 Regular city rides with average distance and fare. |
| 1 | 🚦 Short, dense rides — possibly in heavy traffic zones. |
| 2 | 🌃 Late-night or long-duration rides with higher fares. |
| 3 | 🟢 Moderate city rides, consistent pricing patterns. |
| 4 | 🚌 Frequent commuter trips with stable fares and short distances. |
| 5 | 💎 High-end or intercity rides — rare and expensive. |
| -1 | ⚠️ Noise points — irregular or outlier rides not fitting clusters. |

---

## 🧰 Technologies & Libraries

| Category | Libraries Used |
|-----------|----------------|
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Modeling** | `scikit-learn` |
| **Clustering** | `KMeans`, `AgglomerativeClustering`, `DBSCAN` |
| **Web App** | `streamlit` |
| **Serialization** | `pickle` |

---

## ⚙️ Steps Performed in Notebook

1. **Data Cleaning**
   - Handled missing values using median and mode.
   - Removed duplicates.

2. **Feature Engineering**
   - Encoded categorical variables using one-hot encoding.

3. **Data Transformation**
   - Applied `np.log1p()` to skewed columns.
   - Clipped outliers using IQR.

4. **Scaling & Dimensionality Reduction**
   - Scaled numeric features with `StandardScaler`.
   - Reduced dimensions with `PCA(n_components=2)`.

5. **Model Training & Evaluation**
   - Determined best K for KMeans using **Elbow Method**.
   - Evaluated all models using **Silhouette Score**.
   - Saved trained models and preprocessors as `.pkl` files.

6. **App Development**
   - Built Streamlit UI for input fields and model selection.
   - Preprocessed new data using saved scaler and PCA.
   - Predicted cluster and displayed ride type description.

---

## 🚀 How to Run the Project

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/TaxiRideClustering.git
cd TaxiRideClustering
```

### **2️⃣ Create Virtual Environment**
```
python -m venv myvenv
myvenv\Scripts\activate      # On Windows
source myvenv/bin/activate   # On macOS/Linux
```

### **3️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit App**
```
streamlit run app.py
```

---

## 📂 Project Structure
```
TaxiRideClustering/
│
├── app.py                      # Streamlit App
├── notebook.ipynb              # Full EDA + Model Training
├── taxi_trip_pricing.csv       # Dataset
├── scaler.pkl                  # Saved StandardScaler
├── pca.pkl                     # Saved PCA
├── kmeans_model.pkl            # KMeans model
├── hierarchical_model.pkl      # Hierarchical model
├── dbscan_model.pkl            # DBSCAN model
├── feature_columns.pkl         # Feature alignment
├── requirements.txt            # Required packages
└── README.md                   # Documentation
```

---

## 🧭 App Usage
```
1. Enter ride details such as distance, duration, fare, etc.  
2. Choose **KMeans**, **Hierarchical**, or **DBSCAN** from the model options.  
3. Click **“🔮 Predict Cluster”** to get:
   - Cluster number  
   - Ride type description  
   - Model used  

### 💡 Example Output

✅ Predicted Cluster: 1 (KMeans)
🧭 Description: 🔵 Typical city rides under normal conditions.
```

---

## 💾 Requirements
```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```
