import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# LOAD MODEL COMPONENTS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent


def build_fallback_components():
    # Synthetic training set for demo use when artifacts are unavailable.
    x = np.array([
        [90, 40, 40, 25, 80, 6.5, 200],
        [100, 35, 35, 26, 78, 6.3, 180],
        [20, 20, 20, 28, 65, 6.8, 120],
        [18, 22, 25, 27, 60, 7.0, 110],
        [50, 60, 50, 22, 85, 6.0, 250],
        [55, 65, 48, 23, 88, 5.8, 260],
        [70, 55, 45, 24, 70, 6.2, 150],
        [75, 50, 50, 25, 72, 6.4, 160],
        [35, 25, 30, 30, 55, 7.2, 90],
        [32, 28, 28, 31, 52, 7.1, 85],
    ], dtype=float)

    y_labels = np.array([
        "rice", "rice", "millet", "millet", "sugarcane",
        "sugarcane", "maize", "maize", "cotton", "cotton",
    ])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)

    fitted_scaler = StandardScaler()
    x_scaled = fitted_scaler.fit_transform(x)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(x_scaled, y_encoded)

    return clf, fitted_scaler, encoder


def safe_load_pickle(file_name):
    file_path = BASE_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_name}")
    if file_path.stat().st_size == 0:
        raise EOFError(f"Empty file: {file_name}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


load_error = None
try:
    model = safe_load_pickle("model.pkl")
    scaler = safe_load_pickle("scaler.pkl")
    le = safe_load_pickle("label_encoder.pkl")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    model, scaler, le = build_fallback_components()
    load_error = str(e)

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Agriculture DSS", layout="centered")

st.title("🌾 Smart Agriculture Decision Support System")
st.write("Predict the best crop based on soil and weather conditions")

if load_error:
    st.warning(
        "Model artifacts are missing or invalid. Running with a built-in fallback model. "
        "Upload valid model.pkl, scaler.pkl, and label_encoder.pkl for production predictions."
    )
    st.caption(f"Details: {load_error}")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.header("Enter Soil & Weather Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 150, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 40)
K = st.sidebar.slider("Potassium (K)", 0, 150, 40)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 80)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 200)

input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_crop(input_data):
    scaled = scaler.transform(input_data)
    
    probs = model.predict_proba(scaled)[0]
    classes = le.inverse_transform(model.classes_)
    
    top_indices = np.argsort(probs)[::-1][:3]
    
    return classes, probs, top_indices

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🌱 Predict Best Crop"):

    classes, probs, top_indices = predict_crop(input_data)

    st.subheader("🌾 Prediction Result")

    best_crop = classes[top_indices[0]]
    confidence = probs[top_indices[0]] * 100

    st.success(f"Best Recommended Crop: {best_crop}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Top 3 crops
    st.subheader("📊 Top 3 Crop Suggestions")

    for i in top_indices:
        st.write(f"🌿 {classes[i]} : {probs[i]*100:.2f}%")

    # Bar chart visualization
    st.subheader("📈 Probability Distribution")

    chart_data = {
        classes[i]: probs[i] for i in top_indices
    }

    st.bar_chart(chart_data)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ for Smart Agriculture DSS")