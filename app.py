# =============================
# FILE: app.py
# =============================
# How to run locally:
# 1) Create & activate a virtual env (optional but recommended)
# 2) Install requirements:  pip install -r requirements.txt
# 3) Put your saved Keras model at: models/epilepsy_lstm.h5  (or change MODEL_PATH below)
# 4) (Optional) If you used a specific scaler during training, put it at: models/scaler.pkl
# 5) Run the app:  streamlit run app.py
# 6) Upload a single EEG row (178 values) as CSV, or paste comma-separated numbers.

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Optional

# Try to import TensorFlow/Keras lazily (so the app can still open without it)
MODEL = None
LOADED_SCALER = None
MODEL_PATH = "models/epilepsy_lstm2.keras"
SCALER_PATH = "models/scaler2.pkl"  # Optional

# -----------------------------
# Inject colorful CSS
# -----------------------------
CUSTOM_CSS = """
<style>
/* Background gradient */
.stApp {
  background: linear-gradient(135deg, #f0f4ff 0%, #ffeef8 100%);
}

/* Title styling */
h1, h2, h3 {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* Card look */
.block-container {
  padding-top: 2rem !important;
}

/* Pretty buttons */
.stButton>button {
  border-radius: 16px;
  padding: 0.6rem 1rem;
  border: none;
  box-shadow: 0 4px 14px rgba(0,0,0,0.12);
}

/***** Fancy result chip *****/
.result-chip {
  display: inline-block;
  padding: .5rem 1rem;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: .3px;
}
.result-seizure {
  background: #ffe1e6;
  color: #b8003a;
  border: 2px solid #ff99b3;
}
.result-normal {
  background: #e2ffe9;
  color: #006d2c;
  border: 2px solid #7be495;
}

/* Dataframe tweaks */
.css-1n76uvr, .stDataFrame { /* keep defaults nice */ }

/* Footer */
.footer-note {
  margin-top: 2rem; opacity: .8; font-size: .9rem;
}
</style>
"""

st.set_page_config(page_title="EEG Epilepsy Detector — LSTM", page_icon="🧠", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧠 EEG Epilepsy Detector — LSTM Demo")
st.write(
    "Upload a **single EEG row (178 values)** or paste comma-separated numbers.\n"
    "The app will preprocess the signal, run the **LSTM model**, and show whether it is **Epileptic (class 1)** or **Not Epileptic (classes 2–5)**."
)

# -----------------------------
# Utilities
# -----------------------------
def try_load_model() -> Optional[object]:
    global MODEL
    if MODEL is not None:
        return MODEL
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            MODEL = load_model(MODEL_PATH)
            return MODEL
        except Exception as e:
            st.warning(f"Found model at '{MODEL_PATH}' but failed to load: {e}")
            return None
    else:
        return None


def try_load_scaler():
    global LOADED_SCALER
    if LOADED_SCALER is not None:
        return LOADED_SCALER
    if os.path.exists(SCALER_PATH):
        try:
            import joblib
            LOADED_SCALER = joblib.load(SCALER_PATH)
            return LOADED_SCALER
        except Exception as e:
            st.info(f"No usable scaler found at '{SCALER_PATH}' (optional). Proceeding with per-sample standardization. Details: {e}")
    return None


def parse_csv_row(file_bytes: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes))  # try with header detection
    # We accept first non-empty row
    row = df.dropna(axis=0, how='all').iloc[0].values.astype(float)
    return row


def parse_pasted_numbers(text: str) -> np.ndarray:
    # Accept comma/space/newline separated numbers
    tokens = [t for t in text.replace("\n", ",").replace("\t", ",").split(",") if t.strip() != ""]
    arr = np.array([float(x) for x in tokens], dtype=float)
    return arr


def standardize_row(x: np.ndarray, method: str = "per-sample") -> np.ndarray:
    scaler = try_load_scaler()
    if scaler is not None:
        return scaler.transform(x.reshape(1, -1)).ravel()
    if method == "per-sample":
        mean = np.mean(x)
        std = np.std(x) or 1.0
        return (x - mean) / std
    elif method == "min-max":
        mn, mx = np.min(x), np.max(x)
        if mx - mn == 0:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    else:
        return x


def prepare_for_lstm(x1d: np.ndarray) -> np.ndarray:
    # Expected input shape: (batch, timesteps=178, features=1)
    x = x1d.reshape(1, -1, 1)
    return x


def safe_predict(x_lstm: np.ndarray) -> Optional[float]:
    model = try_load_model()
    if model is None:
        return None
    try:
        prob = float(model.predict(x_lstm, verbose=0).ravel()[0])
        # If model outputs logits or 2-class softmax, adapt here if needed.
        return prob
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def pretty_result(prob: float, threshold: float = 0.5) -> str:
    label = "Epileptic (Seizure)" if prob >= threshold else "Not Epileptic"
    css_class = "result-seizure" if prob >= threshold else "result-normal"
    st.markdown(f"<span class='result-chip {css_class}'> {label} — P(seizure) = {prob:.3f} </span>", unsafe_allow_html=True)
    return label


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")
with st.sidebar:
    st.write("**Model Files**")
    st.write("Place your trained Keras model at:\n`models/epilepsy_lstm.h5`.")
    st.caption("If you used a scaler during training, place it at `models/scaler.pkl`.")

    norm_method = st.selectbox("Normalization (used if no scaler found):", ["per-sample", "min-max", "none"], index=0)
    decision_threshold = st.slider("Decision threshold (P ≥ threshold → Epileptic)", 0.05, 0.95, 0.50, 0.01)

# -----------------------------
# Input Area
# -----------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("① Upload a CSV with one EEG row (178 values)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    sample_row = None
    if uploaded is not None:
        try:
            sample_row = parse_csv_row(uploaded.getvalue())
            st.success(f"Loaded {len(sample_row)} values from CSV.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with col_right:
    st.subheader("② Or paste comma/space separated numbers")
    pasted = st.text_area("Paste exactly 178 numbers (comma/space/newline separated):", height=140, placeholder="0.23, 0.18, 0.11, ...")
    if pasted.strip():
        try:
            arr = parse_pasted_numbers(pasted)
            st.success(f"Parsed {len(arr)} numbers from text.")
            sample_row = arr
        except Exception as e:
            st.error(f"Could not parse numbers: {e}")

# -----------------------------
# Validate Input & Plot
# -----------------------------
if sample_row is not None:
    if len(sample_row) != 178:
        st.warning(f"Expected 178 values, but got {len(sample_row)}. Please provide exactly 178 EEG samples.")
    else:
        with st.expander("Preview EEG Trace (178 time steps)", expanded=True):
            fig = plt.figure()
            plt.plot(np.arange(178), sample_row)
            plt.title("EEG Signal (1-second window, 178 samples)")
            plt.xlabel("Time step")
            plt.ylabel("Amplitude (a.u.)")
            st.pyplot(fig)

        # Standardize
        std_row = standardize_row(sample_row.copy(), method=norm_method)
        x_lstm = prepare_for_lstm(std_row)

        # Predict
        prob = safe_predict(x_lstm)

        st.subheader("Result")
        if prob is None:
            st.info("Model not found. Please place your trained model at **models/epilepsy_lstm.h5** and refresh.\n\nTip: In the meantime, you can still show the UI and plots as part of the demo.")
        else:
            pretty_result(prob, threshold=decision_threshold)

            # Nice metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Probability of Seizure", value=f"{prob:.3f}")
            with m2:
                st.metric(label="Threshold", value=f"{decision_threshold:.2f}")
            with m3:
                st.metric(label="Samples", value="178")

else:
    st.info("Upload a CSV with 178 values **or** paste numbers to begin.")

st.markdown("""
<div class="footer-note">
<strong>Notes</strong><br>
• The LSTM expects input shaped as <code>(batch, 178, 1)</code>.\
• If your original notebook used a specific scaler/normalizer, export it and place as <code>models/scaler.pkl</code> (Joblib).\
• If your saved model outputs logits or a two-class softmax, adapt <code>safe_predict()</code> accordingly.
</div>
""", unsafe_allow_html=True)


# =============================
# FILE: requirements.txt (copy into a separate file)
# =============================
# streamlit
# tensorflow==2.12.0
# numpy
# pandas
# matplotlib
# joblib  # only if using the optional scaler


# =============================
# OPTIONAL: Saving scaler during training (put this in your training notebook)
# =============================
# from sklearn.preprocessing import StandardScaler
# import joblib
# scaler = StandardScaler().fit(X_train)
# joblib.dump(scaler, 'models/scaler.pkl')

# =============================
# OPTIONAL: Model save (in your training notebook)
# =============================
# model.save('models/epilepsy_lstm.h5')
