# =============================
# FILE: app.py
# =============================
# Run:
#   python -m streamlit run app.py
#
# Input:
#   Single EEG row containing exactly 178 values.
#
# Model:
#   Reconstructed LSTM architecture + trained weights:
#   models/epilepsy_lstm2_weights.npz
#
# Scaler:
#   models/scaler2.joblib

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Optional


# =============================
# Configuration
# =============================

MODEL = None
LOADED_SCALER = None

WEIGHTS_PATH = "models/epilepsy_lstm2_weights.npz"
SCALER_PATH = "models/scaler2.joblib"


# =============================
# Streamlit Page Configuration
# =============================

st.set_page_config(
    page_title="EEG Epilepsy Detector — LSTM",
    page_icon="🧠",
    layout="wide"
)


# =============================
# Custom CSS
# =============================

CUSTOM_CSS = """
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f0f4ff 0%,
        #ffeef8 100%
    );
}

/* Headings */
h1, h2, h3 {
    font-family:
        'Inter',
        system-ui,
        -apple-system,
        Segoe UI,
        Roboto,
        Helvetica,
        Arial,
        sans-serif;
}

/* Main container */
.block-container {
    padding-top: 2rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 16px;
    padding: 0.6rem 1rem;
    border: none;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}

/* Result chip */
.result-chip {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.3px;
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

/* Footer */
.footer-note {
    margin-top: 2rem;
    opacity: 0.8;
    font-size: 0.9rem;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================
# Title
# =============================

st.title("🧠 EEG Epilepsy Detector — LSTM Demo")

st.write(
    "Upload a **single EEG row (178 values)** or paste comma-separated numbers.\n"
    "The app will preprocess the signal, run the **LSTM model**, "
    "and show whether it is **Epileptic (class 1)** or "
    "**Not Epileptic (classes 2–5)**."
)


# =============================
# Model Loading
# =============================

def try_load_model() -> Optional[object]:
    """
    Rebuild the LSTM architecture and load the trained weights
    from epilepsy_lstm2_weights.npz.
    """

    global MODEL

    if MODEL is not None:
        return MODEL

    if not os.path.exists(WEIGHTS_PATH):
        st.error(
            f"Model weights not found at "
            f"'{WEIGHTS_PATH}'"
        )
        return None

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Input,
            LSTM,
            Dropout,
            Dense
        )

        # Same architecture used during training
        MODEL = Sequential([
            Input(shape=(178, 1)),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(0.2),

            LSTM(32),

            Dropout(0.2),

            Dense(
                1,
                activation="sigmoid"
            )
        ])

        # Load trained weights
        weights_data = np.load(
            WEIGHTS_PATH
        )

        weights = [
            weights_data[key]
            for key in weights_data.files
        ]

        MODEL.set_weights(weights)

        return MODEL

    except Exception as e:
        st.error(
            f"Failed to load trained model weights: {e}"
        )
        return None


# =============================
# Scaler Loading
# =============================

def try_load_scaler():
    """
    Load the training-time StandardScaler.
    """

    global LOADED_SCALER

    if LOADED_SCALER is not None:
        return LOADED_SCALER

    if os.path.exists(SCALER_PATH):

        try:
            import joblib

            LOADED_SCALER = joblib.load(
                SCALER_PATH
            )

            return LOADED_SCALER

        except Exception as e:

            st.info(
                f"No usable scaler found at "
                f"'{SCALER_PATH}'. "
                f"Proceeding with per-sample "
                f"standardization. "
                f"Details: {e}"
            )

    return None


# =============================
# Input Parsing
# =============================

def parse_csv_row(
    file_bytes: bytes
) -> np.ndarray:

    try:

        df = pd.read_csv(
            io.BytesIO(file_bytes),
            header=None
        )

    except Exception:

        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    if df.empty:
        raise ValueError(
            "The uploaded CSV is empty."
        )

    # First non-empty row
    row = df.iloc[0].values.astype(float)

    return row


def parse_pasted_numbers(
    text: str
) -> np.ndarray:

    # Accept comma, space and newline separated values
    cleaned = (
        text
        .replace("\n", ",")
        .replace("\t", ",")
    )

    tokens = [
        t.strip()
        for t in cleaned.split(",")
        if t.strip() != ""
    ]

    # Also support space-separated values
    expanded_tokens = []

    for token in tokens:

        expanded_tokens.extend(
            token.split()
        )

    arr = np.array(
        [
            float(x)
            for x in expanded_tokens
        ],
        dtype=float
    )

    return arr


# =============================
# Normalization
# =============================

def standardize_row(
    x: np.ndarray,
    method: str = "training-scaler"
) -> np.ndarray:

    # Training-time scaler
    if method == "training-scaler":

        scaler = try_load_scaler()

        if scaler is None:
            raise ValueError(
                "Training scaler not found at "
                "models/scaler2.joblib."
            )

        return scaler.transform(
            x.reshape(1, -1)
        ).ravel()

    # Per-sample standardization
    if method == "per-sample":

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            std = 1.0

        return (x - mean) / std

    # Min-max normalization
    elif method == "min-max":

        mn = np.min(x)
        mx = np.max(x)

        if mx - mn == 0:
            return np.zeros_like(x)

        return (x - mn) / (mx - mn)

    # No normalization
    elif method == "none":

        return x

    else:

        raise ValueError(
            f"Unknown normalization method: {method}"
        )


# =============================
# Prepare LSTM Input
# =============================

def prepare_for_lstm(
    x1d: np.ndarray
) -> np.ndarray:

    # Expected model input:
    # (batch, timesteps, features)
    #
    # (1, 178, 1)

    return x1d.reshape(
        1,
        -1,
        1
    )


# =============================
# Prediction
# =============================

def safe_predict(
    x_lstm: np.ndarray
) -> Optional[float]:

    model = try_load_model()

    if model is None:
        return None

    try:

        prediction = model.predict(
            x_lstm,
            verbose=0
        )

        prob = float(
            prediction.ravel()[0]
        )

        return prob

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        return None


# =============================
# Display Result
# =============================

def pretty_result(
    prob: float,
    threshold: float = 0.5
) -> str:

    if prob >= threshold:

        label = "Epileptic (Seizure)"
        css_class = "result-seizure"

    else:

        label = "Not Epileptic"
        css_class = "result-normal"

    st.markdown(
        f"""
        <span class="result-chip {css_class}">
            {label} — P(seizure) = {prob:.6f}
        </span>
        """,
        unsafe_allow_html=True
    )

    return label


# =============================
# Sidebar
# =============================

st.sidebar.header("⚙️ Settings")

with st.sidebar:

    st.write("**Model Files**")

    st.write(
        "Using trained LSTM weights:\n"
        "`models/epilepsy_lstm2_weights.npz`."
    )

    st.caption(
        "Using training scaler:\n"
        "`models/scaler2.joblib`."
    )

    norm_method = st.selectbox(
        "Normalization:",
        [
            "training-scaler",
            "per-sample",
            "min-max",
            "none"
        ],
        index=0
    )

    decision_threshold = st.slider(
        "Decision threshold "
        "(P ≥ threshold → Epileptic)",
        0.05,
        0.95,
        0.50,
        0.01
    )


# =============================
# Input Area
# =============================

col_left, col_right = st.columns(
    [1, 1]
)


# -----------------------------
# CSV Upload
# -----------------------------

with col_left:

    st.subheader(
        "① Upload a CSV with one EEG row (178 values)"
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    sample_row = None

    if uploaded is not None:

        try:

            sample_row = parse_csv_row(
                uploaded.getvalue()
            )

            st.success(
                f"Loaded {len(sample_row)} "
                f"values from CSV."
            )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )


# -----------------------------
# Paste Input
# -----------------------------

with col_right:

    st.subheader(
        "② Or paste comma/space separated numbers"
    )

    pasted = st.text_area(
        "Paste exactly 178 numbers "
        "(comma/space/newline separated):",
        height=140,
        placeholder=(
            "0.23, 0.18, 0.11, ..."
        )
    )

    if pasted.strip():

        try:

            arr = parse_pasted_numbers(
                pasted
            )

            st.success(
                f"Parsed {len(arr)} "
                f"numbers from text."
            )

            sample_row = arr

        except Exception as e:

            st.error(
                f"Could not parse numbers: {e}"
            )


# =============================
# Validate Input & Plot
# =============================

if sample_row is not None:

    if len(sample_row) != 178:

        st.warning(
            f"Expected 178 values, "
            f"but got {len(sample_row)}. "
            f"Please provide exactly "
            f"178 EEG samples."
        )

    else:

        # -------------------------
        # EEG Graph
        # -------------------------

        with st.expander(
            "Preview EEG Trace (178 time steps)",
            expanded=True
        ):

            fig = plt.figure()

            plt.plot(
                np.arange(178),
                sample_row
            )

            plt.title(
                "EEG Signal "
                "(178 samples)"
            )

            plt.xlabel(
                "Time step"
            )

            plt.ylabel(
                "Amplitude (a.u.)"
            )

            st.pyplot(fig)

            plt.close(fig)


        # -------------------------
        # Standardize
        # -------------------------

        try:

            std_row = standardize_row(
                sample_row.copy(),
                method=norm_method
            )

            x_lstm = prepare_for_lstm(
                std_row
            )

        except Exception as e:

            st.error(
                f"Preprocessing failed: {e}"
            )

            x_lstm = None


        # -------------------------
        # Prediction
        # -------------------------

        if x_lstm is not None:

            prob = safe_predict(
                x_lstm
            )

            st.subheader("Result")

            if prob is None:

                st.info(
                    "Model could not be loaded. "
                    "Please check "
                    "**models/epilepsy_lstm2_weights.npz** "
                    "and refresh."
                )

            else:

                pretty_result(
                    prob,
                    threshold=decision_threshold
                )

                # Metrics
                m1, m2, m3 = st.columns(3)

                with m1:

                    st.metric(
                        label="Probability of Seizure",
                        value=f"{prob:.6f}"
                    )

                with m2:

                    st.metric(
                        label="Threshold",
                        value=f"{decision_threshold:.2f}"
                    )

                with m3:

                    st.metric(
                        label="Samples",
                        value="178"
                    )

else:

    st.info(
        "Upload a CSV with 178 values "
        "**or** paste numbers to begin."
    )


# =============================
# Footer
# =============================

st.markdown(
    """
    <div class="footer-note">
        <strong>Notes</strong><br>
        • The LSTM expects input shaped as
        <code>(batch, 178, 1)</code>.<br>

        • The app uses
        <code>models/scaler2.joblib</code>
        for training-time preprocessing.<br>

        • The trained model weights are stored in
        <code>models/epilepsy_lstm2_weights.npz</code>.
    </div>
    """,
    unsafe_allow_html=True
)