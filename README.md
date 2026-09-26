# 🧠 Epilepsy Detection Using EEG Signals

Using an **LSTM (Long Short-Term Memory) neural network** to classify EEG signals as **Epileptic (seizure activity)** or **Not Epileptic**.

> **Note:** This project performs binary classification. The original dataset has 5 classes, but class **1** represents seizure activity and classes **2–5** represent non-seizure recordings.

---

## 📌 Epilepsy

Epilepsy may occur as a result of a genetic disorder or an acquired brain injury, such as a trauma or stroke. During a seizure, a person can experience abnormal behaviour, symptoms and sensations, sometimes including loss of consciousness. There may be few symptoms between seizures.

Epilepsy is usually treated with medication and, in some cases, surgery, devices or dietary changes.

### ⚡ Seizure

A seizure is a sudden surge of electrical activity in the brain. It can temporarily affect how a person appears or acts. Many different things can occur during a seizure depending on the brain region involved.

---

## 🔍 Detection

Automatic detection of epileptic seizures from **Electroencephalogram (EEG)** signals can help analyze large volumes of EEG recordings.

EEG signals are non-stationary and seizure patterns can vary between patients and recording sessions. EEG data can also contain different types of noise.

This project uses a deep-learning approach based on **LSTM networks** to learn patterns from successive EEG samples. The trained network produces a sigmoid probability that is converted into a binary prediction:

- **1 → Epileptic / seizure**
- **0 → Not Epileptic / non-seizure**

---

## 📊 Dataset

### `data.csv`

The dataset is based on the:

**UCI Machine Learning Repository — Epileptic Seizure Recognition Dataset**

Source:  
https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

The original dataset contains 5 classes. Each recording contains 4097 EEG measurements. In this project, the recordings were divided into chunks of **178 samples**, producing:

- **11,500 EEG samples**
- **178 EEG values per sample**
- **1 label per sample**
- **5 original classes**

The final CSV structure is:

```text
Unnamed: 0 | X1 | X2 | ... | X178 | y
```

### Original labels

| Label | Description | Binary class |
|---:|---|---:|
| 1 | Seizure activity | **1 — Epileptic** |
| 2 | EEG from the tumor area | **0 — Not Epileptic** |
| 3 | EEG from healthy brain area | **0 — Not Epileptic** |
| 4 | Eyes closed | **0 — Not Epileptic** |
| 5 | Eyes open | **0 — Not Epileptic** |

The model therefore performs:

```text
Original y = 1       → 1 (Epileptic)
Original y = 2–5     → 0 (Not Epileptic)
```

---

## 🧠 Model

The project uses a two-layer LSTM architecture:

```text
Input: 178 EEG values
        ↓
LSTM(64, return_sequences=True)
        ↓
Dropout(0.2)
        ↓
LSTM(32)
        ↓
Dropout(0.2)
        ↓
Dense(1, activation="sigmoid")
        ↓
Seizure probability
```

The LSTM expects input in the shape:

```text
(batch, 178, 1)
```

### Training

The original training workflow used:

- LSTM recurrent neural network
- Two LSTM layers
- Adam optimizer
- Binary cross-entropy loss
- Binary classification
- Training/validation split
- EEG preprocessing and normalization

---

## 💾 Saved Model Files

The current working application uses the following files:

```text
models/
├── epilepsy_lstm2_weights.npz
└── scaler2.joblib
```

### `epilepsy_lstm2_weights.npz`

Contains the trained LSTM weights.

The application reconstructs the LSTM architecture and loads these trained weights.

### `scaler2.joblib`

Contains the preprocessing scaler used during training.

The Streamlit application uses this scaler before sending the EEG signal to the LSTM.

> The old `epilepsy_lstm2.keras` file is retained in the repository history/project files, but the current working inference application uses the `.npz` weights file above.

---

# 🚀 Run the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/srd-33/Clg_Project.git
cd Clg_Project
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install streamlit tensorflow numpy pandas matplotlib joblib
```

If the project contains a `requirements.txt` file, you can instead run:

```bash
pip install -r requirements.txt
```

## 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, normally at:

```text
http://localhost:8501
```

---

# 🖥️ Using the EEG Detector

The application accepts a **single EEG sample containing exactly 178 values**.

You can either:

### Option 1 — Upload a CSV

Upload a CSV containing one row with 178 EEG values.

Example:

```text
135,190,229,223,192,125,55,-9,... 
```

### Option 2 — Paste EEG values

Paste exactly 178 EEG values into the text input.

The application will:

```text
178 EEG values
       ↓
Training scaler
       ↓
Reshape → (1, 178, 1)
       ↓
LSTM
       ↓
Sigmoid probability
       ↓
Threshold = 0.50
       ↓
Epileptic / Not Epileptic
```

---

# 📈 Prediction Output

The application displays:

- EEG waveform
- Seizure probability
- Decision threshold
- Number of input samples
- Final classification

The default decision rule is:

```text
P(seizure) >= 0.50 → Epileptic
P(seizure) <  0.50 → Not Epileptic
```

For example:

```text
P(seizure) = 0.972755
→ Epileptic (Seizure)
```

and:

```text
P(seizure) = 0.002348
→ Not Epileptic
```

The application displays the probability to **6 decimal places** so small differences between predictions remain visible.

---

# 📓 Notebooks

## `Model.ipynb`

Contains the model-development/training workflow, including:

1. Loading and exploring the EEG dataset
2. Visualizing the five original classes
3. Preparing the data for binary classification
4. Building the two-layer LSTM
5. Training the model
6. Evaluating training/validation behaviour
7. Saving the trained model information

## `Product.ipynb`

Contains the inference/product workflow for loading the trained model components and performing predictions.

## `Product_clean.ipynb`

A cleaned inference notebook for testing the saved LSTM weights and scaler on a single EEG sample.

---

# 📁 Project Structure

```text
Clg_Project/
│
├── app.py
├── Model.ipynb
├── Product.ipynb
├── Product_clean.ipynb
├── data.csv
├── new_eeg_normal.csv
│
├── models/
│   ├── epilepsy_lstm2_weights.npz
│   ├── scaler2.joblib
│   └── ...
│
└── README.md
```

---

# 🧪 Testing

The trained model was tested against real samples from `data.csv`.

The dataset contains:

```text
11,500 samples
178 EEG features per sample
5 original labels
```

Example verified predictions:

```text
Original label 1
Prediction ≈ 0.972755
→ Epileptic
```

```text
Original label 4
Prediction ≈ 0.002348
→ Not Epileptic
```

```text
Original label 4
Prediction ≈ 0.002438
→ Not Epileptic
```

These examples confirm that the reconstructed LSTM and Streamlit inference pipeline produce different predictions for different EEG signals.

---

# ⚠️ Important

This project is an **academic/demo machine-learning project** and is not a medical diagnostic device.

Predictions should not be used as a substitute for evaluation by a qualified medical professional.

---

# 👨‍💻 Author

**Skanda R. Dixit**

GitHub:  
https://github.com/srd-33

---

## ⭐ If you find this project useful

Feel free to explore, fork, or improve the project.
