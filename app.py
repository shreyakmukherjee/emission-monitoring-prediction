import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import joblib

# ---------- Constants & file paths ----------
SEQ_LENGTH = 30
PRED_LENGTH = 7

FEATURES_PATH = "features.pkl"
TARGETS_PATH = "targets.pkl"
SCALER_X_PATH = "scaler_x.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
MODEL_PATH = "emission_model.pth"   # <- use the state_dict you saved at the end

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load artifacts (you saved them with joblib) ----------
features = joblib.load(FEATURES_PATH)  # list of feature names
targets = joblib.load(TARGETS_PATH)    # list of targets, e.g. ['SO2 AQI', 'NO2 AQI', 'CO AQI']
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

INPUT_SIZE = len(features)
OUTPUT_SIZE = PRED_LENGTH * len(targets)  # flattened (7 days * 3 pollutants)
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2

# ---------- Model defs copied from your training code ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class EmissionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=8, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.embedding(x)                 # [batch, seq_len, hidden]
        x = self.pos_encoder(x)               # add positions
        x = self.transformer_encoder(x)       # [batch, seq_len, hidden]
        x = x.mean(dim=1)                     # pool over seq_len
        return self.decoder(x)                # [batch, output_size]

# ---------- Recreate model & load state dict ----------
model = EmissionPredictor(
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, NUM_HEADS, DROPOUT
).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)   # keys now match
model.eval()

# ---------- Inference helper ----------
def predict_next_7_days(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    df_input: DataFrame containing at least the last 30 rows of the required 'features' columns.
    Returns a DataFrame of shape [7, len(targets)] with predicted AQI values (inverse-transformed).
    """
    # Ensure required columns exist
    missing = [c for c in features if c not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Keep only features and sort by time if needed (assume already ordered)
    X = df_input[features].copy()

    if len(X) < SEQ_LENGTH:
        raise ValueError(f"Need at least {SEQ_LENGTH} rows; got {len(X)}.")

    # Take the last SEQ_LENGTH rows as the context window
    X_last = X.tail(SEQ_LENGTH).to_numpy()

    # Scale inputs with the trained scaler
    X_scaled = scaler_x.transform(X_last)              # [30, input_size]

    # Add batch dimension: [1, 30, input_size]
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()          # [1, OUTPUT_SIZE]
    pred = pred.reshape(PRED_LENGTH, len(targets))     # [7, 3]


    preds_unscaled = scaler_y.inverse_transform(pred)  # [7, 3]

    out = pd.DataFrame(preds_unscaled, columns=targets)
    return out




def build_future_index(n=7):
    """Return labels like Day 1 ... Day n"""
    return [f"Day {i}" for i in range(1, n + 1)]



# ---------- Streamlit UI ----------
st.title("🌫️ AQI 7-Day Forecast (Transformer)")
st.write("Upload at least the **last 30 rows** with the required feature columns to get a 7-day forecast.")

with st.expander("Required feature columns"):
    st.code(", ".join(features), language="text")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    try:
        df_in = pd.read_csv(file)
        st.subheader("Preview of uploaded data")
        st.dataframe(df_in.head())

        # if st.button("Predict next 7 days"):
        #     preds_df = predict_next_7_days(df_in)
        #     st.subheader("Predicted AQI (next 7 days)")
        #     st.dataframe(preds_df.style.format("{:.2f}"))

        #     # Optional: download
        #     st.download_button(
        #         "Download predictions as CSV",
        #         data=preds_df.to_csv(index=False).encode("utf-8"),
        #         file_name="aqi_predictions_7d.csv",
        #         mime="text/csv"
        #     )

        if st.button("Predict next 7 days"):
            preds_df = predict_next_7_days(df_in)     # shape [7, len(targets)]
            preds_df = preds_df.copy()

            # Build nicer x-axis labels
            preds_df.insert(0, "Horizon", build_future_index(n=len(preds_df)))

            st.subheader("Predicted AQI (next 7 days)")
            st.dataframe(preds_df.style.format({col: "{:.2f}" for col in preds_df.columns if col != "Horizon"}))

            # ---- Charts: one per pollutant ----
            st.markdown("### Forecast Charts")
            for col in [c for c in preds_df.columns if c != "Horizon"]:
                st.markdown(f"**{col}**")
                st.line_chart(
                    data=preds_df.set_index("Horizon")[[col]]
                )

            # Optional: download button
            st.download_button(
                "Download predictions as CSV",
                data=preds_df.to_csv(index=False).encode("utf-8"),
                file_name="aqi_predictions_7d.csv",
                mime="text/csv"
            )


    except Exception as e:
        st.error(f"Error: {e}")
