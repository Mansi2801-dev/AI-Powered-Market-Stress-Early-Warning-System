import os
import sys
import numpy as np
import torch
import streamlit as st

# ==========================================
# 1. PAGE CONFIG (MUST BE FIRST)
# ==========================================
st.set_page_config(page_title="Market Stress Dashboard", layout="wide")

# ==========================================
# 2. PATH SETUP
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
src_path = os.path.join(current_dir, "src")

if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.append(src_path)

from model import MarketStressTransformer

try:
    from stress_threshold import stress_regime
except ImportError:
    def stress_regime(score):
        if score < 0.3:
            return "NORMAL"
        elif score < 0.6:
            return "WARNING"
        elif score < 0.8:
            return "HIGH STRESS"
        else:
            return "CRITICAL"

# ==========================================
# 3. DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 4. MODEL LOADING (FIXED ARCHITECTURE)
# ==========================================
@st.cache_resource
def load_trained_model():

    checkpoint = torch.load(
        os.path.join(current_dir, "models", "stress_model.pth"),
        map_location=device
    )

    model = MarketStressTransformer()   # ❗ NO ARGUMENTS

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model

model = load_trained_model()

# ==========================================
# 5. UI
# ==========================================
st.title("📊 Market Stress Early Warning System")
st.markdown("Transformer-based prediction of market stress (t+30 horizon)")

# ==========================================
# 6. INPUT (PLACEHOLDER FOR NOW)
# ==========================================
def get_latest_window():
    # Replace later with real orderbook stream
    return np.random.randn(60, 4).astype(np.float32)

# ==========================================
# 7. PREDICTION
# ==========================================
if st.button("Run Stress Check", type="primary"):

    x = get_latest_window()
    x_tensor = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().item()

    regime = stress_regime(pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stress Score", f"{pred:.4f}")

    with col2:
        st.subheader(f"Regime: {regime}")

    # Alerts
    if regime == "NORMAL":
        st.success("Market is stable")
    elif regime == "WARNING":
        st.warning("Early stress detected")
    elif regime == "HIGH STRESS":
        st.error("High market stress")
    else:
        st.error("🚨 CRITICAL MARKET CONDITION")

# ==========================================
# 8. SIDEBAR
# ==========================================
st.sidebar.title("System Info")
st.sidebar.write("Model: Transformer")
st.sidebar.write("Features: Spread, L10_log, L50_log, Imbalance")
st.sidebar.write("Horizon: +30 steps")