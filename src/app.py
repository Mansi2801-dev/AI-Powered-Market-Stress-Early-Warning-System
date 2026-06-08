import os
import sys
import numpy as np
import torch
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Market Stress Dashboard",
    layout="wide"
)

# ==========================================
# PATH SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

from model import MarketStressTransformer

# ==========================================
# STRESS REGIME FUNCTION
# ==========================================
try:
    from stress_threshold import stress_regime
except ImportError:

    def stress_regime(score):
        if score < -2.064538166778152:
            return "NORMAL"
        elif score < -1.2331195487050295:
            return "WARNING"
        elif score < 0.36684812526712596:
            return "HIGH STRESS"
        else:
            return "CRITICAL"

# ==========================================
# DEVICE
# ==========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_trained_model():

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "stress_model.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = MarketStressTransformer()

    # checkpoint save format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # old save format
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model

model = load_trained_model()

# ==========================================
# PLACEHOLDER INPUT
# ==========================================
def get_latest_window():

    return np.random.randn(
        60,
        4
    ).astype(np.float32)

# ==========================================
# UI
# ==========================================
st.title("📊 Market Stress Early Warning System")

st.markdown(
    """
    Transformer-based prediction of future market stress
    using order book dynamics.
    """
)

# ==========================================
# PREDICT BUTTON
# ==========================================
if st.button(
    "Run Stress Check",
    type="primary"
):

    x = get_latest_window()

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(
            x_tensor
        ).cpu().numpy().item()

    regime = stress_regime(pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Stress Score",
            f"{pred:.4f}"
        )

    with col2:
        st.metric(
            "Regime",
            regime
        )

    if regime == "NORMAL":
        st.success("Market is stable")

    elif regime == "WARNING":
        st.warning("Early stress detected")

    elif regime == "HIGH STRESS":
        st.error("High market stress")

    else:
        st.error("🚨 CRITICAL MARKET CONDITION")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("System Information")

st.sidebar.write("Model: Transformer")
st.sidebar.write("Features:")
st.sidebar.write("- Spread")
st.sidebar.write("- L10_log")
st.sidebar.write("- L50_log")
st.sidebar.write("- Imbalance")

st.sidebar.write("Prediction Horizon: +30 steps")

# ==========================================
# DEBUG PANEL
# ==========================================
with st.sidebar.expander("Debug"):

    st.write(
        "Project Root:",
        PROJECT_ROOT
    )

    st.write(
        "Model Path:",
        os.path.join(
            PROJECT_ROOT,
            "models",
            "stress_model.pth"
        )
    )