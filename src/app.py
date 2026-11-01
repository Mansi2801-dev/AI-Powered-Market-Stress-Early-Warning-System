import os
import sys
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import torch

# 1. Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from model import MarketStressTransformer
from stress_threshold import load_thresholds, stress_regime

st.set_page_config(
    page_title="Market Stress Early Warning System",
    page_icon="⚡",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model & Scaler Loading
@st.cache_resource
def load_system_artifacts():
    model_path = PROJECT_ROOT / "models" / "stress_model.pth"
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    checkpoint = torch.load(model_path, map_location=device)
    model = MarketStressTransformer()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Load feature scaler if available
    scaler_path = PROJECT_ROOT / "models" / "feature_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    thresholds = load_thresholds()
    return model, scaler, thresholds

model, scaler, thresholds = load_system_artifacts()

# 3. Synthetic Order Book Simulator (Realistic Feature Scales)
def generate_market_scenario(scenario_type: str) -> np.ndarray:
    """
    Generates realistic 60-step windows with features:
    [L10_log, L50_log, spread, Imbalance]
    """
    T = 60
    if scenario_type == "Normal Calm Market":
        # High liquidity, tight spread, neutral imbalance
        L10 = np.random.normal(12.5, 0.2, T)
        L50 = np.random.normal(15.0, 0.2, T)
        spread = np.random.normal(0.0002, 0.00002, T)
        imb = np.random.normal(0.0, 0.1, T)

    elif scenario_type == "Liquidity Drain (Warning)":
        # Liquidity steadily drops over 60 timesteps
        L10 = np.linspace(12.0, 9.0, T) + np.random.normal(0, 0.1, T)
        L50 = np.linspace(14.5, 11.0, T) + np.random.normal(0, 0.1, T)
        spread = np.linspace(0.0002, 0.0008, T)
        imb = np.linspace(0.0, -0.4, T)

    elif scenario_type == "High Sell Pressure (High Stress)":
        # Severe bid-ask imbalance and thinning depth
        L10 = np.random.normal(8.5, 0.3, T)
        L50 = np.random.normal(10.5, 0.3, T)
        spread = np.random.normal(0.0020, 0.0002, T)
        imb = np.random.normal(-0.85, 0.05, T)

    else:  # Flash Crash / Book Emptying (Critical)
        # Liquidity void, massive spread
        L10 = np.random.normal(5.0, 0.5, T)
        L50 = np.random.normal(7.0, 0.5, T)
        spread = np.random.normal(0.0150, 0.002, T)
        imb = np.random.normal(-0.95, 0.02, T)

    raw_window = np.column_stack([L10, L50, spread, imb]).astype(np.float32)

    # Scale with training scaler if present
    if scaler is not None:
        scaled_window = scaler.transform(raw_window).astype(np.float32)
        return scaled_window
    return raw_window

# 4. Streamlit UI
st.title("⚡ AI-Powered Market Stress Early Warning System")
st.markdown("Transformer forecasting of order book instability **30 steps into the future** ($t+30$).")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Market Input Feed")
    feed_mode = st.radio("Select Input Source:", ["Simulated Market Scenario", "Sample from Test Dataset"])

    if feed_mode == "Simulated Market Scenario":
        scenario = st.selectbox(
            "Select Order Book Condition:",
            ["Normal Calm Market", "Liquidity Drain (Warning)", "High Sell Pressure (High Stress)", "Flash Crash / Book Emptying (Critical)"]
        )
        window = generate_market_scenario(scenario)
    else:
        test_file = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
        if test_file.exists():
            X_test = np.load(test_file)
            idx = st.slider("Select Test Sample Index:", 0, len(X_test) - 1, 0)
            window = X_test[idx]
        else:
            st.warning("No X_test.npy found. Falling back to Normal Market simulation.")
            window = generate_market_scenario("Normal Calm Market")

    run_check = st.button("🔍 Evaluate Future Market Stress", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Real-Time Forecast Analysis")

    if run_check:
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_score = model(input_tensor).item()

        regime = stress_regime(raw_score, thresholds)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Stress Score", f"{raw_score:.4f}")
        m2.metric("Market Regime", regime)
        m3.metric("Lookback / Horizon", "60 steps / +30 steps")

        st.divider()

        if regime == "NORMAL":
            st.success("### 🟢 Regime: NORMAL\nMarket depth is resilient. Normal order book flow with no early signs of liquidity collapse.")
        elif regime == "WARNING":
            st.warning("### 🟡 Regime: WARNING\nEarly-stage depth erosion detected. Spread variance expanding beyond normal ranges.")
        elif regime == "HIGH STRESS":
            st.error("### 🟠 Regime: HIGH STRESS\nSevere liquidity imbalance. High probability of adverse price slippage in the next 30 steps.")
        else:
            st.error("### 🚨 Regime: CRITICAL\nExtreme market stress detected! Order book depth has depleted. High risk of liquidity cascade or flash crash.")

        with st.expander("📊 Technical Breakdown & Active Thresholds"):
            st.json(thresholds)
            st.line_chart(window, use_container_width=True)