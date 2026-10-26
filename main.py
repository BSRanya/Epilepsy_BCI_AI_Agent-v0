"""
Streamlit demo: Real-Time ReAct Seizure Detection Agent (Path B).
Evaluates on the unseen test file: chb01_04.edf.
"""
import os
import time
import streamlit as st
import mne
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

from src.agent.agent import run_react_step

st.set_page_config(page_title="ReAct Seizure Agent", page_icon="🧠", layout="wide")
st.title("🧠 ReAct Seizure Detection Agent (Path B)")
st.caption("Powered by Groq · Cross-Session Test: chb01_04.edf · EMA Smoothed")


@st.cache_data
def load_eeg(path):
    """Load the unseen test file."""
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
        
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    target = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "FP2-F4"]
    
    # Check if channels exist
    missing = [c for c in target if c not in raw.ch_names]
    if missing:
        st.error(f"Missing channels in {path}: {missing}")
        st.stop()
        
    raw.pick_channels(target)
    raw.filter(0.5, 45.0, verbose=False)
    return raw


# Path to the UNSEEN test file
TEST_EEG_PATH = "data/data/chb01_04.edf"
raw = load_eeg(TEST_EEG_PATH)

data = raw.get_data()
fs = int(raw.info["sfreq"])
WIN = 5 * fs
STRIDE = 1 * fs
TOTAL = (data.shape[1] - WIN) // STRIDE

# In chb01_04.edf, the seizure starts at second 1467. 
# We jump to second 1455 to give the agent 12 seconds of "normal" data before onset.
SEIZURE_START_WINDOW = 1455 

# ---- Session state initialization ----
if "running" not in st.session_state:
    st.session_state.running = False
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "ema_prob" not in st.session_state:
    st.session_state.ema_prob = 0.0
if "trace" not in st.session_state:
    st.session_state.trace = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ---- Controls ----
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    if st.button("▶ Start", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()
with c2:
    if st.button("⏭️ Skip to Seizure", disabled=st.session_state.running):
        st.session_state.idx = SEIZURE_START_WINDOW
        st.session_state.ema_prob = 0.0
        st.session_state.trace = []
        st.session_state.alerts = []
        st.session_state.running = True
        st.rerun()
with c3:
    if st.button("⏹ Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()
with c4:
    progress_val = min(st.session_state.idx / TOTAL, 1.0)
    st.progress(progress_val, text=f"Window {st.session_state.idx} / {TOTAL}")

# ---- Main Layout ----
left, right = st.columns([2, 1])

with left:
    st.subheader("EEG Window (Test File)")
    plot_slot = st.empty()

with right:
    st.subheader("Status")
    raw_metric_slot = st.empty()
    ema_metric_slot = st.empty()
    alert_slot = st.empty()

st.subheader("ReAct Trace")
trace_slot = st.container()

# ---- Simulation Loop (1 window per rerun) ----
if st.session_state.running and st.session_state.idx < TOTAL:
    i = st.session_state.idx
    window = data[:, i * STRIDE : i * STRIDE + WIN]

    with st.spinner(f"Agent reasoning over window {i}..."):
        # 1. Run the ReAct agent
        result = run_react_step(window, trace=st.session_state.trace)
        
        raw_prob = result["detection"]["probability"] if result["detection"] else 0.0
        
        # 2. Update EMA (Alpha = 0.3)
        alpha = 0.3
        st.session_state.ema_prob = (alpha * raw_prob) + ((1 - alpha) * st.session_state.ema_prob)
        ema_prob = st.session_state.ema_prob

    # 3. Update UI - Plots and Metrics
    plot_slot.line_chart({f"Ch{j+1}": window[j] for j in range(window.shape[0])}, height=220)
    
    raw_metric_slot.metric("Raw Model Prob", f"{raw_prob:.2f}")
    
    # We use EMA for the visual threshold to prevent 1-second flickering
    if ema_prob >= 0.40:
        ema_metric_slot.metric("EMA Prob (Visual)", f"{ema_prob:.2f}", delta="SEIZURE", delta_color="inverse")
        alert_slot.error(f"🚨 **SEIZURE DETECTED** — {result['final_answer']}")
        if result["notified"]:
            st.session_state.alerts.append(f"W{i}: {result['final_answer']}")
    elif ema_prob >= 0.25:
        ema_metric_slot.metric("EMA Prob (Visual)", f"{ema_prob:.2f}", delta="Preictal", delta_color="off")
        alert_slot.warning(f"⚠️ **PREICTAL WARNING** — {result['final_answer']}")
    else:
        ema_metric_slot.metric("EMA Prob (Visual)", f"{ema_prob:.2f}", delta="Normal", delta_color="normal")
        alert_slot.success(f"✅ **Normal** — {result['final_answer']}")

    # 4. Render the ReAct Trace dynamically
    trace_slot.empty()
    with trace_slot:
        if st.session_state.trace:
            # Show only the last 3 steps to keep the UI clean
            for step in st.session_state.trace[-3:]:
                if step["action"]:
                    with st.expander(f"🔧 Tool: {step['action']['tool']}", expanded=True):
                        st.markdown(f"**Thought:** {step['thought']}")
                        st.markdown(f"**Action:** `{step['action']['tool']}`")
                        st.markdown(f"**Observation:** `{step['observation']}`")
                elif step["thought"]:
                     st.info(f"**Final Answer:** {step['thought']}")

    # 5. Advance loop
    st.session_state.idx += 1
    time.sleep(1.0)  # Real-time pacing

    if st.session_state.idx >= TOTAL:
        st.session_state.running = False
        st.success("✅ Simulation complete.")
    else:
        st.rerun()

elif st.session_state.idx >= TOTAL and TOTAL > 0:
    st.session_state.running = False
    st.success("✅ Simulation complete.")