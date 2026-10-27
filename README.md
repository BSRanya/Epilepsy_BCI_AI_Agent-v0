# 🧠 ReAct Seizure Detection Agent

Real-time seizure detection with cross-session generalization and LLM-powered reasoning.

---

## 🚀 What it does

- **Trains** a Random Forest classifier on CHB-MIT epilepsy patient data
- **Tests cross-session** — trains on `chb01_03.edf`, evaluates on the unseen `chb01_04.edf`
- **Runs real-time** via Streamlit: processes 5-second EEG windows with EMA-smoothed probability
- **Uses ReAct** — a Groq LLM reasons over detection results and decides when to alert
- **Generates reports** — confusion matrices, ROC/PR curves, threshold analysis, calibration plots

---

## 📊 Pipeline


    T[Train RF on chb01_03] --> E[Evaluate on chb01_04]
    E --> S[Streamlit Demo]
    S --> R[Reports & Plots]




## ⚡ Quick Start

```bash
# 1. Train the model
python train_model.py

# 2. Generate evaluation plots + metrics
python visualize_results.py

# 3. Launch the demo UI
streamlit run main.py
```

---

## 🛠 Tech Stack

| Layer | Tool |
|------|------|
| ML | `scikit-learn` Random Forest |
| EEG processing | `mne`, `numpy`, `scipy.signal` |
| LLM reasoning | `groq` + `qwen/qwen3.8-27b` |
| UI | `streamlit` |
| Features | Custom band‑pass + `welch` PSD estimation |
| Plotting | `matplotlib`, `seaborn` |

---

## 📁 Project Structure

```
BCI_Agent/
├── data/                  # EDF files + summaries
├── src/
│   ├── agent/             # ReAct loop + tool calls
│   ├── models/            # Trained RF model + metadata
│   ├── tools/             # Feature extraction, detection, notifications
│   └── prompts/           # System prompts for the LLM
├── main.py                # Streamlit demo 
├── train_model.py         # Training pipeline
└── visualize_results.py   # Evaluation & plots
```

---

## 🧩 Key Features

- **Cross‑session evaluation** — truly unseen patient file
- **EMA-smoothed probability** — 0.3 alpha prevents 1‑second UI flicker
- **ReAct agent** — LLM decides: seizure (≥0.75), preictal (0.45‑0.75), normal (<0.45)
- **Threshold‑aware UI** — visual alerts at 0.40 and 0.25 EMA thresholds
- **Full evaluation suite** — ROC, PR, confusion matrix, calibration, threshold sweep
- **Modular tools** — features, detection, notification, all reusable

---

## ⚠ Notes

- Model must be trained before running the demo or visualization
- Requires EDF files in `data/data/` and a summary file in `data/data/`
- This is a POC — not for clinical use without proper validation

---

## 📜 License

MIT
