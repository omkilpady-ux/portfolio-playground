# streamlit_app.py (fixed)
# Portfolio Playground – Streamlit (v1.1)
# Fix: removed 'else' syntax bug and improved clarity.

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Playground (Streamlit)", layout="wide")

ASSETS = [
    ("Equities", "Equities (India)", 12.0, 18.0),
    ("Fixed Income", "Fixed Income", 7.0, 5.0),
    ("Mutual Funds", "Mutual Funds (Diversified)", 10.0, 14.0),
    ("Alternatives", "Alternatives (Gold/RE/Crypto)", 9.0, 20.0),
    ("Global", "Global (VOO/QQQ etc.)", 9.5, 16.0),
]

DEFAULT_CORR = pd.DataFrame(
    [
        [1.00, -0.10, 0.85, 0.30, 0.70],
        [-0.10, 1.00, -0.05, 0.00, -0.10],
        [0.85, -0.05, 1.00, 0.25, 0.60],
        [0.30, 0.00, 0.25, 1.00, 0.20],
        [0.70, -0.10, 0.60, 0.20, 1.00],
    ],
    columns=[a[0] for a in ASSETS],
    index=[a[0] for a in ASSETS],
)

# Sidebar controls
st.sidebar.header("Controls")
amount = st.sidebar.number_input("Total Investable Amount (₹)", min_value=0, value=3800000, step=10000)
rf_pct = st.sidebar.number_input("Risk‑Free (annual, %)", min_value=0.0, max_value=50.0, value=6.0, step=0.25)
rf = rf_pct / 100.0
sim_months = st.sidebar.slider("Time Horizon (months)", 12, 180, 60, 6)
num_paths = st.sidebar.slider("Paths (uncertainty)", 1, 200, 20, 1)
monthly_contrib = st.sidebar.number_input("Monthly contribution (₹)", min_value=0, value=0, step=1000)
include_monthly = st.sidebar.checkbox("Include monthly contribution in Scenario FV", value=True)

st.sidebar.markdown("---")
scale_now = st.sidebar.button("Scale weights to 100%")

# Main UI
st.title("Portfolio Playground (Streamlit)")
st.caption("Tweak allocations and see risk/return instantly.")

# Editable assumptions
df_assump = pd.DataFrame(
    {
        "Asset": [a[0] for a in ASSETS],
        "Label": [a[1] for a in ASSETS],
        "Expected Return %": [a[2] for a in ASSETS],
        "Volatility %": [a[3] for a in ASSETS],
    }
)
df_assump = st.data_editor(df_assump, use_container_width=True, disabled=["Asset", "Label"])

corr = st.data_editor(DEFAULT_CORR.copy(), use_container_width=True)

# Manual weights
st.subheader("Allocations (Manual — you decide)")
weights = {}
cols = st.columns(len(ASSETS))
for i, (k, label, _, _) in enumerate(ASSETS):
    weights[k] = cols[i].number_input(f"{label} %", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

sum_w = sum(weights.values())
remainder = 100.0 - sum_w

if sum_w > 100.0:
    st.error(f"Weights total {sum_w:.1f}%. Reduce or click 'Scale weights'.")
    if scale_now:
        weights = {k: v * 100.0 / sum_w for k, v in weights.items()}
        remainder = 0.0
        st.success("Scaled weights to 100%.")
else:
    st.info(f"Unallocated {remainder:.1f}% treated as Cash at {rf_pct:.1f}%.")

assump = df_assump.set_index("Asset")
assump.loc["Cash"] = ["Cash (Risk‑Free)", rf_pct, 0.0]

w = pd.Series(weights)
if remainder > 0:
    w.loc["Cash"] = remainder
else:
    w.loc["Cash"] = 0.0
w = w / 100.0

corr_ext = corr.copy()
corr_ext.loc["Cash"] = 0.0
corr_ext["Cash"] = 0.0
corr_ext.loc["Cash", "Cash"] = 1.0

vol_vec = assump["Volatility %"].reindex(w.index).fillna(0.0) / 100.0
mu_vec = assump["Expected Return %"].reindex(w.index).fillna(rf) / 100.0

cov = np.outer(vol_vec, vol_vec) * corr_ext.values
exp_ret = float(np.dot(w, mu_vec))
variance = float(np.dot(w, np.dot(cov, w)))
sd = math.sqrt(max(variance, 0.0))
sharpe = (exp_ret - rf) / sd if sd > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Expected Return", f"{exp_ret*100:.2f}%")
c2.metric("Volatility", f"{sd*100:.2f}%")
c3.metric("Sharpe", f"{sharpe:.2f}")

# Scenario Simulator
st.subheader("Scenario Simulator")
years = st.slider("Years", 1, 30, 5)

sc = pd.DataFrame({
    "Asset": [a[0] for a in ASSETS],
    "Weight %": [weights[a[0]] for a in ASSETS],
    "Best %": [assump.loc[a[0], "Expected Return %"] + 3 for a in ASSETS],
    "Base %": [assump.loc[a[0], "Expected Return %"] for a in ASSETS],
    "Worst %": [max(assump.loc[a[0], "Expected Return %"] - 6, 0) for a in ASSETS],
})
sc = st.data_editor(sc, use_container_width=True, disabled=["Asset", "Weight %"])

keys = sc["Asset"].tolist()
weights_now = np.array([sc.loc[sc["Asset"] == k, "Weight %"].values[0] for k in keys]) / 100.0
weights_now = np.append(weights_now, max(remainder, 0.0) / 100.0)

best_vec = np.array([sc.loc[sc["Asset"] == k, "Best %"].values[0] for k in keys]) / 100.0
base_vec = np.array([sc.loc[sc["Asset"] == k, "Base %"].values[0] for k in keys]) / 100.0
worst_vec = np.array([sc.loc[sc["Asset"] == k, "Worst %"].values[0] for k in keys]) / 100.0

best_vec = np.append(best_vec, rf)
base_vec = np.append(base_vec, rf)
worst_vec = np.append(worst_vec, rf)

bestR = float(np.dot(weights_now, best_vec))
baseR = float(np.dot(weights_now, base_vec))
worstR = float(np.dot(weights_now, worst_vec))

def fv(ann):
    n = years * 12
    principal = amount * ((1 + ann) ** years)
    if include_monthly and monthly_contrib > 0:
        r_m = (1 + ann) ** (1/12) - 1
        contrib = monthly_contrib * (((1 + r_m) ** n - 1) / r_m)
    else:
        contrib = 0
    return principal + contrib

fig, ax = plt.subplots()
months = years * 12
ax.plot([(1 + bestR) ** (t/12) for t in range(months+1)], label="Best")
ax.plot([(1 + baseR) ** (t/12) for t in range(months+1)], label="Base")
ax.plot([(1 + worstR) ** (t/12) for t in range(months+1)], label="Worst")
ax.legend()
ax.set_xlabel("Months")
ax.set_ylabel("Growth (x)")
ax.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("FV (Best)", f"₹{fv(bestR):,.0f}")
c2.metric("FV (Base)", f"₹{fv(baseR):,.0f}")
c3.metric("FV (Worst)", f"₹{fv(worstR):,.0f}")
excess = (((1 + baseR)**years / (1 + rf)**years) - 1) * 100
c4.metric("Vs Risk‑Free", f"{excess:.1f}%")

st.caption("Educational only. Not investment advice.")
