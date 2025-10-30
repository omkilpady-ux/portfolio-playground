# streamlit_app.py
# Portfolio Playground – Streamlit (v1)
# - Manual weights (no forced auto-balance)
# - Remainder automatically treated as Cash at the risk‑free rate (or you can scale to 100%)
# - Per-asset assumptions editable (Exp. Return, Vol)
# - Risk-free input (default 6%)
# - Scenario Simulator (Best / Base / Worst) with years picker
# - Live metrics: Expected Return, Volatility, Sharpe, Max Drawdown
# - Charts: Allocation pie, Simulated performance, Scenario growth (matplotlib; one chart per figure; no custom colors)
# - CSV exports via download buttons
#
# Run:  
#   pip install streamlit numpy pandas matplotlib
#   streamlit run streamlit_app.py

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Portfolio Playground (Streamlit)",
    layout="wide",
)

# -------------------- Defaults --------------------
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

# -------------------- Sidebar --------------------
st.sidebar.header("Controls")
amount = st.sidebar.number_input("Total Investable Amount (₹)", min_value=0, value=3_800_000, step=10_000)
rf_pct = st.sidebar.number_input("Risk‑Free (annual, %)", min_value=0.0, max_value=50.0, value=6.0, step=0.25)
rf = rf_pct / 100.0
sim_months = st.sidebar.slider("Time Horizon (months) for simulation", min_value=12, max_value=180, value=60, step=6)
num_paths = st.sidebar.slider("Paths (uncertainty)", min_value=1, max_value=200, value=20, step=1)

st.sidebar.markdown("---")
scale_now = st.sidebar.button("Scale weights to 100%")

# -------------------- Main layout --------------------
st.title("Portfolio Playground (Streamlit)")
st.caption("Simple, visual, and educational. Tweak allocations and see risk/return change in real time.")

# ---- Assumptions table (editable) ----
with st.expander("Assumptions (edit if you like)", expanded=True):
    df_assump = pd.DataFrame(
        {
            "Asset": [a[0] for a in ASSETS],
            "Label": [a[1] for a in ASSETS],
            "Expected Return %": [a[2] for a in ASSETS],
            "Volatility %": [a[3] for a in ASSETS],
        }
    )
    df_assump = st.data_editor(df_assump, use_container_width=True, disabled=["Asset", "Label"])

with st.expander("Correlation (advanced)"):
    corr = st.data_editor(DEFAULT_CORR.copy(), use_container_width=True)
else:
    corr = DEFAULT_CORR.copy()

# ---- Manual Weights ----
st.subheader("Allocations (Manual — you decide)")
cols = st.columns(len(ASSETS) + 1)
weights = {}
for i, (key, label, _, _) in enumerate(ASSETS):
    weights[key] = cols[i].number_input(f"{label} %", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key=f"w_{key}")

sum_w = sum(weights.values())
remainder = 100.0 - sum_w

# Remainder = Cash at risk‑free. If >100, warn and optionally scale.
if sum_w > 100.0 + 1e-9:
    st.error(f"Weights add up to {sum_w:.1f}%. Reduce or click 'Scale weights to 100%' in the sidebar.")
    if scale_now:
        weights = {k: v * 100.0 / sum_w for k, v in weights.items()}
        sum_w = 100.0
        remainder = 0.0
        st.success("Scaled weights to 100%.")
else:
    st.info(f"Unallocated: {remainder:.1f}% → treated as **Cash** at risk‑free.")

# Build asset frame including Cash
assump = df_assump.set_index("Asset")
assump = assump[["Label", "Expected Return %", "Volatility %"]].copy()
assump.loc["Cash"] = ["Cash (Risk‑Free)", rf_pct, 0.0]

w_series = pd.Series(weights)
if remainder > 0:
    w_series.loc["Cash"] = remainder
else:
    w_series.loc["Cash"] = 0.0

w = (w_series / 100.0).astype(float)

# Extend correlation to include Cash as zero-corr with 0 vol
corr_ext = corr.copy()
corr_ext.loc["Cash"] = 0.0
corr_ext["Cash"] = 0.0
corr_ext.loc["Cash", "Cash"] = 1.0

vol_vec = (assump["Volatility %"] / 100.0).reindex(w.index).fillna(0.0)
mu_vec = (assump["Expected Return %"] / 100.0).reindex(w.index).fillna(rf)

# Covariance
cov = np.outer(vol_vec, vol_vec) * corr_ext.values

# Portfolio stats
exp_ret = float(np.dot(w, mu_vec))
variance = float(np.dot(w, np.dot(cov, w)))
sd = math.sqrt(max(variance, 0.0))
sharpe = (exp_ret - rf) / sd if sd > 0 else 0.0

# ----- Charts & Metrics row -----
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Expected Return", f"{exp_ret*100:.2f}%")
mc2.metric("Volatility (SD)", f"{sd*100:.2f}%")
mc3.metric("Sharpe", f"{sharpe:.2f}")
# Monte Carlo simulation (single representative path = median of many)

# Simulate function
rng = np.random.default_rng(7)

# Cholesky of monthly covariance
cov_monthly = cov / 12.0
try:
    L = np.linalg.cholesky(cov_monthly)
except np.linalg.LinAlgError:
    # Fallback to near-PSD
    eigvals, eigvecs = np.linalg.eigh(cov_monthly)
    eigvals[eigvals < 0] = 1e-12
    cov_monthly = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(cov_monthly)

mu_monthly = (1.0 + mu_vec) ** (1.0 / 12.0) - 1.0

paths = []
for p in range(num_paths):
    value = 1.0
    series = [value]
    for t in range(sim_months):
        z = rng.standard_normal(size=len(w))
        shock = L @ z
        r = mu_monthly + shock
        port_r = float(np.dot(w, r))
        value *= (1.0 + port_r)
        series.append(value)
    paths.append(series)

arr = np.array(paths)
median_path = np.median(arr, axis=0)
# Max drawdown on median path
peak = median_path[0]
mdd = 0.0
for v in median_path:
    if v > peak:
        peak = v
    mdd = min(mdd, v / peak - 1.0)
mc4.metric("Max Drawdown (sim)", f"{abs(mdd)*100:.2f}%")

# Allocation pie
fig_alloc, ax = plt.subplots()
labels = [assump.loc[idx, "Label"] for idx in w.index if w[idx] > 0]
vals = [w[idx] for idx in w.index if w[idx] > 0]
ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig_alloc, use_container_width=True)

# Simulated performance chart (median path)
fig_sim, ax2 = plt.subplots()
ax2.plot(median_path)
ax2.set_xlabel("Months")
ax2.set_ylabel("Portfolio Index (x)")
ax2.grid(True, alpha=0.3)
st.pyplot(fig_sim, use_container_width=True)

# Download CSV for simulation
sim_df = pd.DataFrame({"month": list(range(len(median_path))), "portfolio_index": median_path})
buf_sim = io.StringIO()
sim_df.to_csv(buf_sim, index=False)
st.download_button("Download Simulation CSV", data=buf_sim.getvalue(), file_name="portfolio_simulation.csv", mime="text/csv")

# -------------------- Scenario Simulator --------------------
st.subheader("Scenario Simulator (Best / Base / Worst)")
colY1, colY2 = st.columns(2)
years = colY1.slider("Years", min_value=1, max_value=30, value=5, step=1)
colY2.caption(f"Risk‑free used for Cash and comparisons: {rf_pct:.2f}%")

sc = pd.DataFrame(
    {
        "Asset": [a[0] for a in ASSETS],
        "Weight %": [weights[a[0]] for a in ASSETS],
        "Best %": [assump.loc[a[0], "Expected Return %"] + 3 for a in ASSETS],
        "Base %": [assump.loc[a[0], "Expected Return %"] for a in ASSETS],
        "Worst %": [max(assump.loc[a[0], "Expected Return %"] - 6, 0) for a in ASSETS],
    }
)
sc = st.data_editor(sc, use_container_width=True, disabled=["Asset", "Weight %"])  # user edits returns

# Compute blended annual returns for scenarios (weighted by current sliders)
keys = sc["Asset"].tolist()
weights_now = np.array([sc.loc[sc["Asset"] == k, "Weight %"].values[0] for k in keys]) / 100.0
# Include Cash remainder in the blend at rf
weights_now = np.append(weights_now, max(remainder, 0.0) / 100.0)

best_vec = np.array([sc.loc[sc["Asset"] == k, "Best %"].values[0] for k in keys]) / 100.0
base_vec = np.array([sc.loc[sc["Asset"] == k, "Base %"].values[0] for k in keys]) / 100.0
worst_vec = np.array([sc.loc[sc["Asset"] == k, "Worst %"].values[0] for k in keys]) / 100.0

# Append Cash scenario returns (rf) to each
best_vec = np.append(best_vec, rf)
base_vec = np.append(base_vec, rf)
worst_vec = np.append(worst_vec, rf)

bestR = float(np.dot(weights_now, best_vec))
baseR = float(np.dot(weights_now, base_vec))
worstR = float(np.dot(weights_now, worst_vec))

months = years * 12
line = pd.DataFrame({
    "month": list(range(months + 1)),
    "Best": [(1 + bestR) ** (t / 12.0) for t in range(months + 1)],
    "Base": [(1 + baseR) ** (t / 12.0) for t in range(months + 1)],
    "Worst": [(1 + worstR) ** (t / 12.0) for t in range(months + 1)],
})

# Scenario chart
fig_sc, ax3 = plt.subplots()
ax3.plot(line["month"], line["Best"], label="Best")
ax3.plot(line["month"], line["Base"], label="Base")
ax3.plot(line["month"], line["Worst"], label="Worst")
ax3.set_xlabel("Months")
ax3.set_ylabel("Growth (x)")
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig_sc, use_container_width=True)

# Future values
def fv(ann):
    return amount * ((1 + ann) ** years)

c1, c2, c3, c4 = st.columns(4)
c1.metric("FV (Best)", f"₹{fv(bestR):,.0f}", f"CAGR {bestR*100:.1f}%")
c2.metric("FV (Base)", f"₹{fv(baseR):,.0f}", f"CAGR {baseR*100:.1f}%")
c3.metric("FV (Worst)", f"₹{fv(worstR):,.0f}", f"CAGR {worstR*100:.1f}%")
excess = ( (1+baseR)**years / (1+rf)**years - 1.0 ) * 100.0
c4.metric("Vs Risk‑Free", f"{excess:.1f}%", f"after {years}y")

# Download scenario CSV
buf_line = io.StringIO()
line.to_csv(buf_line, index=False)
st.download_button("Download Scenario CSV", data=buf_line.getvalue(), file_name="scenario_paths.csv", mime="text/csv")

st.caption("Educational only. Not investment advice.")
