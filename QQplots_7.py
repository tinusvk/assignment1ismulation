import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load data
FILE = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/Assingment1/ass_part_1_dataset.xlsx"
SHEET = "Data"

df = pd.read_excel(FILE, sheet_name = SHEET)
df = df.sort_values(["Day", "Time of day in seconds", "Total time in seconds"]).reset_index(drop=True)

FILE2 = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/dataSim2.xlsx"
SHEET2 = "Sheet1"

df_sim = pd.read_excel(FILE2, sheet_name = SHEET2)



# Clean column names (your file has trailing spaces in some column names)
df.columns = [c.strip() for c in df.columns]
df_sim.columns = [c.strip() for c in df_sim.columns]


# Optional: map visitor type to labels (assumption: 1/2 coding)
# If your course uses a different coding, just edit this mapping.
type_map = {1: "Employee", 2: "Student"}
df["Visitor type label"] = df["Visitor type"].map(type_map).fillna(df["Visitor type"].astype(str))
df_sim["Visitor type label"] = df_sim["Visitor type"].map(type_map).fillna(df_sim["Visitor type"].astype(str))


# ============================================================
# Two-sample Q-Q plots: waiting times by visitor type
# X-axis: REAL waiting times
# Y-axis: SIMULATED waiting times
# Real data: Day 1, 09:00–17:00
# Simulated data: df_sim (already Day 1)
# ============================================================

OPEN_SEC = 9 * 3600
CLOSE_SEC = 17 * 3600

visitor_types = ["Employee", "Student"]

for vtype in visitor_types:

    # --- Real data: Day 1, opening hours, filtered by visitor type ---
    real_day1 = df[
        (df["Day"] == 1) &
        (df["Time of day in seconds"].between(OPEN_SEC, CLOSE_SEC)) &
        (df["Visitor type label"] == vtype)
    ].copy()

    w_real = real_day1["Waiting time"].to_numpy()
    w_real = w_real[np.isfinite(w_real) & (w_real >= 0)]

    # --- Simulated data: all rows, filtered by visitor type ---
    sim_type = df_sim[df_sim["Visitor type label"] == vtype].copy()

    w_sim = sim_type["Waiting time"].to_numpy()
    w_sim = w_sim[np.isfinite(w_sim) & (w_sim >= 0)]

    print(f"\n{vtype}")
    print(f"  Real waiting times (Day 1, 09–17): n = {len(w_real)}")
    print(f"  Sim  waiting times (Day 1):        n = {len(w_sim)}")

    if len(w_real) < 5 or len(w_sim) < 5:
        print(f"  Skipping Q-Q plot for {vtype} (not enough data)")
        continue

    # --- Two-sample Q-Q ---
    w_real_sorted = np.sort(w_real)
    w_sim_sorted  = np.sort(w_sim)

    m = min(len(w_real_sorted), len(w_sim_sorted))
    p = (np.arange(1, m + 1) - 0.5) / m

    q_real = np.quantile(w_real_sorted, p)
    q_sim  = np.quantile(w_sim_sorted, p)

    # --- Plot ---
    plt.figure()
    plt.scatter(q_real, q_sim, alpha=0.7)

    lo = min(q_real.min(), q_sim.min())
    hi = max(q_real.max(), q_sim.max())
    plt.plot([lo, hi], [lo, hi], linewidth=2)

    plt.xlabel("Real waiting time quantiles (seconds)")
    plt.ylabel("Simulated waiting time quantiles (seconds)")
    plt.title(f"Two-sample Q-Q plot: waiting times — {vtype}\n(Day 1, 09:00–17:00)")
    plt.tight_layout()
    plt.show()