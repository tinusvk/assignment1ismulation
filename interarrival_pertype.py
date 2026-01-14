import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Load data (your code) ---
FILE = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/Assingment1/ass_part_1_dataset.xlsx"
SHEET = "Data"

df = pd.read_excel(FILE, sheet_name=SHEET)
df.columns = [c.strip() for c in df.columns]

type_map = {1: "Employee", 2: "Student"}
df["Visitor type label"] = df["Visitor type"].map(type_map).fillna(df["Visitor type"].astype(str))

df = df.sort_values(["Day", "Time of day in seconds", "Total time in seconds"]).reset_index(drop=True)

# -------------------------------
# 1) Build interarrival times
# -------------------------------
arrival_col = "Time of day in seconds"  # choose arrival clock; alternatively "Total time in seconds" if that's correct

df = df.sort_values(["Day", "Visitor type label", arrival_col]).reset_index(drop=True)
df["interarrival"] = df.groupby(["Day", "Visitor type label"])[arrival_col].diff()

# -----------------------
# KS test: Interarrival times vs Exponential (by visitor type)
# Replicates the style of your provided KS example
# -----------------------

alpha = 0.05
num_sim = 1000

for visitor_type, g in df.groupby("Visitor type label"):

    # Build within-day interarrivals for this type (same style as you did)
    g = g.sort_values(["Day", "Time of day in seconds"]).copy()
    g["Interarrival"] = np.nan

    for day, gg in g.groupby("Day", sort=True):
        idx = gg.index
        times = gg["Time of day in seconds"].values
        inter = np.diff(times, prepend=np.nan)
        g.loc[idx, "Interarrival"] = inter

    # sample = positive interarrivals only
    sample = g["Interarrival"].dropna().values
    sample = sample[sample > 0]

    # safety
    if len(sample) < 5:
        print(f"\n=== {visitor_type} ===")
        print("Too few interarrival observations after cleaning; skipping.")
        continue

    # ---- Fit exponential parameter (same as your approach) ----
    lambda_hat = 1 / sample.mean()
    scale_hat = 1 / lambda_hat

    # ---- Compute KS test statistic D (same structure as your example) ----
    sample.sort()
    diff_cdf = []  # list to store differences between theoretical and empirical cdf

    for i in range(len(sample)):
        cur_exp_cdf = stats.expon.cdf(x=sample[i], loc=0, scale=scale_hat)  # exponential CDF at sample point
        diff_plus = (i + 1)/len(sample) - cur_exp_cdf
        diff_min  = cur_exp_cdf - i/len(sample)
        diff_abs = max(diff_plus, diff_min)
        diff_cdf.append(diff_abs)

    D = max(diff_cdf)

    print(f"\n=== {visitor_type} ===")
    print(f"n = {len(sample)}")
    print(f"lambda_hat = {lambda_hat:.6f} per second (scale = {scale_hat:.6f})")
    print("D =", D)

    # ---- Simulate distribution of D (EXACT same idea as your example) ----
    sim_size = len(sample)
    D_dist_est = []

    for _ in range(num_sim):
        unif_sample = np.random.uniform(size=sim_size)
        unif_sample.sort()

        cur_D = max(
            max(
                (j+1)/len(unif_sample) - unif_sample[j],
                unif_sample[j] - j/len(unif_sample)
            )
            for j in range(len(unif_sample))
        )
        D_dist_est.append(cur_D)

    D_dist_est.sort()

    # critical value
    d_alpha = D_dist_est[int(num_sim*(1-alpha)) - 1]
    print(f"d_alpha (alpha={alpha}) =", d_alpha)

    # p-value (same style as your example)
    p_val = np.mean(D < np.array(D_dist_est))
    print("p-value (simulated) =", p_val)

    # Optional: show the D distribution as a histogram + observed D
    plt.figure()
    plt.hist(D_dist_est, bins=30)
    plt.axvline(D, linewidth=2, label="Observed D")
    plt.axvline(d_alpha, linewidth=2, label=f"Critical value (alpha={alpha})")
    plt.title(f"Simulated KS D distribution — {visitor_type}")
    plt.xlabel("D")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()