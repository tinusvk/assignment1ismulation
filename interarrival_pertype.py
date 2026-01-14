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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Quick sanity prints (do this once) ---
print("Unique visitor type labels:", df["Visitor type label"].unique())
print("Counts:\n", df["Visitor type label"].value_counts(dropna=False))
print("Service time non-null count:", df["Service time"].notna().sum())
print("Service time > 0 count:", (df["Service time"] > 0).sum())

# If you're in some IDEs, this helps force interactive plotting:
plt.ion()

# -----------------------
# Helpers: manual KS stat
# -----------------------
def ks_statistic_gamma(sample_sorted, alpha, theta):
    n = len(sample_sorted)
    D = 0.0
    for i in range(n):
        F = stats.gamma.cdf(sample_sorted[i], a=alpha, loc=0, scale=theta)
        diff_plus = (i + 1) / n - F
        diff_min  = F - i / n
        D = max(D, diff_plus, diff_min)
    return D

def ks_gamma_fixed_params(sample, alpha, theta, num_sim=2000, alpha_level=0.05, random_state=42):
    rng = np.random.default_rng(random_state)

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, np.array([])

    D_obs = ks_statistic_gamma(x, alpha, theta)

    D_dist = np.empty(num_sim)
    for s in range(num_sim):
        sim = rng.gamma(shape=alpha, scale=theta, size=n)
        sim.sort()
        D_dist[s] = ks_statistic_gamma(sim, alpha, theta)

    d_crit = np.quantile(D_dist, 1 - alpha_level)
    p_val = np.mean(D_dist >= D_obs)
    return D_obs, p_val, d_crit, D_dist

def fit_gamma_moments(sample):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    m = x.mean()
    v = x.var(ddof=1)
    if m <= 0 or v <= 0:
        return np.nan, np.nan
    a_hat = (m * m) / v
    theta_hat = v / m
    return a_hat, theta_hat

def qqplot_gamma(sample, alpha, theta, title):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        print(f"Skipping QQ plot (n={n}) for {title}")
        return

    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.gamma.ppf(p, a=alpha, loc=0, scale=theta)

    plt.figure()
    plt.scatter(q, x)
    lo = min(q.min(), x.min())
    hi = max(q.max(), x.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Theoretical quantiles (Gamma)")
    plt.ylabel("Empirical quantiles (Service time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

# -----------------------
# Run per visitor type
# -----------------------
alpha0 = 7.5
theta0 = 1.0

num_sim = 2000
alpha_level = 0.05

print("\n=== KS tests: service times vs Gamma, by visitor type ===")

summary_rows = []
grouped = list(df.groupby("Visitor type label"))
print("Number of groups found:", len(grouped))

for vtype, g in grouped:
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    print(f"\n--- {vtype} ---")
    print("Raw rows in group:", len(g))
    print("Positive service times:", len(service))

    # ALWAYS plot a histogram first (so you know plotting works)
    plt.figure()
    plt.hist(service, bins=40)
    plt.xlabel("Service time (seconds)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram: service times — {vtype}")
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

    if len(service) < 5:
        print("Not enough data for KS/QQ (need at least 5 positive values).")
        continue

    # 1) KS vs fixed hypothesis
    D0, p0, dcrit0, Ddist0 = ks_gamma_fixed_params(
        service, alpha=alpha0, theta=theta0,
        num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

    # 2) KS vs fitted gamma (method of moments)
    a_hat, theta_hat = fit_gamma_moments(service)

    Dhat, phat, dcrithat, Ddisthat = ks_gamma_fixed_params(
        service, alpha=a_hat, theta=theta_hat,
        num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

    print(f"Fixed Gamma(alpha={alpha0}, theta={theta0}): D={D0:.6g}, p~{p0:.6g}, crit~{dcrit0:.6g}")
    print(f"MoM   Gamma(alpha={a_hat:.6g}, theta={theta_hat:.6g}): D={Dhat:.6g}, p~{phat:.6g}, crit~{dcrithat:.6g}")

    summary_rows.append({
        "Visitor type": vtype,
        "n": len(service),
        "Hypothesis alpha": alpha0,
        "Hypothesis theta": theta0,
        "D (fixed)": D0,
        "p-value (fixed)": p0,
        "Critical D_0.05 (fixed)": dcrit0,
        "MoM alpha_hat": a_hat,
        "MoM theta_hat": theta_hat,
        "D (MoM)": Dhat,
        "p-value (MoM)": phat,
        "Critical D_0.05 (MoM)": dcrithat,
    })

    # QQ plots
    qqplot_gamma(service, alpha0, theta0, title=f"Q-Q: Service vs Gamma({alpha0},{theta0}) — {vtype}")
    qqplot_gamma(service, a_hat, theta_hat, title=f"Q-Q: Service vs Gamma(MoM fit) — {vtype}")

    # Plot simulated D distribution for fixed H0
    plt.figure()
    plt.hist(Ddist0, bins=35)
    plt.axvline(D0, linewidth=2, label="Observed D")
    plt.axvline(dcrit0, linewidth=2, label=f"Critical (alpha={alpha_level})")
    plt.title(f"KS D distribution under H0 Gamma({alpha0},{theta0}) — {vtype}")
    plt.xlabel("D")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

summary_df = pd.DataFrame(summary_rows)
print("\n=== Summary table ===")
print(summary_df)

# If you want to stop interactive mode at end:
plt.ioff()
plt.show()
