import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load data
FILE = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/Assingment1/ass_part_1_dataset.xlsx"
SHEET = "Data"

df = pd.read_excel(FILE, sheet_name = SHEET)
df = df.sort_values(["Day", "Time of day in seconds", "Total time in seconds"]).reset_index(drop=True)


# Clean column names (your file has trailing spaces in some column names)
df.columns = [c.strip() for c in df.columns]

# Optional: map visitor type to labels (assumption: 1/2 coding)
# If your course uses a different coding, just edit this mapping.
type_map = {1: "Employee", 2: "Student"}
df["Visitor type label"] = df["Visitor type"].map(type_map).fillna(df["Visitor type"].astype(str))

# Sort by arrival time (important for interarrival calculations)
df = df.sort_values(["Day", "Time of day in seconds", "Total time in seconds"]).reset_index(drop=True)

# ------------------------------------------------------------
# KS test + QQ plots for Gamma service times, per visitor type
# (no statsmodels; manual KS like your example; simulation p-val)
# ------------------------------------------------------------

# --- assumes df already loaded + cleaned + has "Visitor type label" ---
# df.columns stripped, df["Visitor type label"] exists, etc.

# -----------------------
# Helpers: manual KS stat
# -----------------------
def ks_statistic_gamma(sample_sorted, alpha, theta):
    """
    Manual KS statistic D for Gamma(shape=alpha, scale=theta, loc=0).
    sample_sorted must be sorted ascending.
    """
    n = len(sample_sorted)
    D = 0.0
    for i in range(n):
        F = stats.gamma.cdf(sample_sorted[i], a=alpha, loc=0, scale=theta)
        diff_plus = (i + 1) / n - F
        diff_min  = F - i / n
        D = max(D, diff_plus, diff_min)
    return D

def ks_gamma_fixed_params(sample, alpha, theta, num_sim=2000, alpha_level=0.05, random_state=42):
    """
    KS test for Gamma with FIXED parameters (alpha, theta).
    Simulate distribution of D under H0 (Gamma(alpha,theta)) like your example.
    Returns: D_obs, p_value_sim, d_crit, D_dist
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        raise ValueError(f"Sample too small after cleaning (n={n}).")

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
    """
    Method-of-moments fit for Gamma(loc=0).
    For Gamma: mean = a*theta, var = a*theta^2.
    a_hat = mean^2/var, theta_hat = var/mean
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    m = x.mean()
    v = x.var(ddof=1)
    a_hat = (m * m) / v if v > 0 else np.nan
    theta_hat = v / m if m > 0 else np.nan
    return a_hat, theta_hat

def qqplot_gamma(sample, alpha, theta, title):
    """
    QQ plot against Gamma(alpha, theta) using SciPy gamma.ppf.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)

    # plotting positions
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.gamma.ppf(p, a=alpha, loc=0, scale=theta)

    plt.figure()
    plt.scatter(q, x)
    # 45-degree line
    lo = min(q.min(), x.min())
    hi = max(q.max(), x.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Theoretical quantiles (Gamma)")
    plt.ylabel("Empirical quantiles (Service time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------
# Run per visitor type
# -----------------------

# Your hypothesized parameters (can change these)
alpha0 = 7.5   # shape
theta0 = 1.0   # scale

num_sim = 3000
alpha_level = 0.05

print("\n=== KS tests: service times vs Gamma, by visitor type ===")

summary_rows = []

for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].values
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        print(f"\n{vtype}: not enough positive service times (n={len(service)}). Skipping.")
        continue

    # 1) Test against YOUR fixed hypothesis Gamma(alpha0, theta0)
    D0, p0, dcrit0, Ddist0 = ks_gamma_fixed_params(
        service, alpha=alpha0, theta=theta0,
        num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

    # 2) Also test against a "fitted" gamma (moments) as "some sort of gamma"
    a_hat, theta_hat = fit_gamma_moments(service)
    Dhat, phat, dcrithat, Ddisthat = ks_gamma_fixed_params(
        service, alpha=a_hat, theta=theta_hat,
        num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

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

    print(f"\n--- {vtype} ---")
    print(f"n = {len(service)}")
    print(f"Fixed Gamma(alpha={alpha0}, theta={theta0}):    D={D0:.6g}, p~{p0:.6g}, crit~{dcrit0:.6g}")
    print(f"MoM fit   Gamma(alpha={a_hat:.6g}, theta={theta_hat:.6g}): D={Dhat:.6g}, p~{phat:.6g}, crit~{dcrithat:.6g}")

    # -----------------------
    # QQ plots
    # -----------------------
    qqplot_gamma(service, alpha0, theta0, title=f"Q-Q: Service times vs Gamma({alpha0}, {theta0}) — {vtype}")
    qqplot_gamma(service, a_hat, theta_hat, title=f"Q-Q: Service times vs Gamma(MoM fit) — {vtype}")

    # -----------------------
    # Optional: show simulated D distribution (like your example)
    # -----------------------
    plt.figure()
    plt.hist(Ddist0, bins=35)
    plt.axvline(D0, linewidth=2, label="Observed D")
    plt.axvline(dcrit0, linewidth=2, label=f"Critical (alpha={alpha_level})")
    plt.title(f"KS D distribution under H0 (Gamma({alpha0},{theta0})) — {vtype}")
    plt.xlabel("D")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

summary_df = pd.DataFrame(summary_rows)
print("\n=== Summary table ===")
print(summary_df)
