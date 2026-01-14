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
# Using MLE for (alpha, theta) with loc fixed at 0
# Manual KS statistic (like your example) + parametric bootstrap p-value
# ------------------------------------------------------------

# -----------------------
# 1) Manual KS statistic
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

# -----------------------
# 2) MLE fit (SciPy) with loc fixed at 0
# -----------------------
def fit_gamma_mle(sample):
    """
    Fit Gamma(shape, scale) by MLE with loc fixed at 0.
    Returns (alpha_hat, theta_hat).
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    # SciPy returns: a, loc, scale
    a_hat, loc_hat, scale_hat = stats.gamma.fit(x, floc=0)
    return a_hat, scale_hat

# -----------------------
# 3) KS test with parametric bootstrap (needed when params are estimated)
# -----------------------
def ks_gamma_mle_bootstrap(sample, num_sim=3000, alpha_level=0.05, random_state=42):
    """
    KS test for Gamma where (alpha, theta) are estimated by MLE from the sample.
    Uses parametric bootstrap:
      - Fit Gamma by MLE to observed data (loc=0)
      - Compute observed KS D
      - Simulate many samples from fitted Gamma
      - Refit Gamma by MLE for each simulated sample
      - Compute KS D each time
      - p-value = fraction(sim_D >= observed_D)
      - critical value = (1-alpha) quantile of sim_D
    Returns: (alpha_hat, theta_hat, D_obs, p_val, d_crit, D_dist)
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        raise ValueError(f"Sample too small after cleaning (n={n}).")

    # Fit on observed sample
    alpha_hat, theta_hat = fit_gamma_mle(x)

    # Observed KS statistic
    D_obs = ks_statistic_gamma(x, alpha_hat, theta_hat)

    # Bootstrap distribution
    D_dist = np.empty(num_sim)
    for s in range(num_sim):
        sim = rng.gamma(shape=alpha_hat, scale=theta_hat, size=n)
        sim.sort()

        # Refit on simulated sample (this is the key difference vs fixed-params KS)
        a_sim, theta_sim = fit_gamma_mle(sim)

        # Compute KS D under refit params
        D_dist[s] = ks_statistic_gamma(sim, a_sim, theta_sim)

    d_crit = np.quantile(D_dist, 1 - alpha_level)
    p_val = np.mean(D_dist >= D_obs)

    return alpha_hat, theta_hat, D_obs, p_val, d_crit, D_dist

# -----------------------
# 4) QQ plot (no statsmodels)
# -----------------------
def qqplot_gamma(sample, alpha, theta, title):
    """
    QQ plot against Gamma(alpha, theta) using SciPy gamma.ppf.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        print(f"Skipping QQ plot (n={n}): {title}")
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

# -----------------------
# 5) Run per visitor type
# -----------------------
num_sim = 3000
alpha_level = 0.05

print("\n=== KS tests: service times vs Gamma(MLE), by visitor type ===")

summary_rows = []

for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        print(f"\n{vtype}: not enough positive service times (n={len(service)}). Skipping.")
        continue

    # Fit + KS bootstrap test
    a_hat, theta_hat, D_obs, p_val, d_crit, D_dist = ks_gamma_mle_bootstrap(
        service, num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

    summary_rows.append({
        "Visitor type": vtype,
        "n": len(service),
        "alpha_hat (MLE)": a_hat,
        "theta_hat (MLE)": theta_hat,
        "D_obs": D_obs,
        "p_value (bootstrap)": p_val,
        f"D_crit_{alpha_level}": d_crit
    })

    print(f"\n--- {vtype} ---")
    print(f"n = {len(service)}")
    print(f"MLE fit: alpha_hat={a_hat:.6g}, theta_hat={theta_hat:.6g}")
    print(f"KS D_obs={D_obs:.6g}")
    print(f"bootstrap critical value (alpha={alpha_level}) = {d_crit:.6g}")
    print(f"bootstrap p-value = {p_val:.6g}")

    # Optional: show simulated D distribution
    plt.figure()
    plt.hist(D_dist, bins=35)
    plt.axvline(D_obs, linewidth=2, label="Observed D")
    plt.axvline(d_crit, linewidth=2, label=f"Critical (alpha={alpha_level})")
    plt.title(f"Bootstrap KS D distribution — {vtype}")
    plt.xlabel("D")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Summary table printed
summary_df = pd.DataFrame(summary_rows)
print("\n=== Summary table ===")
print(summary_df)

# -----------------------
# 6) QQ plots at the end (per visitor type, using MLE params)
# -----------------------
print("\n=== Q-Q plots (Gamma MLE) ===")
for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        continue

    a_hat, theta_hat = fit_gamma_mle(service)
    qqplot_gamma(service, a_hat, theta_hat, title=f"Q-Q: Service times vs Gamma(MLE) — {vtype}")
