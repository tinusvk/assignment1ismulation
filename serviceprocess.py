import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================================
# Load data
# ============================================================
FILE = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/Assingment1/ass_part_1_dataset.xlsx"
SHEET = "Data"

df = pd.read_excel(FILE, sheet_name=SHEET)

# Clean column names (some files have trailing spaces)
df.columns = [c.strip() for c in df.columns]

# Map visitor type to labels (edit mapping if your coding differs)
type_map = {1: "Employee", 2: "Student"}
df["Visitor type label"] = df["Visitor type"].map(type_map).fillna(df["Visitor type"].astype(str))

# Sort by arrival time (not required for service-time fit, but fine to keep)
df = df.sort_values(["Day", "Time of day in seconds", "Total time in seconds"]).reset_index(drop=True)

# ============================================================
# KS test + QQ plots for Gamma service times, per visitor type
# WITH parameters estimated by MLE AND parametric bootstrap
# Using a 3-parameter Gamma (shape, loc, scale): loc is estimated too
# ============================================================

# -----------------------
# 1) Manual KS statistic (Gamma with loc)
# -----------------------
def ks_statistic_gamma_loc(sample_sorted, a, loc, scale):
    """
    Manual KS statistic D for Gamma(shape=a, loc=loc, scale=scale).
    sample_sorted must be sorted ascending.
    """
    n = len(sample_sorted)
    D = 0.0
    for i in range(n):
        F = stats.gamma.cdf(sample_sorted[i], a=a, loc=loc, scale=scale)
        D = max(D, (i + 1) / n - F, F - i / n)
    return D

# -----------------------
# 2) MLE fit (SciPy): 3-parameter Gamma (loc free)
# -----------------------
def fit_gamma_mle_3p(sample):
    """
    Fit Gamma(shape, loc, scale) by MLE (all 3 parameters free).
    Returns (a_hat, loc_hat, scale_hat).
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    a_hat, loc_hat, scale_hat = stats.gamma.fit(x)  # loc free
    return a_hat, loc_hat, scale_hat

# -----------------------
# 3) KS test with parametric bootstrap (params estimated)
# -----------------------
def ks_gamma_mle_bootstrap_3p(sample, num_sim=3000, alpha_level=0.05, random_state=42):
    """
    KS test for Gamma where (shape, loc, scale) are estimated by MLE from the sample.
    Uses parametric bootstrap:
      - Fit Gamma by MLE to observed data
      - Compute observed KS D
      - Simulate many samples from fitted Gamma
      - Refit Gamma by MLE for each simulated sample
      - Compute KS D each time
      - p-value = fraction(sim_D >= observed_D)
      - critical value = (1-alpha) quantile of sim_D

    Returns: (a_hat, loc_hat, scale_hat, D_obs, p_val, d_crit, D_dist)
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        raise ValueError(f"Sample too small after cleaning (n={n}).")

    # Fit on observed sample
    a_hat, loc_hat, scale_hat = fit_gamma_mle_3p(x)

    # Observed KS statistic
    D_obs = ks_statistic_gamma_loc(x, a_hat, loc_hat, scale_hat)

    # Bootstrap distribution
    D_dist = np.empty(num_sim)
    for s in range(num_sim):
        sim = rng.gamma(shape=a_hat, scale=scale_hat, size=n) + loc_hat
        sim.sort()

        # Refit on simulated sample
        a_sim, loc_sim, scale_sim = fit_gamma_mle_3p(sim)

        # KS D under refit params
        D_dist[s] = ks_statistic_gamma_loc(sim, a_sim, loc_sim, scale_sim)

    d_crit = np.quantile(D_dist, 1 - alpha_level)
    p_val = np.mean(D_dist >= D_obs)

    return a_hat, loc_hat, scale_hat, D_obs, p_val, d_crit, D_dist

# -----------------------
# 4) QQ plot against 3-parameter Gamma
# -----------------------
def qqplot_gamma_3p(sample, a, loc, scale, title):
    """
    QQ plot against Gamma(a, loc, scale) using SciPy gamma.ppf.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        print(f"Skipping QQ plot (n={n}): {title}")
        return

    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.gamma.ppf(p, a=a, loc=loc, scale=scale)

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

# ============================================================
# 5) Run per visitor type
# ============================================================
num_sim = 3000
alpha_level = 0.05

print("\n=== KS tests: service times vs Gamma(MLE), by visitor type (3-parameter Gamma) ===")

summary_rows = []

for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        print(f"\n{vtype}: not enough positive service times (n={len(service)}). Skipping.")
        continue

    # Fit + KS bootstrap test (3-parameter Gamma)
    a_hat, loc_hat, scale_hat, D_obs, p_val, d_crit, D_dist = ks_gamma_mle_bootstrap_3p(
        service, num_sim=num_sim, alpha_level=alpha_level, random_state=42
    )

    summary_rows.append({
        "Visitor type": vtype,
        "n": len(service),
        "shape a_hat (MLE)": a_hat,
        "loc_hat (MLE)": loc_hat,
        "scale_hat (MLE)": scale_hat,
        "D_obs": D_obs,
        "p_value (bootstrap)": p_val,
        f"D_crit_{alpha_level}": d_crit
    })

    print(f"\n--- {vtype} ---")
    print(f"n = {len(service)}")
    print(f"MLE fit: shape={a_hat:.6g}, loc={loc_hat:.6g}, scale={scale_hat:.6g}")
    print(f"KS D_obs={D_obs:.6g}")
    print(f"bootstrap critical value (alpha={alpha_level}) = {d_crit:.6g}")
    print(f"bootstrap p-value = {p_val:.6g}")

    # Show simulated D distribution
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

# ============================================================
# 6) QQ plots at the end (per visitor type, using 3p MLE params)
# ============================================================
print("\n=== Q-Q plots (Gamma MLE, 3-parameter) ===")
for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        continue

    a_hat, loc_hat, scale_hat = fit_gamma_mle_3p(service)
    qqplot_gamma_3p(
        service, a_hat, loc_hat, scale_hat,
        title=f"Q-Q: Service times vs Gamma(MLE, 3p) — {vtype}"
    )
