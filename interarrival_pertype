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

# keep only valid positive interarrivals
df_ia = df.loc[df["interarrival"].notna() & (df["interarrival"] > 0), ["Visitor type label", "interarrival"]].copy()

# -------------------------------
# 2) Manual KS statistic for Exp(scale)
# -------------------------------
def ks_statistic_expon(sample, scale):
    """
    Manual KS D for Exponential(loc=0, scale=scale).
    sample must be sorted.
    """
    n = len(sample)
    D = 0.0
    for i in range(n):
        F = stats.expon.cdf(sample[i], loc=0, scale=scale)
        diff_plus = (i + 1) / n - F
        diff_min  = F - i / n
        D = max(D, diff_plus, diff_min)
    return D

def ks_exp_parametric_bootstrap(sample, num_sim=2000, alpha=0.05, random_state=42):
    """
    KS test for Exponential where lambda (scale) is estimated from the sample.
    Uses parametric bootstrap to get p-value and critical value.
    Returns: D, p_value, d_alpha, lambda_hat, scale_hat
    """
    rng = np.random.default_rng(random_state)

    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample) & (sample > 0)]
    sample.sort()
    n = len(sample)
    if n < 5:
        raise ValueError(f"Sample too small (n={n}). Need at least ~5 positive observations.")

    # Fit exponential: scale_hat = mean, lambda_hat = 1/mean
    scale_hat = sample.mean()
    lambda_hat = 1.0 / scale_hat

    # Observed KS statistic
    D = ks_statistic_expon(sample, scale_hat)

    # Bootstrap distribution of D under Exp(fit), re-fitting each time
    D_dist = np.empty(num_sim)
    for s in range(num_sim):
        sim = rng.exponential(scale=scale_hat, size=n)
        sim.sort()

        scale_sim = sim.mean()  # refit
        D_dist[s] = ks_statistic_expon(sim, scale_sim)

    # critical value and p-value
    d_alpha = np.quantile(D_dist, 1 - alpha)
    p_val = np.mean(D_dist >= D)

    return D, p_val, d_alpha, lambda_hat, scale_hat, D_dist

# -------------------------------
# 3) Run per visitor type + plots
# -------------------------------
alpha = 0.05
num_sim = 2000

summary_rows = []

for vtype, g in df_ia.groupby("Visitor type label"):
    sample = g["interarrival"].values

    D, p_val, d_alpha, lam_hat, scale_hat, D_dist = ks_exp_parametric_bootstrap(
        sample, num_sim=num_sim, alpha=alpha, random_state=42
    )

    summary_rows.append({
        "Visitor type": vtype,
        "n": len(sample),
        "lambda_hat": lam_hat,
        "scale_hat": scale_hat,
        "D": D,
        f"d_alpha_{alpha}": d_alpha,
        "p_value": p_val
    })

    print(f"\n=== {vtype} ===")
    print(f"n = {len(sample)}")
    print(f"lambda_hat = {lam_hat:.6g}   (scale_hat = {scale_hat:.6g})")
    print(f"D = {D:.6g}")
    print(f"bootstrap d_alpha (alpha={alpha}) = {d_alpha:.6g}")
    print(f"bootstrap p-value = {p_val:.6g}")

    # Plot 1: Empirical CDF vs fitted exponential CDF
    x = np.sort(sample)
    n = len(x)
    emp_cdf = np.arange(1, n + 1) / n
    theo_cdf = stats.expon.cdf(x, loc=0, scale=scale_hat)

    plt.figure()
    plt.step(x, emp_cdf, where="post", label="Empirical CDF")
    plt.plot(x, theo_cdf, label="Fitted Exp CDF")
    plt.title(f"CDF comparison — {vtype}")
    plt.xlabel("Interarrival time (seconds)")
    plt.ylabel("CDF")
    plt.legend()
    plt.show()

    # Plot 2: Q-Q plot without statsmodels (use SciPy probplot)
    plt.figure()
    stats.probplot(sample, dist=stats.expon, sparams=(0, scale_hat), plot=plt)
    plt.title(f"Q-Q plot (Exponential fit) — {vtype}")
    plt.show()

    # Plot 3 (optional): bootstrap distribution of D
    plt.figure()
    plt.hist(D_dist, bins=30)
    plt.axvline(D, linewidth=2, label="Observed D")
    plt.axvline(d_alpha, linewidth=2, label=f"Critical value (alpha={alpha})")
    plt.title(f"Bootstrap KS D distribution — {vtype}")
    plt.xlabel("D")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

summary = pd.DataFrame(summary_rows)
summary
