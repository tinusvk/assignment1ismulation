import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma as gamma_func  # <-- Gamma FUNCTION, not stats.gamma

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

def fit_weibull_2p(sample):
    """
    Fit 2-parameter Weibull (shape c, scale lambda), loc fixed at 0.
    Returns (c_hat, scale_hat).
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    # SciPy Weibull: weibull_min(c, loc, scale)
    c_hat, loc_hat, scale_hat = stats.weibull_min.fit(x, floc=0)
    return c_hat, scale_hat


print("\n=== 2-parameter Weibull MLE (loc fixed at 0) ===")

weibull_results = {}

for vtype, g in df.groupby("Visitor type label"):
    service = g["Service time"].to_numpy()
    service = service[np.isfinite(service) & (service > 0)]

    if len(service) < 5:
        continue

    c_hat, scale_hat = fit_weibull_2p(service)
    weibull_results[vtype] = (c_hat, scale_hat)

    # Weibull moments (with loc=0):
    # mean = scale * Gamma(1 + 1/c)
    # var  = scale^2 * (Gamma(1 + 2/c) - (Gamma(1 + 1/c))^2)
    mean = scale_hat * gamma_func(1.0 + 1.0 / c_hat)
    var = (scale_hat ** 2) * (gamma_func(1.0 + 2.0 / c_hat) - (gamma_func(1.0 + 1.0 / c_hat) ** 2))

    print(f"\n--- {vtype} ---")
    print(f"n = {len(service)}")
    print(f"Weibull MLE (2p): shape c = {c_hat:.6g}, scale = {scale_hat:.6g}")
    print(f"Mean = {mean:.3f}")
    print(f"Variance = {var:.3f}")

def plot_hist_with_weibull(service_times, c, scale, title, bins=40):
    x = np.asarray(service_times, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        print(f"Skipping plot (n={len(x)}): {title}")
        return

    plt.figure()
    plt.hist(x, bins=bins, density=True, alpha=0.5, label="Service time histogram (density)")

    x_grid = np.linspace(0, x.max(), 800)
    pdf = stats.weibull_min.pdf(x_grid, c=c, loc=0, scale=scale)

    plt.plot(x_grid, pdf, linewidth=2, label=f"Weibull PDF (c={c:.3g}, scale={scale:.3g})")

    plt.xlabel("Service time (seconds)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


for vtype, g in df.groupby("Visitor type label"):
    if vtype not in weibull_results:
        continue
    c_hat, scale_hat = weibull_results[vtype]
    service = g["Service time"].to_numpy()

    plot_hist_with_weibull(
        service_times=service,
        c=c_hat,
        scale=scale_hat,
        title=f"Service time histogram + Weibull fit — {vtype}",
        bins=40
    )

# ============================================================
# Weibull Q-Q plots using the fitted Weibull parameters
# ============================================================

def qqplot_weibull(service_times, c, scale, title):
    x = np.asarray(service_times, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)
    if n < 5:
        print(f"Skipping Q-Q plot (n={n}): {title}")
        return

    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.weibull_min.ppf(p, c=c, loc=0, scale=scale)

    plt.figure()
    plt.scatter(q, x, alpha=0.7)
    lo = min(q.min(), x.min())
    hi = max(q.max(), x.max())
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("Theoretical quantiles (Weibull)")
    plt.ylabel("Empirical quantiles (Service time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


for vtype, g in df.groupby("Visitor type label"):
    if vtype not in weibull_results:
        continue
    c_hat, scale_hat = weibull_results[vtype]
    service = g["Service time"].to_numpy()

    qqplot_weibull(
        service_times=service,
        c=c_hat,
        scale=scale_hat,
        title=f"Q-Q plot: Service times vs Weibull — {vtype}"
    )