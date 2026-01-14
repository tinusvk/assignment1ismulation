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

# -----------------------
# 1) Contextual information
# -----------------------
print("\n=== Contextual info ===")
print(f"Rows (visitors): {len(df)}")
print(f"Days in dataset (unique): {df['Day'].nunique()} -> {sorted(df['Day'].unique())[:10]}{'...' if df['Day'].nunique()>10 else ''}")
print("Visitor type counts:")
print(df["Visitor type label"].value_counts(dropna=False))
print("\nDesk opening hours per assignment: weekdays 09:00–17:00 (visitors arriving before 17:00 still served).")

# Opening times in seconds since midnight
OPEN_SEC = 9 * 3600
CLOSE_SEC = 17 * 3600

# Quick check: are arrivals mostly within open hours?
within_open = df["Time of day in seconds"].between(OPEN_SEC, CLOSE_SEC)
print(f"\nArrivals within 09:00–17:00: {within_open.mean()*100:.1f}%")

# -----------------------
# 2) Plot of arrivals over time (overall)
# -----------------------
# Count arrivals per day
arrivals_per_day = df.groupby("Day")["Visitor Number"].count().sort_index()

# Cumulative sum over days
cumulative_arrivals = arrivals_per_day.cumsum()

days = cumulative_arrivals.index.values
cum_counts = cumulative_arrivals.values

plt.figure()
plt.plot(days, cum_counts, marker="o")

# Axis formatting
plt.xlabel("Day")
plt.ylabel("Cumulative number of arrivals")
plt.title("Cumulative arrivals over time (all visitors)")
plt.xticks(days)  # only integer days

plt.tight_layout()
plt.show()

# -----------------------
# Arrivals per day
# -----------------------

days = arrivals_per_day.index.values
counts = arrivals_per_day.values

y_max = counts.max()
y_min = counts.min()

plt.figure()
plt.plot(days, counts, marker="o")

# Horizontal lines for min and max
plt.axhline(y=y_max, linestyle="--", linewidth=1, label=f"Maximum = {y_max}")
plt.axhline(y=y_min, linestyle="--", linewidth=1, label=f"Minimum = {y_min}")

# Axis formatting
plt.xlabel("Day")
plt.ylabel("Number of arrivals")
plt.title("Arrivals per day")
plt.xticks(days)   # force integer day ticks only

plt.legend()
plt.tight_layout()
plt.show()


# -----------------------
# Cumulative arrivals over time by visitor type (by day)
# -----------------------
plt.figure()

for visitor_type, g in df.groupby("Visitor type label"):
    # Count arrivals per day for this visitor type
    arrivals_per_day_type = (
        g.groupby("Day")["Visitor Number"]
        .count()
        .sort_index()
    )

    # Cumulative sum over days
    cumulative_arrivals_type = arrivals_per_day_type.cumsum()

    days = cumulative_arrivals_type.index.values
    cum_counts = cumulative_arrivals_type.values

    plt.plot(days, cum_counts, marker="o", label=str(visitor_type))

# Axis formatting
all_days = sorted(df["Day"].unique())
plt.xticks(all_days)

plt.xlabel("Day")
plt.ylabel("Cumulative number of arrivals")
plt.title("Cumulative arrivals over time by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 3) Descriptive statistics
# -----------------------
print("\n=== Descriptive statistics (overall) ===")
print(df[["Waiting time", "Service time"]].describe())

print("\n=== Descriptive statistics (by visitor type) ===")
print(df.groupby("Visitor type label")[["Waiting time", "Service time"]].describe())

# Also useful: percent waiting at all
print("\n=== Waiting incidence ===")
print("Share with zero waiting time:", (df["Waiting time"] == 0).mean())

# -----------------------
# 4) Histograms
# -----------------------

# -----------------------
# Histogram of arrivals by time of day (clock time)
# -----------------------

# Convert seconds since midnight to hours
time_in_hours = df["Time of day in seconds"] / 3600

plt.figure()
plt.hist(time_in_hours, bins=30)

# Format x-axis as clock time
hour_ticks = range(9, 18)  # 09:00 to 17:00
plt.xticks(hour_ticks, [f"{h:02d}:00" for h in hour_ticks])

plt.xlabel("Time of day")
plt.ylabel("Frequency")
plt.title("Histogram of arrival times during the day")

plt.tight_layout()
plt.show()

# 4b) Interarrival times (overall) - within each day
# (reset at day boundaries, so you don’t mix overnight gaps)
df["Interarrival"] = np.nan
for day, g in df.groupby("Day", sort=True):
    idx = g.index
    times = g["Time of day in seconds"].values
    inter = np.diff(times, prepend=np.nan)
    df.loc[idx, "Interarrival"] = inter

interarrival = df["Interarrival"].dropna()
interarrival = interarrival[interarrival >= 0]  # safety

plt.figure()
plt.hist(interarrival, bins=40)
plt.xlabel("Interarrival time (seconds) within day")
plt.ylabel("Frequency")
plt.title("Histogram: interarrival times (within-day)")
plt.tight_layout()
plt.show()

# Interarrival by type (optional)
plt.figure()
for t, g in df.groupby("Visitor type label"):
    g2 = g.copy()
    g2["Interarrival"] = np.nan
    for day, gg in g2.groupby("Day", sort=True):
        idx = gg.index
        times = gg["Time of day in seconds"].values
        inter = np.diff(times, prepend=np.nan)
        g2.loc[idx, "Interarrival"] = inter
    ia = g2["Interarrival"].dropna()
    ia = ia[ia >= 0]
    plt.hist(ia, bins=40, alpha=0.5, label=str(t))
plt.xlabel("Interarrival time (seconds) within day")
plt.ylabel("Frequency")
plt.title("Histogram: interarrival times by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

# 4c) Waiting time histogram
plt.figure()
plt.hist(df["Waiting time"], bins=40)
plt.xlabel("Waiting time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram: waiting times")
plt.tight_layout()
plt.show()

# 4d) Service time histogram
plt.figure()
plt.hist(df["Service time"], bins=40)
plt.xlabel("Service time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram: service times")
plt.tight_layout()
plt.show()

# Service time by type
plt.figure()
for t, g in df.groupby("Visitor type label"):
    plt.hist(g["Service time"], bins=40, alpha=0.5, label=str(t))
plt.xlabel("Service time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram: service times by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone. You now have: plots over time, descriptive statistics, histograms, plus contextual info in console.")

#---------------------------
#QQ Plots
#---------------------------
df["Interarrival"] = np.nan

for day, g in df.groupby("Day", sort=True):
    idx = g.index
    times = g["Time of day in seconds"].values
    inter = np.diff(times, prepend=np.nan)
    df.loc[idx, "Interarrival"] = inter

# Remove invalid values
interarrival = df["Interarrival"].dropna()
interarrival = interarrival[interarrival > 0]


lambda_hat = 1 / interarrival.mean()

print(f"Estimated arrival rate lambda = {lambda_hat:.6f} per second")


# -----------------------
# Q-Q plot: interarrival times vs exponential
# -----------------------

plt.figure()

stats.probplot(
    interarrival,
    dist=stats.expon,
    sparams=(0, 1/lambda_hat),
    plot=plt
)

plt.title("Q-Q plot: interarrival times vs exponential distribution")
plt.tight_layout()
plt.show()


# -----------------------
# Q-Q plots by visitor type
# -----------------------

for visitor_type, g in df.groupby("Visitor type label"):
    g = g.copy()
    g["Interarrival"] = np.nan

    for day, gg in g.groupby("Day", sort=True):
        idx = gg.index
        times = gg["Time of day in seconds"].values
        inter = np.diff(times, prepend=np.nan)
        g.loc[idx, "Interarrival"] = inter

    ia = g["Interarrival"].dropna()
    ia = ia[ia > 0]

    lambda_hat_type = 1 / ia.mean()

    plt.figure()
    stats.probplot(
        ia,
        dist=stats.expon,
        sparams=(0, 1/lambda_hat_type),
        plot=plt
    )

    plt.title(f"Q-Q plot: interarrival times ({visitor_type})")
    plt.tight_layout()
    plt.show()


service = df["Service time"].values
service = service[service > 0]   # keep positive values only

# Descriptive statistics
print("Service time statistics:")
print(stats.describe(service))

# Histogram
plt.figure()
plt.hist(service, bins=40)
plt.xlabel("Service time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of service times")
plt.tight_layout()
plt.show()

loc, scale = stats.expon.fit(service, floc=0)

plt.figure()
stats.probplot(service, dist=stats.expon, sparams=(loc, scale), plot=plt)
plt.title("Q-Q plot: service times vs exponential")
plt.tight_layout()
plt.show()

# Fit gamma distribution (loc fixed at 0)
a, loc, scale = stats.gamma.fit(service, floc=0)

plt.figure()
stats.probplot(service, dist=stats.gamma, sparams=(a, loc, scale), plot=plt)
plt.title("Q-Q plot: service times vs gamma")
plt.tight_layout()
plt.show()

# Fit lognormal distribution (loc fixed at 0)
s, loc, scale = stats.lognorm.fit(service, floc=0)

plt.figure()
stats.probplot(service, dist=stats.lognorm, sparams=(s, loc, scale), plot=plt)
plt.title("Q-Q plot: service times vs lognormal")
plt.tight_layout()
plt.show()


# -----------------------
# Arrival times during the day per visitor type
# -----------------------

OPEN_SEC = 9 * 3600
CLOSE_SEC = 17 * 3600

# Keep arrivals in opening hours (optional but usually desired)
df_open = df[df["Time of day in seconds"].between(OPEN_SEC, CLOSE_SEC)].copy()

# Convert to hours since midnight (clock time)
df_open["Arrival hour"] = df_open["Time of day in seconds"] / 3600

# Also convenient: hour bin as integer hour (9..17)
df_open["Arrival hour bin"] = np.floor(df_open["Arrival hour"]).astype(int)

# -----------------------
# 1) Histogram overlay: arrival times by visitor type
# -----------------------
plt.figure()
for t, g in df_open.groupby("Visitor type label"):
    plt.hist(g["Arrival hour"], bins=30, alpha=0.5, label=str(t))

hour_ticks = range(9, 18)
plt.xticks(hour_ticks, [f"{h:02d}:00" for h in hour_ticks])
plt.xlabel("Arrival time of day")
plt.ylabel("Frequency")
plt.title("Histogram: arrival times during the day by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 2) Normalized histogram overlay (compare shapes)
# -----------------------
plt.figure()
for t, g in df_open.groupby("Visitor type label"):
    plt.hist(g["Arrival hour"], bins=30, density=True, alpha=0.5, label=str(t))

plt.xticks(hour_ticks, [f"{h:02d}:00" for h in hour_ticks])
plt.xlabel("Arrival time of day")
plt.ylabel("Density")
plt.title("Normalized histogram: arrival times by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 3) Arrivals per hour (counts) by visitor type
# -----------------------
# Make sure all hours 9..17 are present for each type (fill missing with 0)
all_hours = list(range(9, 18))

counts_hour_type = (
    df_open
    .groupby(["Visitor type label", "Arrival hour bin"])["Visitor Number"]
    .count()
    .rename("count")
    .reset_index()
)

plt.figure()
for t, g in counts_hour_type.groupby("Visitor type label"):
    # Align to all hours
    y = pd.Series(g["count"].values, index=g["Arrival hour bin"]).reindex(all_hours, fill_value=0)
    plt.plot(all_hours, y.values, marker="o", label=str(t))

plt.xticks(all_hours, [f"{h:02d}:00" for h in all_hours])
plt.xlabel("Hour of day")
plt.ylabel("Number of arrivals")
plt.title("Arrivals per hour (09:00–17:00) by visitor type")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# Cumulative arrivals vs time of day (per visitor type)
# -----------------------

# Keep arrivals within opening hours
df_open = df[df["Time of day in seconds"].between(OPEN_SEC, CLOSE_SEC)].copy()

# Convert to hours since midnight
df_open["Arrival hour"] = df_open["Time of day in seconds"] / 3600

plt.figure()

for visitor_type, g in df_open.groupby("Visitor type label"):
    # Sort by time of day
    g = g.sort_values("Arrival hour").reset_index(drop=True)

    # Cumulative arrivals (adds up correctly!)
    g["Cumulative arrivals"] = np.arange(1, len(g) + 1)

    # Plot
    plt.plot(
        g["Arrival hour"],
        g["Cumulative arrivals"],
        label=str(visitor_type)
    )

# Axis formatting
plt.xticks(range(9, 18), [f"{h:02d}:00" for h in range(9, 18)])
plt.xlabel("Time of day")
plt.ylabel("Cumulative number of arrivals")
plt.title("Cumulative arrivals during the day by visitor type")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 8) Overlay the TWO fitted Gamma distributions (from screenshot)
#    over service-time histograms per visitor type
# ============================================================

gamma_params_from_screenshot = {
    "Employee": {"shape": 69.1838, "loc": -900.691, "scale": 18.3022},
    "Student":  {"shape": 109.123, "loc": -1293.29, "scale": 15.1885},
}

def plot_hist_with_given_gamma(service_times, shape, loc, scale, title, bins=40):
    """
    Histogram (density=True) of service times with OVERLAY
    of Gamma(shape, loc, scale) using GIVEN parameters.
    """
    x = np.asarray(service_times, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        print(f"Skipping plot (n={len(x)}): {title}")
        return

    plt.figure()

    # Histogram in density units so PDF matches scale
    plt.hist(x, bins=bins, density=True, alpha=0.5, label="Service time histogram (density)")

    # PDF grid over observed range
    x_grid = np.linspace(x.min(), x.max(), 600)
    pdf = stats.gamma.pdf(x_grid, a=shape, loc=loc, scale=scale)

    # Overlay Gamma PDF
    plt.plot(
        x_grid,
        pdf,
        linewidth=2,
        label=f"Gamma PDF (shape={shape:.4g}, loc={loc:.4g}, scale={scale:.4g})"
    )

    plt.xlabel("Service time (seconds)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Make the two plots (Employee and Student) ---
for vtype, g in df.groupby("Visitor type label"):
    if vtype not in gamma_params_from_screenshot:
        continue

    params = gamma_params_from_screenshot[vtype]
    service = g["Service time"].to_numpy()

    plot_hist_with_given_gamma(
        service_times=service,
        shape=params["shape"],
        loc=params["loc"],
        scale=params["scale"],
        title=f"Service time histogram + fitted Gamma (given params) — {vtype}",
        bins=40
    )

# ============================================================
# 9) Q-Q plots using the SAME Gamma parameters (from screenshot)
# ============================================================

def qqplot_given_gamma(service_times, shape, loc, scale, title):
    """
    Q-Q plot of service_times against Gamma(shape, loc, scale)
    using GIVEN parameters (no refitting).
    """
    x = np.asarray(service_times, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    n = len(x)

    if n < 5:
        print(f"Skipping Q-Q plot (n={n}): {title}")
        return

    # plotting positions
    p = (np.arange(1, n + 1) - 0.5) / n

    # theoretical quantiles from the given Gamma
    q = stats.gamma.ppf(p, a=shape, loc=loc, scale=scale)

    plt.figure()
    plt.scatter(q, x, alpha=0.7)
    lo = min(q.min(), x.min())
    hi = max(q.max(), x.max())
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("Theoretical quantiles (Gamma)")
    plt.ylabel("Empirical quantiles (Service time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Make the two QQ plots (Employee and Student) ---
for vtype, g in df.groupby("Visitor type label"):
    if vtype not in gamma_params_from_screenshot:
        continue

    params = gamma_params_from_screenshot[vtype]
    service = g["Service time"].to_numpy()

    qqplot_given_gamma(
        service_times=service,
        shape=params["shape"],
        loc=params["loc"],
        scale=params["scale"],
        title=f"Q-Q plot: Service times vs Gamma (given params) — {vtype}"
    )
# ============================================================
# Overlay 2-parameter Gamma distributions (loc = 0)
# using the ESTIMATED PARAMETERS from the screenshot
# ============================================================

gamma_2p_params = {
    "Employee": {"shape": 4.56327, "scale": 80.1021},
    "Student":  {"shape": 3.87712, "scale": 93.9189},
}

def plot_hist_with_gamma_2p(service_times, shape, scale, title, bins=40):
    """
    Histogram (density=True) of service times with OVERLAY
    of 2-parameter Gamma(shape, scale), loc fixed at 0.
    """
    x = np.asarray(service_times, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) < 5:
        print(f"Skipping plot (n={len(x)}): {title}")
        return

    plt.figure()

    # Histogram as density
    plt.hist(
        x,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Service time histogram (density)"
    )

    # Grid for Gamma PDF (only positive support)
    x_grid = np.linspace(0, x.max(), 600)
    pdf = stats.gamma.pdf(x_grid, a=shape, loc=0, scale=scale)

    # Overlay Gamma PDF
    plt.plot(
        x_grid,
        pdf,
        linewidth=2,
        label=f"Gamma PDF (shape={shape:.4g}, scale={scale:.4g})"
    )

    plt.xlabel("Service time (seconds)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Create plots per visitor type (2-parameter Gamma)
# ------------------------------------------------------------
for vtype, g in df.groupby("Visitor type label"):
    if vtype not in gamma_2p_params:
        continue

    params = gamma_2p_params[vtype]
    service = g["Service time"].to_numpy()

    plot_hist_with_gamma_2p(
        service_times=service,
        shape=params["shape"],
        scale=params["scale"],
        title=f"Service time histogram + 2p Gamma fit — {vtype}",
        bins=40
    )


















