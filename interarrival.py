import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FILE = "/Users/tijnvankruijsbergen/Documents/Econometrie/Bachelor 2/Blok 3/Simulatie/Assingment1/ass_part_1_dataset.xlsx"
SHEET = "Data"

df = pd.read_excel(FILE, sheet_name=SHEET)
df.columns = [c.strip() for c in df.columns]
df = df.sort_values(["Day", "Time of day in seconds"]).reset_index(drop=True)

# Interarrival times within each day
df["interarrival"] = df.groupby("Day")["Time of day in seconds"].diff()
sample = df["interarrival"].dropna()
sample = sample[sample > 0].to_numpy()
sample.sort()
N = len(sample)

print("N =", N)

# Fit exponential (scale = mean)
scale_hat = sample.mean()
print("Estimated scale =", scale_hat)

# Manual KS statistic
diff_cdf = []
for i in range(N):
    F_exp = stats.expon.cdf(sample[i], loc=0, scale=scale_hat)
    diff_plus = (i + 1)/N - F_exp
    diff_min  = F_exp - i/N
    diff_cdf.append(max(diff_plus, diff_min))
D = max(diff_cdf)
print("KS statistic D =", D)

# Simulate KS distribution (distribution-free) like your example
num_sim = 1000
D_dist_est = []
for _ in range(num_sim):
    u = np.random.uniform(size=N)
    u.sort()
    cur_D = max(max((j + 1)/N - u[j], u[j] - j/N) for j in range(N))
    D_dist_est.append(cur_D)

D_dist_est = np.array(D_dist_est)
alpha = 0.05
d_alpha = np.quantile(D_dist_est, 1 - alpha)
p_val = np.mean(D_dist_est >= D)

print("Critical value d_alpha =", d_alpha)
print("Approximate p-value   =", p_val)

# Exponential Q–Q plot (no statsmodels)
p = (np.arange(1, N + 1) - 0.5) / N
theo_q = stats.expon.ppf(p, loc=0, scale=scale_hat)

plt.figure()
plt.plot(theo_q, sample, 'o')
mn = min(theo_q.min(), sample.min())
mx = max(theo_q.max(), sample.max())
plt.plot([mn, mx], [mn, mx], 'k--')
plt.xlabel("Theoretical exponential quantiles")
plt.ylabel("Sample interarrival quantiles")
plt.title("Exponential Q–Q plot")
plt.tight_layout()
plt.show()
