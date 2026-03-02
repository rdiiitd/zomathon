import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

#parameters 
N_MERCHANTS = 100
ORDERS_PER_MERCHANT = 200
BASE_KPT_RANGE = (10, 18)
ALPHA = 1.5
NOISE_STD = 2
THRESHOLD = 1.5

GAMMA = 1.5

WEATHER_DIGITAL_BOOST = 0.4
WEATHER_HIDDEN_DAMPEN = 0.5
WEATHER_BAD_PROB      = 0.2

MERCHANT_TYPES = ["honest", "rider_influenced", "batch"]

#stimulating merchants 
rng = np.random.default_rng(42)

merchants = []
for m in range(N_MERCHANTS):
    merchants.append({
        "merchant_id": m,
        "base_kpt": rng.uniform(*BASE_KPT_RANGE),
        "merchant_type": np.random.choice(MERCHANT_TYPES, p=[0.3, 0.5, 0.2]),
        "hidden_load_rate": rng.uniform(0.1, 3.0),
        "wifi_capacity": int(rng.integers(20, 80))
    })
merchant_df = pd.DataFrame(merchants)

#stimulating orders 
records = []
for _, merch in merchant_df.iterrows():
    order_times = np.sort(rng.uniform(0, 1440, ORDERS_PER_MERCHANT))
    for t in order_times:
        records.append({
            "merchant_id": merch["merchant_id"],
            "base_kpt": merch["base_kpt"],
            "merchant_type": merch["merchant_type"],
            "hidden_load_rate": merch["hidden_load_rate"],
            "wifi_capacity": merch["wifi_capacity"],
            "order_time": t
        })

df = pd.DataFrame(records).reset_index(drop=True)


def tod_multiplier(t):
    if 720 <= t <= 840:     return 1.5
    elif 1080 <= t <= 1260: return 2.0
    else:                   return 1.0

df["tod_multiplier"] = df["order_time"].apply(tod_multiplier)

#weather signal
time_slots  = np.arange(0, 1440, 60)
bad_weather = rng.random(len(time_slots)) < WEATHER_BAD_PROB
weather_map = dict(zip(time_slots, bad_weather))

def get_weather(t):
    return weather_map.get(int(t // 60) * 60, False)

df["is_bad_weather"]             = df["order_time"].apply(get_weather)
df["weather_digital_multiplier"] = np.where(df["is_bad_weather"], 1.0 + WEATHER_DIGITAL_BOOST, 1.0)
df["weather_hidden_multiplier"]  = np.where(df["is_bad_weather"], WEATHER_HIDDEN_DAMPEN, 1.0)

#zomato load 
window = 20
df = df.sort_values(["merchant_id", "order_time"]).reset_index(drop=True)

zomato_loads = []
for mid, group in df.groupby("merchant_id"):
    times = group["order_time"].values
    load  = np.array([np.sum((times >= t - window) & (times < t)) for t in times])
    zomato_loads.extend(load)

df["zomato_active_load"] = zomato_loads

#hidden load
df["hidden_load"] = rng.poisson(
    (df["hidden_load_rate"] * df["tod_multiplier"] * df["weather_hidden_multiplier"]).clip(0.01)
)
df["total_load"] = df["zomato_active_load"] + df["hidden_load"]

#true kpt 
df["true_kpt"] = (
    df["base_kpt"] + ALPHA * df["total_load"] + rng.normal(0, NOISE_STD, len(df))
).clip(1)
df["true_ready_time"] = df["order_time"] + df["true_kpt"]

#baseline predicted kpt 
df["predicted_kpt"]      = (df["base_kpt"] + ALPHA * df["zomato_active_load"]).clip(1)
df["rider_arrival_time"] = df["order_time"] + df["predicted_kpt"]
df["pickup_time"]        = df[["true_ready_time", "rider_arrival_time"]].max(axis=1)

# merchant FOR
def compute_for_time(row):
    if row["merchant_type"] == "honest":
        return row["true_ready_time"]
    elif row["merchant_type"] == "rider_influenced":
        return max(row["true_ready_time"], row["rider_arrival_time"])
    elif row["merchant_type"] == "batch":
        return np.ceil(row["true_ready_time"] / 5) * 5

df["for_time"]     = df.apply(compute_for_time, axis=1)
df["observed_kpt"] = df["for_time"] - df["order_time"]

#FOR correction
df["for_minus_rider"]      = df["for_time"] - df["rider_arrival_time"]
df["corrected_ready_time"] = np.where(
    df["for_minus_rider"] <= THRESHOLD,
    df["rider_arrival_time"],
    df["for_time"]
)
df["corrected_kpt"]  = df["corrected_ready_time"] - df["order_time"]
df["was_corrected"]  = df["for_minus_rider"] <= THRESHOLD
df["reliable_label"] = df["was_corrected"] | (df["merchant_type"] == "honest")

#CSI
csi_values = []
for mid, group in df.groupby("merchant_id"):
    n         = len(group)
    raw_var   = (0.02 + group["hidden_load"].values * 0.04 * group["tod_multiplier"].values
                 + rng.normal(0, 0.005, n)).clip(0)
    mn, mx    = raw_var.min(), raw_var.max()
    csi_values.extend(((raw_var - mn) / (mx - mn + 1e-6)).clip(0, 1))

df["csi_rush_score"] = csi_values


implied_hiddens = {}

for mid, group in df.groupby("merchant_id"):
    reliable = group[group["reliable_label"]]
    if len(reliable) < 10:
        reliable = group

    expected_kpt   = reliable["base_kpt"] + ALPHA * reliable["zomato_active_load"]
    residual       = reliable["corrected_kpt"] - expected_kpt
    implied_hiddens[mid] = residual.median() / ALPHA

df["implied_hidden_load"] = df["merchant_id"].map(implied_hiddens)

# CASE-1 FOR correction
df["t1_effective_load"] = df["zomato_active_load"] + df["implied_hidden_load"].clip(0)
df["t1_predicted_kpt"]  = (df["base_kpt"] + ALPHA * df["t1_effective_load"]).clip(1)
df["t1_rider_arrival"]  = df["order_time"] + df["t1_predicted_kpt"]
df["t1_pickup_time"]    = df[["true_ready_time", "t1_rider_arrival"]].max(axis=1)

#validation

actual_hidden_per_merchant = df.groupby("merchant_id")["hidden_load_rate"].mean()
implied_hidden_series      = pd.Series(implied_hiddens, name="implied_hidden")

validation_df = pd.DataFrame({
    "actual_avg_hidden": actual_hidden_per_merchant,
    "implied_hidden":    implied_hidden_series
})

corr, pval = stats.pearsonr(
    validation_df["actual_avg_hidden"],
    validation_df["implied_hidden"]
)

print("\n" + "=" * 55)
print("  VALIDATION: Implied Hidden Load vs Actual Hidden Load")
print("=" * 55)
print(f"  Pearson Correlation : {corr:.3f}")
print(f"  p-value             : {pval:.4f}")
print("=" * 55)

#case-2: IMPROVED PREDICTED KPT

df["weather_adjusted_zomato"] = df["zomato_active_load"] * df["weather_digital_multiplier"]

df["effective_load"] = (
    df["weather_adjusted_zomato"]
    + df["implied_hidden_load"].clip(0)
    + GAMMA * df["csi_rush_score"]
)

df["improved_predicted_kpt"] = (df["base_kpt"] + ALPHA * df["effective_load"]).clip(1)
df["improved_rider_arrival"]  = df["order_time"] + df["improved_predicted_kpt"]
df["improved_pickup_time"]    = df[["true_ready_time", "improved_rider_arrival"]].max(axis=1)

# case-3: POS integration — actual hidden load

df["t3_effective_load"] = (
    df["weather_adjusted_zomato"]
    + df["hidden_load"]
    + GAMMA * df["csi_rush_score"]
)
df["t3_predicted_kpt"]  = (df["base_kpt"] + ALPHA * df["t3_effective_load"]).clip(1)
df["t3_rider_arrival"]  = df["order_time"] + df["t3_predicted_kpt"]
df["t3_pickup_time"]    = df[["true_ready_time", "t3_rider_arrival"]].max(axis=1)

#metrics
def get_metrics(pred_kpt, rider_arr, pickup_t):
    wait = (pickup_t - rider_arr).mean()
    eta  = np.abs(pred_kpt - df["true_kpt"]).mean()
    p90  = np.percentile(np.abs(pred_kpt - df["true_kpt"]), 90)
    idle = np.maximum(pred_kpt - df["true_kpt"], 0).mean()
    return wait, eta, p90, idle

b_wait,  b_eta,  b_p90,  b_idle  = get_metrics(df["predicted_kpt"],      df["rider_arrival_time"],  df["pickup_time"])
t1_wait, t1_eta, t1_p90, t1_idle = get_metrics(df["t1_predicted_kpt"],   df["t1_rider_arrival"],    df["t1_pickup_time"])
t2_wait, t2_eta, t2_p90, t2_idle = get_metrics(df["improved_predicted_kpt"], df["improved_rider_arrival"], df["improved_pickup_time"])
t3_wait, t3_eta, t3_p90, t3_idle = get_metrics(df["t3_predicted_kpt"],   df["t3_rider_arrival"],    df["t3_pickup_time"])

orders_per_day = 1_000_000

print("\n" + "=" * 75)
print(f"  {'Metric':<28} {'Baseline':>9} {'case 1':>9} {'case 2':>9} {'case 3':>9}")
print("=" * 75)
print(f"  {'Avg Rider Wait (min)':<28} {b_wait:>9.2f} {t1_wait:>9.2f} {t2_wait:>9.2f} {t3_wait:>9.2f}")
print(f"  {'ETA Error P50 (min)':<28} {b_eta:>9.2f}  {t1_eta:>9.2f} {t2_eta:>9.2f} {t3_eta:>9.2f}")
print(f"  {'ETA Error P90 (min)':<28} {b_p90:>9.2f}  {t1_p90:>9.2f} {t2_p90:>9.2f} {t3_p90:>9.2f}")
print(f"  {'Rider Idle Time (min)':<28} {b_idle:>9.2f} {t1_idle:>9.2f} {t2_idle:>9.2f} {t3_idle:>9.2f}")
print("=" * 75)
print(f"\n  Rider hours saved daily vs baseline ({orders_per_day:,} orders/day):")
print(f"  case 1 : {(b_wait - t1_wait) * orders_per_day / 60:>10,.0f} hours")
print(f"  case 2 : {(b_wait - t2_wait) * orders_per_day / 60:>10,.0f} hours")
print(f"  case 3 : {(b_wait - t3_wait) * orders_per_day / 60:>10,.0f} hours")
print("=" * 75)

# Weather
bad_weather_mask  = df["is_bad_weather"]
good_weather_mask = ~df["is_bad_weather"]

bw_b = (df[bad_weather_mask]["pickup_time"]          - df[bad_weather_mask]["rider_arrival_time"]).mean()
bw_i = (df[bad_weather_mask]["improved_pickup_time"] - df[bad_weather_mask]["improved_rider_arrival"]).mean()
gw_b = (df[good_weather_mask]["pickup_time"]         - df[good_weather_mask]["rider_arrival_time"]).mean()
gw_i = (df[good_weather_mask]["improved_pickup_time"]- df[good_weather_mask]["improved_rider_arrival"]).mean()

print("\n" + "=" * 55)
print("  WEATHER BREAKDOWN — Avg Rider Wait (min)")
print("=" * 55)
print(f"  {'Condition':<20} {'Baseline':>8} {'case 2':>8} {'Saved':>8}")
print("-" * 55)
print(f"  {'Bad Weather':<20} {bw_b:>8.2f} {bw_i:>8.2f} {bw_b-bw_i:>8.2f}")
print(f"  {'Good Weather':<20} {gw_b:>8.2f} {gw_i:>8.2f} {gw_b-gw_i:>8.2f}")
print(f"  {'% orders bad weather':<20} {bad_weather_mask.mean()*100:>7.1f}%")
print("=" * 55)


# NIGHTLY FEEDBACK LOOP

df["improved_rider_wait"] = df["improved_pickup_time"] - df["improved_rider_arrival"]
df["improved_idle"]       = np.maximum(df["improved_rider_arrival"] - df["true_ready_time"], 0)

WAIT_THRESHOLD = 1.0
IDLE_THRESHOLD = 1.0

feedback = df.groupby("merchant_id")[["improved_rider_wait", "improved_idle"]].apply(
    lambda g: pd.Series({
        "avg_improved_wait": g["improved_rider_wait"].mean(),
        "avg_improved_idle": g["improved_idle"].mean(),
        "recommendation": (
            "increase_multiplier" if g["improved_rider_wait"].mean() > WAIT_THRESHOLD
            else "decrease_multiplier" if g["improved_idle"].mean() > IDLE_THRESHOLD
            else "stable"
        )
    }), include_groups=False
).reset_index()

print("\n" + "=" * 55)
print("  NIGHTLY FEEDBACK LOOP — Next Period Adjustments")
print("=" * 55)
print(feedback["recommendation"].value_counts().to_string())
print(f"\n  Avg residual wait  after improvement: {df['improved_rider_wait'].mean():.3f} min")
print(f"  Avg residual idle  after improvement: {df['improved_idle'].mean():.3f} min")
print("=" * 55)


# VISUALISATION

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("KPT Simulation: Baseline vs Improved", fontsize=14, fontweight="bold")

# Plot 1: Rider Wait Distribution
axes[0,0].hist(df["pickup_time"] - df["rider_arrival_time"],
               bins=40, alpha=0.6, label="Baseline", color="red")
axes[0,0].hist(df["improved_pickup_time"] - df["improved_rider_arrival"],
               bins=40, alpha=0.6, label="Tier 2", color="green")
axes[0,0].set_title("Rider Wait Distribution"); axes[0,0].set_xlabel("Minutes"); axes[0,0].legend()

# Plot 2: ETA Error Distribution
axes[0,1].hist(df["predicted_kpt"] - df["true_kpt"],
               bins=40, alpha=0.6, label="Baseline", color="red")
axes[0,1].hist(df["improved_predicted_kpt"] - df["true_kpt"],
               bins=40, alpha=0.6, label="Tier 2", color="green")
axes[0,1].set_title("ETA Error Distribution"); axes[0,1].set_xlabel("Minutes"); axes[0,1].legend()

# Plot 3: CSI Rush Score vs Hidden Load
axes[0,2].scatter(df["hidden_load"], df["csi_rush_score"], alpha=0.3, s=10, color="purple")
csi_corr, csi_p = stats.pearsonr(df["hidden_load"], df["csi_rush_score"])
axes[0,2].set_title(f"CSI Rush Score vs Hidden Load\n(r={csi_corr:.2f}, p={csi_p:.4f})")
axes[0,2].set_xlabel("Actual Hidden Load"); axes[0,2].set_ylabel("CSI Rush Score (0-1)")

# Plot 4: Per-Merchant Rider Wait
mw = (df.assign(bw=df["pickup_time"]-df["rider_arrival_time"],
                iw=df["improved_pickup_time"]-df["improved_rider_arrival"])
      .groupby("merchant_id").agg(baseline=("bw","mean"), improved=("iw","mean")).reset_index())
x = np.arange(len(mw))
axes[0,3].bar(x-0.2, mw["baseline"], 0.4, label="Baseline", color="red",   alpha=0.7)
axes[0,3].bar(x+0.2, mw["improved"], 0.4, label="Tier 2",   color="green", alpha=0.7)
axes[0,3].set_title("Per-Merchant Avg Rider Wait"); axes[0,3].set_xlabel("Merchant ID"); axes[0,3].legend()

# Plot 5: Validation
axes[1,0].scatter(validation_df["actual_avg_hidden"], validation_df["implied_hidden"],
                  color="steelblue", s=80, zorder=3)
m_fit, b_fit = np.polyfit(validation_df["actual_avg_hidden"], validation_df["implied_hidden"], 1)
xl = np.linspace(validation_df["actual_avg_hidden"].min(), validation_df["actual_avg_hidden"].max(), 100)
axes[1,0].plot(xl, m_fit*xl+b_fit, color="orange", linewidth=2, label=f"r = {corr:.2f}")
axes[1,0].set_title("Validation: Implied vs Actual Hidden Load")
axes[1,0].set_xlabel("Actual hidden_load_rate"); axes[1,0].set_ylabel("Implied Hidden Load")
axes[1,0].legend()
axes[1,0].annotate(f"Pearson r = {corr:.3f}\np = {pval:.4f}", xy=(0.05,0.85),
    xycoords="axes fraction", fontsize=10, color="darkgreen",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

# Plot 6: Summary Metrics
metrics_names = ["Rider Wait", "ETA P50", "ETA P90", "Idle Time"]
baseline_vals = [b_wait,  b_eta,  b_p90,  b_idle]
improved_vals = [t2_wait, t2_eta, t2_p90, t2_idle]
x2 = np.arange(len(metrics_names))
axes[1,1].bar(x2-0.2, baseline_vals, 0.4, label="Baseline", color="red",   alpha=0.7)
axes[1,1].bar(x2+0.2, improved_vals, 0.4, label="Tier 2",   color="green", alpha=0.7)
axes[1,1].set_title("Key Metrics: Baseline vs Tier 2")
axes[1,1].set_xticks(x2); axes[1,1].set_xticklabels(metrics_names)
axes[1,1].set_ylabel("Minutes"); axes[1,1].legend()

# Plot 7: case COMPARISON — Rider Wait
tier_labels  = ["Baseline", "Case 1\n(Stats only)", "case 2\n(+Weather+CSI)", "case 3\n(+POS)"]
tier_colors  = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD"]
wait_vals    = [b_wait, t1_wait, t2_wait, t3_wait]
bars = axes[1,2].bar(tier_labels, wait_vals, color=tier_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
axes[1,2].set_title("Rider Wait by Deployment case", fontweight="bold")
axes[1,2].set_ylabel("Avg Rider Wait (min)")
for bar, val in zip(bars, wait_vals):
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1,2].set_ylim(0, max(wait_vals)*1.2)

# Plot 8: Cases COMPARISON — Rider Hours Saved
hours_saved = [0,
               (b_wait-t1_wait)*orders_per_day/60,
               (b_wait-t2_wait)*orders_per_day/60,
               (b_wait-t3_wait)*orders_per_day/60]
bars = axes[1,3].bar(tier_labels, hours_saved, color=tier_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
axes[1,3].set_title("Rider Hours Saved Daily\n(vs Baseline, 1M orders/day)", fontweight="bold")
axes[1,3].set_ylabel("Hours / Day")
for bar, val in zip(bars, hours_saved):
    axes[1,3].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                   f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1,3].set_ylim(0, max(hours_saved)*1.25)

plt.tight_layout()
plt.savefig("simulation.png", dpi=150)
plt.show()
print("\nPlot saved.")
