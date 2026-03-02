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
