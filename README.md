# Improving Kitchen Prep Time (KPT) Prediction
Signal Enrichment via Label Correction, Hidden Load Estimation & WiFi-CSI Sensing

## Overview

Food delivery platforms rely on accurate **Kitchen Prep Time (KPT)** prediction to dispatch riders efficiently.  
However, real-world restaurant operations introduce two major sources of error:

1. **Hidden kitchen load** Platforms only observe their own digital orders, while restaurants simultaneously handle **dine-in customers and competitor platform orders**.

2. **Biased human inputs** Restaurant staff often delay pressing the **Food Order Ready (FOR)** button until the rider arrives to prevent food from sitting idle.  
   This produces **incorrect training labels** that inflate historical preparation times.

These issues cause systematic errors in **ETA prediction**, increasing rider wait time and reducing delivery efficiency.

This project proposes an **explainable signal-enrichment pipeline** that improves prediction accuracy by:

- correcting biased training labels  
- estimating hidden kitchen load using residual inference  
- detecting real-time physical rush using **WiFi Channel State Information (CSI)** The system was evaluated using a **large-scale simulation representing ~1M orders/day**.

---

# Core Problem

Most baseline KPT models rely on a simple linear relationship between order count and preparation time:

$$KPT_{pred} = \alpha \times N_{visible}$$

Where:

- $N_{visible}$ = platform orders visible to the system  
- $\alpha$ = average prep time per item  

This fails because it ignores **hidden kitchen load**.

The true relationship is closer to:

$$KPT_{true} = \alpha (N_{visible} + N_{hidden})$$

Where:

- $N_{hidden}$ represents **dine-in and competitor orders**.

---

# System Architecture

The proposed pipeline improves prediction accuracy through three stages:

1. **Label Correction**
2. **Hidden Load Estimation**
3. **Real-Time Rush Detection (WiFi CSI)**

Each stage progressively improves the quality of signals used for prediction.

---

# Step 1 — Algorithmic Label Correction

Historical training data is often corrupted by human behavior.

Define:

$$\Delta = T_{FOR} - T_{arrival}$$

Where:

- $T_{FOR}$ = timestamp when the merchant marks food ready  
- $T_{arrival}$ = rider arrival time  

### Late Marking Bias

If merchants delay marking food ready until the rider arrives:

$$\Delta \leq \tau$$

Where $\tau = 1.5$ minutes.

These samples are flagged as **rider-influenced**.

Corrected label:

$$KPT_{corrected} = T_{arrival} - T_{order}$$

This removes artificial delays in historical labels.

---

### Batch Processing Bias

During rush periods, staff often mark multiple orders ready simultaneously.

Detection condition:

$$N_{same\_timestamp} > 1$$

These samples are excluded from training because they represent **backlog clearing rather than real cooking completion**.

---

### Honest Signals

If the rider arrives **before the food is ready**:

$$\Delta > \tau$$

These samples are treated as **high-confidence ground truth**.

---

# Step 2 — Hidden Kitchen Load Estimation

Even after correcting labels, the platform cannot observe **dine-in or competitor orders**.

We estimate hidden load using **historical residuals**.

Residual:

$$R = KPT_{true} - KPT_{expected}$$

Where:

$$KPT_{expected} = \alpha \times N_{visible}$$

The hidden load can be estimated as:

$$N_{hidden} = \frac{R}{\alpha}$$

To reduce sensitivity to outliers we use **median aggregation**:

$$\hat{N}_{hidden} = median(N_{hidden})$$

This captures the **typical hidden load** without being skewed by rare extreme delays.

---

# Step 3 — Weather-Aware Context

Weather significantly changes order dynamics.

- Bad weather → more delivery orders  
- Bad weather → fewer dine-in customers  

We adjust weights dynamically:

$$KPT = \alpha (w_1 N_{visible} + w_2 N_{hidden})$$

Where:

- $w_1 > 1$ during bad weather  
- $w_2 < 1$ during bad weather  

---

# Step 4 — Real-Time Rush Detection using WiFi CSI

Statistical estimation cannot capture **sudden dine-in rushes**.

We propose deploying **WiFi CSI sensors** for high-volume restaurants.

### Hardware

- Raspberry Pi Zero / Raspberry Pi 4  
- Nexmon firmware  
- commodity WiFi chipset  

Instead of counting devices, CSI measures **signal distortion caused by human bodies**.

We compute:

$$CSI_{score} = Var(|CSI_{amplitude}|)$$

Higher variance indicates **more physical occupancy**.

---

# Final Prediction Equation

Combining all signals:

$$KPT_{final} = \alpha \left( w_1 N_{visible} + w_2 \hat{N}_{hidden} + \gamma CSI_{score} \right)$$

Where:

- $N_{visible}$ = platform orders  
- $\hat{N}_{hidden}$ = estimated hidden load  
- $CSI_{score}$ = real-time occupancy signal  
- $\gamma$ converts physical crowd signal into equivalent kitchen load.

---

# Deployment Strategy

### Phase 1 — Software Only

Uses:

- historical logs  
- weather APIs  

Advantages:

- deployable to **300K+ merchants instantly**
- zero hardware cost.

---

### Phase 2 — Hardware Augmentation

Deploy **Raspberry Pi CSI sensors** to high-volume restaurants.

Benefits:

- detects real-time dine-in rush
- improves KPT prediction accuracy.

---

### Phase 3 — POS Integration

Enterprise chains can share:

- real dine-in order counts  
- table occupancy  
- kitchen queue length  

Hidden load becomes:

$$N_{hidden} = N_{POS}$$

This removes estimation error entirely.

---

# Preventing Feedback Loops

Continuous real-time updates can create prediction drift.

Solution:

- prediction parameters remain fixed during the day  
- nightly batch jobs evaluate rider wait metrics  
- parameters are updated **once per day**.

---

# Simulation Setup

Evaluation used a synthetic dataset with:

- **100 merchants**
- **200 orders per merchant**
- full-day demand cycles (lunch and dinner peaks)

Merchant behavior distribution:

| Merchant Type | Share |
|---------------|------|
| Honest | 30% |
| Rider-influenced | 50% |
| Batch updates | 20% |

Additional parameters:

- hidden load range: **0.1 – 3.0 orders**
- **20%** of time slots with bad weather.

Correlation between inferred and true hidden load:

$$r = 0.290, \quad p = 0.003$$

(statistically significant)

---

# Results

| Metric | Baseline | Approach 1 | Approach 2 | Approach 3 |
|------|------|------|------|------|
| Avg Rider Wait | 2.60 min | 2.21 (-15%) | 1.69 (-35%) | 0.49 (-81%) |
| ETA Error P50 | 2.96 | 2.76 | 2.63 | 1.92 |
| ETA Error P90 | 6.34 | 5.93 (-6%) | 5.55 (-12%) | 3.98 (-37%) |

Large-scale simulation impact:

- **15,000+ rider hours saved per day** using CSI sensing  
- **35,000+ rider hours saved per day** with POS integration.

A slight increase in rider idle time is intentional — the system prefers **food waiting briefly instead of riders waiting**.

---

# Key Insight

The biggest limitation in KPT prediction is **signal quality rather than model complexity**.

By improving:

- label quality  
- load visibility  
- real-time physical signals  

we significantly improve ETA prediction while maintaining **full explainability**.
