# Experiment 2: Selective Participation in Federated Learning

---

## 1. Experiment Name

**Task-Aware Selective Participation for Heterogeneous Federated Learning**

---

## 2. Hypothesis

> Clients in federated learning can make rational, privacy-preserving participation decisions by measuring domain alignment between global model knowledge and local task distribution — opting out when negative transfer is likely.

**Core Claims:**

1. **Negative transfer is detectable** — A lightweight probe (few-shot evaluation on local data) can predict whether federation will help or hurt a client's utility before committing to training.

2. **Selective participation improves system-wide utility** — When misaligned clients abstain, both they (local-only training) and the federation (less noise in aggregation) benefit.

3. **Participation decisions leak minimal information** — The opt-in/opt-out signal reveals bounded information about client data distribution, quantifiable via mutual information.

---

## 3. Business Objective

**Enable rational FL adoption in enterprise/cross-org settings.**

In real-world federated deployments:
- Organizations hesitate to join federations without guarantees of benefit
- Negative transfer wastes compute and may degrade local model quality
- "One-size-fits-all" federation ignores domain heterogeneity

**Value proposition:** A participation oracle lets clients answer: *"Should I join this round?"* — reducing wasted resources, improving trust, and making FL practical for heterogeneous consortiums.

**Use cases:**
- Healthcare networks with different specialties (radiology vs pathology)
- Financial institutions with different customer segments
- Multi-lingual deployments with uneven language coverage

---

## 4. ML Objective

**Maximize per-client utility while minimizing unnecessary federation overhead.**

Formally, for client $i$ with local data $D_i$:

```
Decide: participate(round_t) = True  IFF  E[U_fed(D_i)] > E[U_local(D_i)]
```

Where:
- $U_{fed}$ = utility (task metric) after participating in federation
- $U_{local}$ = utility after local-only training
- Expectation is estimated via a **low-cost probe** (not full training)

**Proxy objective for the probe:**
```
alignment_score = -KL(P_global(y|x) || P_local(y|x))  on sample S ~ D_i
```

High alignment → likely benefit from federation → participate.

---

## 5. Entities

### 5.1 Actors

| Entity | Role | Data/State |
|--------|------|------------|
| **Server** | Orchestrates rounds, distributes global model | Global model weights, participation history |
| **Client** | Holds local data, decides participation | Local dataset $D_i$, local model, participation oracle |
| **Participation Oracle** | Client-side module; estimates federation value | Probe samples, alignment threshold $\tau$ |

### 5.2 Artifacts

| Artifact | Description |
|----------|-------------|
| **Global Model** | Current federated model (or LoRA adapters) |
| **Probe Set** | Small held-out sample from client's local data (e.g., 50-100 examples) |
| **Alignment Score** | Scalar measuring global model's relevance to local task |
| **Participation Threshold** | Hyperparameter $\tau$; participate if alignment > $\tau$ |

### 5.3 Client States

```
┌─────────────┐
│   IDLE      │  (Waiting for round invitation)
└──────┬──────┘
       │ receive global model
       v
┌─────────────┐
│   PROBE     │  (Run participation oracle)
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   v       v
┌──────┐ ┌──────┐
│PARTIC│ │ABSTAIN│
│IPATE │ │      │
└──┬───┘ └──┬───┘
   │        │
   v        v
 Train    Train
 w/ fed   local-only
   │        │
   └───┬────┘
       v
┌─────────────┐
│  EVALUATE   │
└─────────────┘
```

---

## 6. High Level Diagram

```
                              FEDERATION SERVER
                    ┌─────────────────────────────────┐
                    │                                 │
                    │   Global Model (LoRA adapters)  │
                    │                                 │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    v               v               v
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Client 1  │   │ Client 2  │   │ Client 3  │
            │ (SQuAD)   │   │ (NQ)      │   │ (SAMSum)  │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  v               v               v
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │   PROBE   │   │   PROBE   │   │   PROBE   │
            │           │   │           │   │           │
            │ align=0.8 │   │ align=0.7 │   │ align=0.2 │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  v               v               v
              threshold τ = 0.5             threshold τ = 0.5
                  │               │               │
            ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
            │PARTICIPATE│   │PARTICIPATE│   │  ABSTAIN  │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  v               v               v
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │Train w/   │   │Train w/   │   │Train      │
            │global init│   │global init│   │local-only │
            │+ aggregate│   │+ aggregate│   │           │
            └─────┬─────┘   └─────┬─────┘   └───────────┘
                  │               │
                  └───────┬───────┘
                          v
                    ┌───────────┐
                    │ Aggregate │
                    │ (FedAvg)  │
                    └───────────┘


PROBE DETAIL (Client-side):
┌────────────────────────────────────────────────────────┐
│                                                        │
│   Sample S ~ D_local (50 examples)                     │
│                        │                               │
│                        v                               │
│   ┌────────────────────────────────────┐              │
│   │  Compute on S:                      │              │
│   │  - loss_global = L(global_model, S) │              │
│   │  - loss_baseline = L(base_model, S) │              │
│   │                                     │              │
│   │  alignment = loss_baseline - loss_global           │
│   │             ─────────────────────────              │
│   │                  loss_baseline                     │
│   └────────────────────────────────────┘              │
│                        │                               │
│                        v                               │
│              alignment > τ ?                           │
│                  /        \                            │
│                YES         NO                          │
│                 │           │                          │
│            PARTICIPATE   ABSTAIN                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 7. Evaluation & Metrics

### 7.1 Primary Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| **Per-client utility** | Task metric (F1/ROUGE) for each client | Higher is better |
| **Participation accuracy** | % of correct participate/abstain decisions (vs oracle hindsight) | Higher is better |
| **System efficiency** | Total compute (FLOPs) across all clients | Lower is better |
| **Negative transfer rate** | % of participating clients whose utility decreased vs local-only | Lower is better |

### 7.2 Probe Quality Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| **Probe-outcome correlation** | Spearman correlation between alignment score and actual utility gain | > 0.7 |
| **Decision ROC-AUC** | AUC for predicting "federation helps" binary outcome | > 0.8 |
| **Probe cost ratio** | Probe compute / Full training compute | < 0.05 |

### 7.3 Privacy Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| **Participation leakage** | Mutual information between participation decisions and data distribution | Quantify, minimize |
| **MIA-AUC (participating)** | Membership inference vulnerability for participating clients | ~0.5 (random) |
| **MIA-AUC (abstaining)** | Membership inference on abstaining clients (should be 0.5 by construction) | 0.5 |

### 7.4 Experimental Conditions

| Condition | Clients | Expected Outcome |
|-----------|---------|------------------|
| **Homogeneous** | C1, C2, C3 all QA | All participate, all benefit |
| **Heterogeneous** | C1=QA, C2=QA, C3=Summarization | C3 abstains, trains locally |
| **Mixed** | C1=QA, C2=Summarization, C3=QA | C2 abstains; C1,C3 federate |
| **Threshold sweep** | Vary $\tau$ from 0.1 to 0.9 | Tradeoff curve: participation rate vs avg utility |

### 7.5 Baselines

| Baseline | Description |
|----------|-------------|
| **Always Participate** | Standard FL — all clients join every round |
| **Always Abstain** | Local-only training — no federation |
| **Random Participation** | Coin flip per round |
| **Oracle (hindsight)** | Participate IFF actual utility gain > 0 (upper bound) |

### 7.6 Success Criteria

1. **Probe predicts utility gain** — Correlation > 0.7 between alignment score and actual $\Delta$utility
2. **Selective > Always Participate** — Average per-client utility improves when clients can opt out
3. **Low probe overhead** — Probe costs < 5% of full local training
4. **Minimal participation leakage** — Mutual information bounded and documented

---

## Next Steps

1. [ ] Implement `ParticipationOracle` class with configurable alignment metrics
2. [ ] Add `--selective` flag to `main.py` for enabling participation decisions
3. [ ] Run homogeneous baseline (expect all participate)
4. [ ] Run heterogeneous experiment (expect C3 to abstain)
5. [ ] Sweep threshold $\tau$ and plot participation-utility tradeoff
6. [ ] Measure probe computation cost
7. [ ] Analyze participation leakage via reconstruction attack

