# Movement Ecology: Unsupervised Behavioral State Recognition

## 1. Project Overview
This project implements a **Hidden Markov Model (HMM)** pipeline to identify latent behavioral states (e.g., *Resting* vs. *Flight*) from GPS tracking data of *Guanay Cormorants*. 

Unlike traditional rule-based methods, this approach uses **unsupervised learning** to discover behavioral patterns based on movement statistics. A key engineering contribution of this repository is the implementation of **trajectory resampling** to ensure the mathematical validity of the Markov property in discrete-time HMMs.

## 2. Mathematical Logic

### 2.1 Feature Engineering with Resampling
Raw GPS data often suffers from irregular sampling intervals ($dt$). Directly applying HMMs to irregular time series violates the assumption of constant transition probabilities per time step.
We solve this by **Linear Interpolation Resampling**:

1.  **Coordinate Interpolation**:
    $$\text{Lat}_t, \text{Lon}_t = \text{Interp}(\text{Lat}_{raw}, \text{Lon}_{raw}, t)$$
    where $t$ is a regular grid (e.g., every 60 seconds).
    
2.  **Log-Speed Feature**:
    Speed is log-transformed to approximate a Gaussian distribution, which fits the `GaussianHMM` assumption:
    $$x_t = \ln\left(\frac{\text{Haversine}(\text{Loc}_t, \text{Loc}_{t+1})}{\Delta t} + \epsilon\right)$$

### 2.2 Unsupervised HMM
We model the movement as a doubly stochastic process:
* **Hidden States ($z_t$)**: Discrete behavioral modes (0: Rest, 1: Flight).
* **Observations ($x_t$)**: The log-speed emitted by the current state.

The model parameters $\theta = \{A, \mu, \Sigma, \pi\}$ are estimated via the **Baum-Welch algorithm** (Expectation-Maximization).

## 3. Directory Structure
```text
.
├── data/
│   └── raw/                   # Raw GPS CSV files
├── src/
│   ├── features.py            # Haversine logic & Resampling pipeline
│   ├── models.py              # HMM wrapper with semantic state mapping
│   └── visualization.py       # (Optional) Plotting utilities
├── notebooks/                 # Exploratory analysis
├── main.py                    # Production entry point
├── requirements.txt           # Dependencies
└── README.md