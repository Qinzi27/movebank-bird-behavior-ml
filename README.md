# Movebank Bird Behavior Analysis with HMM

## Project Overview
This project implements an unsupervised learning pipeline to identify behavioral states (e.g., *Resting*, *Foraging*, *Commuting*) from GPS tracking data of Guanay Cormorants. It utilizes **Hidden Markov Models (HMM)** to decode latent states based on movement statistics.

## Key Features
* **Trajectory Resampling**: Converts irregular GPS fixes into regular time intervals (e.g., 60s) using linear interpolation to satisfy HMM discrete-time assumptions.
* **Physics-Informed Features**: Calculates speed and turning angles from geodesic coordinates (Haversine formula).
* **Unsupervised Learning**: Uses Gaussian HMMs to cluster movement patterns without human annotation.

## Directory Structure
```text
.
├── data/
│   └── raw/                   # Place your .csv files here
├── src/
│   ├── features.py            # Feature engineering (Resampling, Haversine)
│   ├── models.py              # HMM wrapper
│   └── main.py                # Execution entry point
├── requirements.txt           # Dependencies
└── README.md