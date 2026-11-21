# Informative missingness in ICU mortality prediction (PhysioNet/CinC 2012)

Code for the MedPRAI 2027 submission "Informative Missingness in ICU Mortality Prediction: Imputation,
Missingness Indicators, and Robustness to Measurement-Policy Shift on the PhysioNet/CinC 2012 Challenge".

## Data
Download version 1.0.0 of the PhysioNet/CinC Challenge 2012 database (open access, ODC-By v1.0):
https://physionet.org/content/challenge-2012/1.0.0/
Required files: set-a.tar.gz, set-b.tar.gz, set-c.tar.gz, Outcomes-a.txt, Outcomes-b.txt, Outcomes-c.txt.
Place them in `data/raw/` and extract the three archives there (giving `data/raw/set-a/`, `set-b/`, `set-c/`).

## Environment
Python 3.10+; see `requirements.txt` (xgboost 2.1.1). All results were produced on one CPU core with random seed 2027.

## Run (from the repository root)
    export P12_ROOT=$(pwd)
    python code/01_features.py   # parse raw records -> out/raw.pkl
    python code/02_models.py     # 17 configurations, tuning on set A only, testing on sets B and C, bootstrap CIs
    python code/03_shift.py      # measurement-shift stress tests, measurement-dropout augmentation, SHAP shares, tables
    python code/04_saps.py       # paired comparison with SAPS-I on stays with a recorded score
    python code/fig1.py; python code/fig234.py

## Outputs (`out/`)
main_results.csv (pooled B+C and per-set B, C metrics), shift_raw.csv (every seed), shift_agg.csv (seed means),
saps_paired.json, cohort.csv, measurement_prevalence_A.csv, shap_*.csv. Sets B and C are never used for tuning,
threshold selection or model choice.
