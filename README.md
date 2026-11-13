# TriageRank

**A research-only benchmark for early-warning classification and organ-priority ranking.**

My master's work made me interested in a subtle distinction: detecting whether a patient may deteriorate and deciding *what to investigate first* are different modelling problems. TriageRank is a deliberately self-contained experiment with a seeded **synthetic** cohort so the ML workflow can be inspected without clinical credentials, private datasets or patient records.

## What I built
- Synthetic feature and label generator with deliberately incomplete observations
- Train/test split with stratification; calibrated sepsis classifier (3-fold calibration on the training set only)
- Separate organ risk classifiers for renal, cardiovascular and respiratory labels
- AUROC, AUPRC, Brier score, label prevalence and top-2 organ hit rate
- Reproducible JSON metrics, unit tests and documented methodological limits

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
python -m triagerank --report reports/generated/benchmark.json
pytest -q
```

## Important boundary
This is **not a clinical model**, does not access MIMIC-IV and must never be used to advise, diagnose, or prioritise real patients. Synthetic labels are generated from manually chosen relationships; favourable evaluation numbers would only demonstrate that the program can recover relationships in its own simulated dataset. Real clinical research requires cohort definitions, temporal splitting, external validation, subgroup assessment, ethical approval/data governance and prospective clinical evaluation.
