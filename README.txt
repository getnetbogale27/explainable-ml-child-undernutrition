Imbalance-Aware Explainable Machine Learning for Concurrent Child Undernutrition Modeling

This repository contains reproducible research code and scripts for the manuscript
"Imbalance-Aware Explainable Machine Learning for Concurrent Child Undernutrition Modeling"
(Young Lives Ethiopia). The repository is structured to enable end-to-end regeneration
of preprocessing, model training, evaluation, tables, and figures without shipping
restricted data.

Repository structure
- data/: place the prepared dataset here (restricted data not included).
- src/: Python package with preprocessing, modeling, evaluation, and explainability.
- scripts/: entrypoints that reproduce figures and tables.
- configs/: configuration files for paths, model grids, and CV settings.
- results/: generated model artifacts and intermediate outputs.
- figures/: generated plots (ROC, PR, calibration, SHAP summaries).
- tables/: generated tables (metrics, model selection summaries).
- docs/: documentation, including outcome definitions.
- notebooks/: exploratory and draft analysis notebooks.
- tests/: minimal unit tests for leakage checks and data shape validation.

Quickstart
1) Create an environment
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
2) Add data
   - Place the dataset at data/young_lives_ethiopia.csv (or update configs/default.txt).
   - The dataset should contain the outcome column and covariates described in the manuscript.
3) Run the full pipeline
   python scripts/run_pipeline.py --config configs/default.txt
4) Reproduce figures and tables
   python scripts/make_figures.py --config configs/default.txt
   python scripts/make_tables.py --config configs/default.txt

Reproducibility notes
- Train/test split is performed before any preprocessing.
- Imputation, encoding, scaling, and SMOTE are fit only on training data.
- SMOTE is applied within cross-validation folds to prevent leakage.
- Model selection uses macro-F1 and balanced accuracy.

Citation
If you use this code, please cite the manuscript and this repository. See CITATION.txt.

Manuscript link
Placeholder: https://doi.org/XX.XXXX/placeholder

License
MIT License. See LICENSE.txt.
