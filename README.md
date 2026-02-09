# Imbalance-Aware Explainable Machine Learning for Child Undernutrition Modeling

This repository contains reproducible research code and scripts for the manuscript
"Imbalance-Aware Explainable Machine Learning for Concurrent Child Undernutrition Modeling"
(Young Lives Ethiopia). It is structured to enable end-to-end regeneration of preprocessing,
model training, evaluation, tables, and figures without shipping restricted data.

## Repository map

- **Configuration**
  - [`configs/`](configs/) — Model, experiment, or pipeline configuration files.
- **Data**
  - [`data/`](data/) — Raw and processed datasets (restricted data not included).
- **Documentation**
  - [`docs/`](docs/) — Project documentation and references.
- **Figures**
  - [`figures/`](figures/) — Generated plots and supplementary visual assets.
- **Notebooks**
  - [`notebooks/`](notebooks/) — Exploratory analysis, prototyping, and reports.
- **Scripts**
  - [`scripts/`](scripts/) — CLI utilities and one-off automation tasks.
- **Source code**
  - [`src/`](src/) — Core library code for data prep, modeling, and explainability.
- **Tests**
  - [`tests/`](tests/) — Automated tests for the codebase.
- **Dependencies**
  - [`requirements.txt`](requirements.txt) — Python dependency list.

## Quickstart

1. Create an environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Add data:
   - Place the dataset at `data/young_lives_ethiopia.csv` (or update `configs/default.txt`).
   - The dataset should contain the outcome column and covariates described in the manuscript.
3. Run the full pipeline:
   ```bash
   python scripts/run_pipeline.py --config configs/default.txt
   ```
4. Reproduce figures and tables:
   ```bash
   python scripts/make_figures.py --config configs/default.txt
   python scripts/make_tables.py --config configs/default.txt
   ```

## Reproducibility notes

- Train/test split is performed before any preprocessing.
- Imputation, encoding, scaling, and SMOTE are fit only on training data.
- SMOTE is applied within cross-validation folds to prevent leakage.
- Model selection uses macro-F1 and balanced accuracy.

## Citation

If you use this code, please cite the manuscript and this repository. See `CITATION.txt`.

## Manuscript link

Placeholder: <https://doi.org/XX.XXXX/placeholder>

## License

MIT License. See `LICENSE.txt`.
