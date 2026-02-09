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
   - Place the dataset at `data/Baseline Data.csv` (or update `configs/default.txt`).
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

## Supplementary tables (S5-S7)

Generate specific supplementary tables directly from the raw dataset:

```bash
python scripts/make_table_s5_bivariate.py --data data/young_lives_ethiopia.csv --target concurrent_conditions --out results/tables/table_s5_bivariate.csv
python scripts/make_table_s6_fold_accuracies.py --data data/young_lives_ethiopia.csv --target concurrent_conditions --out results/tables/table_s6_fold_accuracies.csv
python scripts/make_table_s7_performance_splits.py --data data/young_lives_ethiopia.csv --target concurrent_conditions --out results/tables/table_s7_performance_splits.csv
```

## Testing and linting

Run the test suite with:

```bash
python -m pytest
```

Run lint checks with:

```bash
python -m ruff check .
```

## Reproducibility notes

- Train/test split is performed before any preprocessing.
- Imputation, encoding, scaling, and SMOTE are fit only on training data.
- SMOTE is applied within cross-validation folds to prevent leakage.
- Model selection uses macro-F1 and balanced accuracy.

## Data Availability

The dataset used in this study was obtained from the Young Lives Study. Access to the data can be obtained either by completing the form available at [Young Lives Data Access](https://www.younglives.org.uk/use-our-data-form), selecting the dataset "Young Lives: Rounds 1-5 constructed files, 2002-2016" (and then extract only 2002 or baseline survey only where authors used in this study), or by creating a user account through the UK Data Service, subject to their terms and conditions. Additionally, the survey questionnaires for round 1 are available through [Young Lives Round 1 Questionnaires](https://www.younglives.org.uk/round-1-questionnaires). After downloading the data, place `Baseline Data.csv` in the `data/` folder to run the code.

## Citation

If you use this code, please cite the manuscript and this repository. See `CITATION.txt`.

## Manuscript status and authors

This manuscript is currently submitted to Scientific Reports (Springer Nature) and is under review.

**Authors**
- Getnet Bogale Begashaw (corresponding author; Getnetbogale145@gmail.com, Getnetbogale@dbu.edu.et)
- Temesgen Zewotir
- Haile Mekonnen Fenta
- Mulu Abebe Asmamaw
- Abebe Mengistu Legass

## License

MIT License. See `LICENSE`.
