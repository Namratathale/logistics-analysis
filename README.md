# Logistics Data Analysis: Internship Project

Data collection, cleaning and preprocessing pipeline for a logistics / supply chain dataset
(DataCo Smart Supply Chain for Big Data Analysis).

## What the pipeline does
1. Loads the raw CSV and records its shape, types and sample rows
2. Audits quality: missing values, duplicates, impossible values, inconsistent text
3. Cleans: drops empty/PII columns, removes duplicates, parses dates, fixes invalid values,
   standardises text, imputes missing values, engineers delay features
4. Detects outliers (IQR and z-score) and caps them at the IQR fences
5. Transforms and normalises (log1p for skewed columns, standardisation, min-max scaling)
6. Encodes categorical variables (one-hot)
7. Validates the result and saves a before/after summary

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download the dataset (see below) and put the CSV in data/raw/
python week2_pipeline.py
```

## Dataset
Search "DataCo Smart Supply Chain for Big Data Analysis" on Mendeley Data (also mirrored on Kaggle).
Check the licence on the dataset page. The raw file is not stored in this repository.
Expected path: `data/raw/DataCoSupplyChainDataset.csv`

## Outputs
- `outputs/figures/`: images of each step (used in the report)
- `outputs/tables/`: before/after tables as CSV
- `outputs/pipeline_log.txt`: full run log
- `data/sample/clean_orders_sample.csv`: 1,000-row sample of the cleaned data

## Project structure
```
logistics-analysis/
├── data/ (raw, processed, sample)
├── outputs/ (figures, tables)
├── docs/            # Week 1 and Week 2 reports
├── week2_pipeline.py
├── requirements.txt
└── README.md
```

## Author
[Your name], Logistics Data Analyst Intern
"# logistics-analysis" 
