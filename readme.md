# Freight Rate Prediction Assessment

This repository contains a reproducible machine-learning solution for the freight-rate prediction assessment.

## What this solution does
- Trains a CatBoost regression model on the provided development data
- Uses forward-chaining time-based validation instead of a random split
- Generates validation predictions for the final load set
- Produces filled December predictions for the fixed chart scenario
- Creates a scorer chart and a submission-ready report

## Key files
- make_submission.py - end-to-end training, feature engineering, prediction, and report pipeline
- score.py - scorer and chart generation
- validation_predictions.csv - predictions for the validation load set
- december_chart_inputs_filled.csv - filled predictions for the December chart input
- submission_report.docx - submission report in DOCX format
- requirements.txt - Python dependencies

## Reproduce locally
```bash
python -m pip install -r requirements.txt
python make_submission.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_filled.csv
```

## Approach
- The validation strategy is chronological because this is a forecasting problem.
- Feature engineering includes route, time, equipment, and seasonal signals.
- Missing values and abnormal weight values are handled before modeling.
- CatBoost was selected for strong performance on tabular data with categorical features.

This project is set up so the full workflow can be run locally and the outputs can be submitted directly.