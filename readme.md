# Freight Rate Prediction Assessment

This repository contains a complete machine learning solution for the freight-rate prediction assessment. It trains a CatBoost regression model on the provided development data, validates it using a forward-chaining time-based split, generates predictions for the final validation set, fills the fixed December scenario inputs, runs the scorer, and produces a DOCX report with the December chart.

## What this solution includes
- A reproducible training and prediction pipeline
- A realistic validation approach for time-series forecasting
- Feature engineering for route, seasonality, and distance-based signals
- Data-quality handling for missing values and abnormal weight inputs
- Generated outputs for submission:
  - validation_predictions.csv
  - december_chart_inputs_filled.csv
  - scorer_results/candidate_december.png
  - submission_report.docx

## Project structure
- make_submission.py — end-to-end training, prediction, scoring, and report generation
- requirements.txt — dependencies
- score.py — provided scorer used to validate the outputs
- validation_predictions.csv — final predictions for the 12,000 validation loads
- december_chart_inputs_filled.csv — filled December scenario inputs
- submission_report.docx — detailed assessment report
- submission_guide.md — Loom and submission guidance

## Run locally
```bash
python -m pip install -r requirements.txt
python make_submission.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_filled.csv
```

## Notes
This approach focuses on generalization rather than overfitting by validating on future months rather than using a random data split. That makes it more appropriate for a forecasting problem like this one.