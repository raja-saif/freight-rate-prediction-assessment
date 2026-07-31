# Freight Rate Prediction Assessment

## Summary
This solution trains a CatBoost regression model on the provided development data, validates it with a forward-chaining time-based split, and produces:
- validation predictions for the 12,000 final loads
- filled December predictions for the fixed chart scenario
- a scorer-generated December chart
- a detailed DOCX report

## Files
- validation_predictions.csv
- december_chart_inputs_filled.csv
- scorer_results/candidate_december.png
- submission_report.docx
- make_submission.py
- requirements.txt

## Run
```bash
python -m pip install -r requirements.txt
python make_submission.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_filled.csv
```

## Approach
- Used forward-chaining validation rather than a random split because this is a forecasting problem.
- Engineered route, time, and seasonal features.
- Handled missing values and abnormal weight values.
- Chose CatBoost for strong performance on tabular data with categorical features.
