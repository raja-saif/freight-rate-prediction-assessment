# Final Submission Guide for the Freight Rate Prediction Assessment

## What to submit
1. GitHub repository containing the full solution, dependencies, and run instructions.
2. validation_predictions.csv with exactly two columns: load_id,predicted_rate.
3. submission_report.docx containing:
   - the validation strategy and split rationale
   - the data exploration findings
   - the feature engineering and model choice
   - the December chart produced by score.py
4. Loom video link (2-3 minutes) explaining the same story in a concise, presentation-ready format.

## Files already generated
- validation_predictions.csv
- december_chart_inputs_filled.csv
- scorer_results/candidate_december.png
- submission_report.docx

## Re-run instructions
```bash
python -m pip install -r requirements.txt
python make_submission.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_filled.csv
```

## Final Loom script
"Hello, I built a freight-rate prediction solution using the provided development data and a forward-chaining validation approach. I did not use a random split because this is a time-series forecasting problem, and a random split would leak future patterns into training. Instead, I trained on earlier months and validated on later months so the evaluation reflects the actual deployment setting.

During exploration, I found that distance is the strongest driver of posted rate, but the data also shows clear route-level structure, equipment effects, and seasonal behavior. That is why I engineered features such as route, equipment, month, day of week, and cyclical day-of-year signals. I also handled data-quality issues by imputing missing values and normalizing negative weight values before the log transform.

I chose CatBoost because it is very strong on tabular data with categorical features like pickup, delivery, route, and equipment. It captures nonlinear interactions well without requiring a large sparse matrix, and it produced stable forward-chaining validation results. I then used that model to generate the required validation predictions, the filled December file, and the scorer chart.

The final output is reproducible, the scorer runs successfully, and the solution is structured in a way that is easy to explain in an interview or a presentation."

## Why this approach is strong
- Why not a simple linear regression?
  - It would miss nonlinear interactions and route-specific effects.
  - It would not handle high-cardinality categories as effectively.
- Why not a random train/test split?
  - It would overestimate performance by mixing earlier and later periods.
  - It is less realistic for forecasting.
- Why not a distance-only model?
  - It would ignore seasonal patterns, route behavior, and equipment differences.
- Why CatBoost?
  - It is robust on categorical and tabular data.
  - It provided stable validation performance and is well suited to this problem.

## Final submission message
If you want to sound confident in the interview or Loom, say this:
"I chose a realistic forecasting approach, not a shortcut. I validated the model in a way that reflects how it would be used in production, and I handled the data quality issues carefully so the final predictions are robust and explainable."