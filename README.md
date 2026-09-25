# 🤖 AutoML Intelligence Dashboard

A Streamlit-based AutoML dashboard for tabular CSV/Excel datasets.

## Features

- CSV and Excel upload
- Dataset profiling
- Missing-value analysis
- Numerical/categorical detection
- Target selection
- Classification/regression detection
- Automatic preprocessing
- Multiple ML algorithms
- Model comparison
- Best-model selection
- Confusion matrix / ROC curve
- Regression plots
- Feature importance
- Interactive prediction
- Downloadable report

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is a strong portfolio MVP, not a claim that every possible dataset can be solved automatically. Real production AutoML needs additional handling for time series, NLP, image data, leakage detection, high-cardinality features, imbalance strategies, hyperparameter optimization, and resource limits.
