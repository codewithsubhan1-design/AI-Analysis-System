import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_dataset
from modules.data_analyzer import analyze_dataset
from modules.problem_detector import detect_problem_type
from modules.preprocessing import build_preprocessor, split_data
from modules.model_trainer import get_models, train_models
from modules.evaluator import evaluate_models
from modules.visualizer import (
    target_distribution, correlation_heatmap, numeric_distributions,
    confusion_matrix_plot, roc_curve_plot, feature_importance_plot,
    regression_actual_vs_predicted, regression_residual_plot
)
from modules.predictor import prediction_form, make_prediction
from modules.report import create_report

st.set_page_config(page_title="AutoML Intelligence", page_icon="🤖", layout="wide")

st.title("🤖 AutoML Intelligence Dashboard")
st.caption("Upload a tabular dataset → analyze → train multiple models → select the best → predict.")

if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "task" not in st.session_state:
    st.session_state.task = None
if "target" not in st.session_state:
    st.session_state.target = None

with st.sidebar:
    st.header("📁 Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            st.session_state.df = load_dataset(uploaded)
            st.success("Dataset loaded!")
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

df = st.session_state.df

if df is None:
    st.info("Upload a CSV/Excel dataset from the sidebar to begin.")
    st.markdown("""
### What this app does
- Automatic dataset profiling
- Target selection
- Classification / regression detection
- Automatic preprocessing
- Multiple ML models
- Model comparison
- Best-model selection
- Performance charts
- Prediction interface
- Feature importance
- Downloadable report
""")
    st.stop()

# Overview
st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

with st.expander("👀 Dataset Preview", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

analysis = analyze_dataset(df)

st.subheader("🔎 Automatic Data Analysis")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Numerical", len(analysis["numeric_cols"]))
a2.metric("Categorical", len(analysis["categorical_cols"]))
a3.metric("Datetime", len(analysis["datetime_cols"]))
a4.metric("Constant", len(analysis["constant_cols"]))

with st.expander("Missing Values"):
    missing = analysis["missing"].sort_values(ascending=False)
    st.dataframe(missing[missing > 0].rename("missing_values").to_frame(), use_container_width=True)

with st.expander("Column Information"):
    st.dataframe(analysis["column_info"], use_container_width=True)

# Target
st.subheader("🎯 Target Selection")
target = st.selectbox("Select the column you want the model to predict:", df.columns)

task, task_info = detect_problem_type(df[target])
st.session_state.target = target
st.session_state.task = task

t1, t2 = st.columns(2)
t1.success(f"Detected task: **{task.upper()}**")
t2.info(task_info)

if task == "classification":
    st.write("Classes:", df[target].nunique())
    st.dataframe(df[target].value_counts(dropna=False).rename("count").to_frame(), use_container_width=True)
else:
    st.write("Target statistics")
    st.dataframe(df[target].describe().to_frame().T, use_container_width=True)

# Visualizations
st.subheader("📈 Dataset Trends & Visualizations")
tab1, tab2, tab3 = st.tabs(["Target", "Correlation", "Distributions"])

with tab1:
    try:
        fig = target_distribution(df, target, task)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Target chart unavailable: {e}")

with tab2:
    try:
        fig = correlation_heatmap(df, analysis["numeric_cols"])
        if fig:
            st.pyplot(fig)
        else:
            st.info("Not enough numerical columns for a correlation heatmap.")
    except Exception as e:
        st.warning(f"Correlation chart unavailable: {e}")

with tab3:
    try:
        figs = numeric_distributions(df, analysis["numeric_cols"][:12])
        for fig in figs:
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Distribution charts unavailable: {e}")

# Training
st.subheader("🤖 Automatic Model Training")

test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)

models = get_models(task)

if st.button("🚀 Train & Compare Models", type="primary"):
    if df[target].isna().all():
        st.error("The selected target contains only missing values.")
        st.stop()

    with st.spinner("Building preprocessing pipeline and training models..."):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Remove rows with missing target values.
            valid = y.notna()
            X = X.loc[valid].copy()
            y = y.loc[valid].copy()

            # Basic datetime conversion: extract useful components.
            for col in list(X.columns):
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    X[col] = X[col].astype("int64") // 10**9

            preprocessor = build_preprocessor(X)
            X_train, X_test, y_train, y_test = split_data(
                X, y, task, test_size=test_size, random_state=random_state
            )

            results, fitted_models = train_models(
                models, preprocessor, X_train, X_test, y_train, y_test, task
            )
            evaluated = evaluate_models(results, task)

            st.session_state.results = evaluated
            st.session_state.best_model = fitted_models[evaluated.iloc[0]["Model"]]
            st.session_state.preprocessor = preprocessor
            st.session_state.X_columns = list(X.columns)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("Training completed successfully!")
        except Exception as e:
            st.exception(e)

results = st.session_state.results

if results is not None:
    st.subheader("🏆 Model Comparison")
    st.dataframe(results, use_container_width=True)

    best_name = results.iloc[0]["Model"]
    best_row = results.iloc[0]

    st.success(f"🏆 Best Model: **{best_name}**")

    metric_cols = [c for c in results.columns if c != "Model"]
    cols = st.columns(min(len(metric_cols), 5))
    for i, col in enumerate(metric_cols[:5]):
        cols[i].metric(col, f"{best_row[col]:.4f}")

    # Model charts
    st.subheader("📊 Best Model Performance")
    best = st.session_state.best_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    try:
        if task == "classification":
            st.pyplot(confusion_matrix_plot(best, X_test, y_test))
            if hasattr(best, "predict_proba") and y_test.nunique() == 2:
                st.pyplot(roc_curve_plot(best, X_test, y_test))
        else:
            st.pyplot(regression_actual_vs_predicted(best, X_test, y_test))
            st.pyplot(regression_residual_plot(best, X_test, y_test))
    except Exception as e:
        st.warning(f"Performance chart unavailable: {e}")

    if task in ["classification", "regression"]:
        try:
            fig = feature_importance_plot(best, st.session_state.preprocessor, X_test.columns)
            if fig:
                st.subheader("🧠 Feature Importance")
                st.pyplot(fig)
        except Exception as e:
            st.info(f"Feature importance unavailable for this model: {e}")

    # Prediction
    st.subheader("🔮 Make a Prediction")
    X_original = df.drop(columns=[target])
    input_data = prediction_form(X_original)

    if st.button("🔮 Predict"):
        try:
            prepared = input_data.copy()
            for col in list(prepared.columns):
                if col not in st.session_state.X_columns:
                    prepared = prepared.drop(columns=[col])
            prepared = prepared[st.session_state.X_columns]
            pred, proba = make_prediction(best, prepared, task)
            st.success(f"Prediction: **{pred}**")
            if proba is not None:
                st.write("Prediction probabilities:")
                st.dataframe(pd.DataFrame([proba]), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Download report
    report = create_report(df, target, task, analysis, results)
    st.download_button(
        "📥 Download Model Analysis Report",
        data=report,
        file_name="automl_report.txt",
        mime="text/plain"
    )
