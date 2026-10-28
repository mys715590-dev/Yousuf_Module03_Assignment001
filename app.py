import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
BUNDLE_PATH = os.path.join(
    ARTIFACT_DIR, "preprocessing_bundle.joblib"
)
MODEL_PATH = os.path.join(
    ARTIFACT_DIR, "salary_ann.keras"
)

st.set_page_config(
    page_title="Salary Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #050d19, #0b1e35, #071426
    );
    color: #edf5ff;
}
section[data-testid="stSidebar"] {
    background: #08172a;
}
div[data-testid="stMetric"] {
    background: #102746;
    border: 1px solid #1d4770;
    padding: 18px;
    border-radius: 14px;
}
h1, h2, h3 {
    color: #49caff !important;
}
.stButton > button {
    background: #0877ff;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(BUNDLE_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return bundle, model


st.sidebar.title("◈ SALARY INTELLIGENCE")
st.sidebar.caption("ADVANCED ML ANALYTICS")

if not os.path.exists(BUNDLE_PATH) or not os.path.exists(MODEL_PATH):
    st.title("Customer Salary Prediction")
    st.error(
        "Trained model files not found. "
        "Run python train_model.py first."
    )
    st.stop()

bundle, model = load_artifacts()
metrics = bundle["metrics"]

st.title("Customer Salary Prediction")
st.caption(
    "ANN-powered salary regression and model analytics"
)

overview, predict, analytics, about = st.tabs([
    "Overview",
    "Predict Salary",
    "Analytics",
    "Project Information"
])

with overview:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{metrics['MAE']:,.2f}"
    )
    col2.metric(
        "Root Mean Squared Error",
        f"{metrics['RMSE']:,.2f}"
    )
    col3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

    if metrics["R2"] < 0:
        st.warning(
            "Negative R²: the model performs worse than "
            "predicting the average target on the test set."
        )
    elif metrics["R2"] < 0.1:
        st.warning(
            "The model explains very little salary variation. "
            "Interpret predictions cautiously."
        )

    st.markdown("### Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"]
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

with predict:
    st.subheader("Predict Estimated Salary")
    st.write(
        "Enter the customer's details to generate a prediction."
    )

    values = {}

    with st.form("salary_prediction_form"):
        left, right = st.columns(2)

        for i, feature in enumerate(
            bundle["feature_columns"]
        ):
            with (left if i % 2 == 0 else right):
                if feature in bundle["numeric_features"]:
                    default = float(
                        pd.to_numeric(
                            pd.Series([0]),
                            errors="coerce"
                        ).iloc[0]
                    )

                    # Set useful defaults for common dataset columns.
                    if feature == "CreditScore":
                        default = 650
                    elif feature == "Age":
                        default = 35
                    elif feature == "Tenure":
                        default = 5
                    elif feature == "Balance":
                        default = 50000
                    elif feature == "NumOfProducts":
                        default = 2
                    elif feature == "EstimatedSalary":
                        default = 50000

                    if feature in [
                        "HasCrCard", "IsActiveMember", "Exited"
                    ]:
                        value = st.selectbox(
                            feature, [0, 1], key=feature
                        )
                    else:
                        value = st.number_input(
                            feature,
                            value=float(default),
                            key=feature
                        )
                else:
                    value = st.text_input(
                        feature, key=feature
                    )

                values[feature] = value

        submitted = st.form_submit_button(
            "Predict Salary",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame(
            [values],
            columns=bundle["feature_columns"]
        )

        try:
            processed = bundle["preprocessor"].transform(
                input_df
            )
            processed = np.asarray(
                processed, dtype="float32"
            )

            prediction = float(
                model.predict(processed, verbose=0)[0][0]
            )

            st.success("Prediction generated")
            st.metric(
                "Estimated Salary",
                f"{prediction:,.2f}"
            )

            st.caption(
                "This is a model estimate, not a guaranteed salary."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with analytics:
    st.subheader("Training Dataset Analytics")

    csv_path = os.path.join(
        BASE_DIR, "Churn_Modelling.csv"
    )

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Columns", f"{len(df.columns)}")

        if "EstimatedSalary" in df.columns:
            c3.metric(
                "Average Salary",
                f"{df['EstimatedSalary'].mean():,.2f}"
            )

            fig = px.histogram(
                df,
                x="EstimatedSalary",
                nbins=35,
                title="Estimated Salary Distribution",
                color_discrete_sequence=["#20baff"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        st.markdown("### Dataset Preview")
        st.dataframe(
            df.head(20),
            use_container_width=True
        )
    else:
        st.warning(
            "Churn_Modelling.csv was not found."
        )

with about:
    st.subheader("Project Methodology")

    st.markdown("""
    **Algorithm:** Artificial Neural Network regression

    **Preprocessing:**
    - Median imputation for numeric features
    - Standard scaling for numeric features
    - One-hot encoding for categorical features
    - Train/test split before fitting preprocessing

    **Evaluation:**
    - MAE
    - RMSE
    - R² score

    **Target:** EstimatedSalary

    **Important limitation:** The Churn Modelling dataset
    was designed for churn analysis. Salary may not be
    predictable from its available customer attributes.
    Poor performance should be reported honestly.
    """)