import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# EMI ELIGIBILITY & MAXIMUM MONTHLY EMI PREDICTION APP
# ============================================================

st.set_page_config(
    page_title="EMI Eligibility Prediction",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR, "classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_DIR, "regression_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR, "preprocessor.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        raise FileNotFoundError(
            "classification_model.pkl was not found."
        )

    if not os.path.exists(REGRESSION_MODEL_PATH):
        raise FileNotFoundError(
            "regression_model.pkl was not found."
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "preprocessor.pkl was not found."
        )

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return (
        classification_model,
        regression_model,
        preprocessor
    )


# ============================================================
# FIND FEATURE INFORMATION
# ============================================================

def get_feature_information(preprocessor):

    numeric_features = []
    categorical_features = []
    categorical_values = {}

    # --------------------------------------------------------
    # ColumnTransformer
    # --------------------------------------------------------

    if hasattr(preprocessor, "transformers_"):

        for name, transformer, columns in preprocessor.transformers_:

            if name == "remainder":
                continue

            if columns is None:
                continue

            try:
                columns = list(columns)
            except Exception:
                columns = [columns]

            transformer_name = transformer.__class__.__name__

            # ----------------------------------------------
            # Numeric transformer
            # ----------------------------------------------

            numeric_names = [
                "StandardScaler",
                "MinMaxScaler",
                "RobustScaler",
                "MaxAbsScaler",
                "Normalizer",
                "PowerTransformer",
                "QuantileTransformer"
            ]

            if transformer_name in numeric_names:

                numeric_features.extend(columns)

            # ----------------------------------------------
            # Categorical transformer
            # ----------------------------------------------

            elif transformer_name in [
                "OneHotEncoder",
                "OrdinalEncoder"
            ]:

                categorical_features.extend(columns)

                # Get categories if available
                if hasattr(transformer, "categories_"):

                    for col, categories in zip(
                        columns,
                        transformer.categories_
                    ):

                        categorical_values[col] = [
                            str(x)
                            for x in categories
                        ]

            # ----------------------------------------------
            # Pipeline containing encoder/scaler
            # ----------------------------------------------

            elif hasattr(transformer, "steps"):

                found_categorical = False
                found_numeric = False

                for step_name, step in transformer.steps:

                    if hasattr(step, "categories_"):

                        found_categorical = True

                        if hasattr(step, "categories_"):

                            for col, categories in zip(
                                columns,
                                step.categories_
                            ):

                                categorical_values[col] = [
                                    str(x)
                                    for x in categories
                                ]

                    if step.__class__.__name__ in numeric_names:                                                                                                                                                              found_numeric = True

                if found_categorical:

                    categorical_features.extend(columns)

                elif found_numeric:

                    numeric_features.extend(columns)

                else:

                    numeric_features.extend(columns)

            else:

                # Unknown transformer.
                # Assume numeric unless categories are available.

                if hasattr(transformer, "categories_"):

                    categorical_features.extend(columns)

                    for col, categories in zip(
                        columns,
                        transformer.categories_
                    ):

                        categorical_values[col] = [
                            str(x)
                            for x in categories
                        ]

                else:

                    numeric_features.extend(columns)

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    numeric_features = list(
        dict.fromkeys(numeric_features)
    )

    categorical_features = list(
        dict.fromkeys(categorical_features)
    )

    return (
        numeric_features,
        categorical_features,
        categorical_values
    )


# ============================================================
# FALLBACK FEATURE NAMES
# ============================================================

def get_all_features(preprocessor):

    if hasattr(preprocessor, "feature_names_in_"):

        return list(
            preprocessor.feature_names_in_
        )

    numeric_features, categorical_features, _ = (
        get_feature_information(preprocessor)
    )

    return (
        numeric_features +
        categorical_features
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💰 EMI Eligibility Prediction System")

st.write(
    """
    This application uses the trained machine-learning models
    from the EMI prediction project to predict:

    1. EMI Eligibility
    2. Maximum Monthly EMI
    """
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        classification_model,
        regression_model,
        preprocessor
    ) = load_models()

except Exception as e:

    st.error("Unable to load the trained models.")

    st.exception(e)

    st.stop()


# ============================================================
# GET FEATURE INFORMATION
# ============================================================

try:

    (
        numeric_features,
        categorical_features,
        categorical_values
    ) = get_feature_information(preprocessor)

    all_features = get_all_features(preprocessor)

except Exception as e:

    st.error(
        "Unable to read the feature information from "
        "preprocessor.pkl."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SHOW MODEL INFORMATION
# ============================================================

with st.expander("🔎 Model information"):

    st.write(
        "Number of input features:",
        len(all_features)
    )

    if numeric_features:

        st.write(
            "Numeric features:",
            numeric_features
        )

    if categorical_features:

        st.write(
            "Categorical features:",
            categorical_features
        )


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("📋 Enter Applicant Details")

if not all_features:

    st.error(
        "No input features were found in the saved preprocessor."
    )

    st.stop()


user_input = {}


# ============================================================
# CREATE INPUT FIELDS
# ============================================================

for feature in all_features:

    # --------------------------------------------------------
    # CATEGORICAL FEATURE
    # --------------------------------------------------------

    if feature in categorical_features:

        options = categorical_values.get(feature, [])

        if options:

            user_input[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options
            )

        else:

            user_input[feature] = st.text_input(
                feature.replace("_", " ").title()
            )

    # --------------------------------------------------------
    # NUMERIC FEATURE
    # --------------------------------------------------------

    else:

        user_input[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=0.0,
            step=1.0
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict EMI Eligibility",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [user_input],
            columns=all_features
        )

        # ----------------------------------------------------
        # CONVERT NUMERIC FEATURES
        # ----------------------------------------------------

        for col in numeric_features:

            if col in input_df.columns:

                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce"
                )

        # ----------------------------------------------------
        # HANDLE MISSING NUMERIC VALUES
        # ----------------------------------------------------

        for col in numeric_features:

            if col in input_df.columns:

                if input_df[col].isna().any():

                    input_df[col] = input_df[col].fillna(0)

        # ----------------------------------------------------
        # TRANSFORM DATA
        # ----------------------------------------------------

        transformed_data = preprocessor.transform(
            input_df
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        classification_prediction = (
            classification_model.predict(
                transformed_data
            )
        )

        eligibility = classification_prediction[0]
        st.write("Model classes:", classification_model.classes_)
        st.write("Predicted class:", eligibility)

        # ----------------------------------------------------
        # REGRESSION
        # ----------------------------------------------------

        regression_prediction = (
            regression_model.predict(
                transformed_data
            )
        )

        predicted_emi = float(
            regression_prediction[0]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # ELIGIBILITY RESULT
        # ----------------------------------------------------

        with col1:

            st.metric(
                "EMI Eligibility",
                str(eligibility)
            )

        # ----------------------------------------------------
        # EMI RESULT
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Predicted Maximum Monthly EMI",
                f"₹{predicted_emi:,.2f}"
            )

        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        eligibility_text = str(
            eligibility
        ).lower()

        if "eligible" in eligibility_text and "not" not in eligibility_text:

            st.success(
                "The applicant is predicted to be EMI eligible."
            )

        elif "high" in eligibility_text:

            st.warning(
                "The applicant is predicted to be High Risk."
            )

        else:

            st.error(
                "The applicant is predicted to be Not Eligible."
            )

        # ----------------------------------------------------
        # TECHNICAL OUTPUT
        # ----------------------------------------------------

        with st.expander("View prediction details"):

            st.write(
                "Classification prediction:",
                eligibility
            )

            st.write(
                "Predicted maximum monthly EMI:",
                predicted_emi
            )

            st.write("Input data used for prediction:")

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.write(
            """
            This usually means that the input columns do not
            exactly match the columns used when the model was
            trained.
            """
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EMI Eligibility & Maximum Monthly EMI Prediction System"
)