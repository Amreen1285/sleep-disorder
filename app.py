import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sleep Disorder Classification",
    page_icon="😴",
    layout="wide"
)

# ============================================================
# PROJECT FILES
# ============================================================

MODEL_FILE = "final_sleep_disorder_svm_model.pkl"
PREPROCESSOR_FILE = "sleep_disorder_preprocessor.pkl"
LABEL_ENCODER_FILE = "sleep_disorder_label_encoder.pkl"
DATASET_FILE = "Sleep_health_and_lifestyle_dataset.csv"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_FILE)

    preprocessor = joblib.load(PREPROCESSOR_FILE)

    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    return model, preprocessor, label_encoder


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATASET_FILE)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    model, preprocessor, label_encoder = load_model_files()

    df = load_dataset()

except Exception as e:

    st.error(
        "Required project files could not be loaded. "
        "Make sure the .pkl files and CSV file are in the same folder as app.py."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("😴 Sleep Disorder Classification Using Machine Learning")

st.write(
    "A machine-learning-based system for predicting possible sleep "
    "disorder categories using lifestyle and physiological information."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.write("**Domain:** Machine Learning")

    st.write("**Task:** Classification")

    st.write("**Final Model:** Optimized SVM")

    st.write("**Dataset:** Sleep Health and Lifestyle Dataset")

    st.divider()

    st.subheader("Input Features")

    st.write("• Age")
    st.write("• Occupation")
    st.write("• BMI Category")
    st.write("• Sleep Duration")
    st.write("• Stress Level")


# ============================================================
# GET DROPDOWN VALUES FROM DATASET
# ============================================================

occupation_values = sorted(
    df["Occupation"].dropna().unique().tolist()
)

bmi_values = sorted(
    df["BMI Category"].dropna().unique().tolist()
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Patient Information")

st.write(
    "Enter the required information to generate a sleep disorder prediction."
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    occupation = st.selectbox(
        "Occupation",
        occupation_values
    )

    bmi_category = st.selectbox(
        "BMI Category",
        bmi_values
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    sleep_duration = st.number_input(
        "Sleep Duration (hours)",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.1
    )

    stress_level = st.number_input(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Sleep Disorder",
    type="primary",
    use_container_width=True
):

    try:

        # --------------------------------------------------------
        # CREATE INPUT DATAFRAME
        # --------------------------------------------------------

        input_df = pd.DataFrame([
            {
                "Age": age,
                "Occupation": occupation,
                "BMI Category": bmi_category,
                "Sleep Duration": sleep_duration,
                "Stress Level": stress_level
            }
        ])


        # --------------------------------------------------------
        # PREPROCESS INPUT
        # --------------------------------------------------------

        transformed = preprocessor.transform(input_df)


        # --------------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------------

        prediction_encoded = model.predict(transformed)

        prediction = label_encoder.inverse_transform(
            prediction_encoded
        )[0]


        # --------------------------------------------------------
        # PREDICTION PROBABILITY
        # --------------------------------------------------------

        confidence = None

        probabilities = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                transformed
            )[0]

            confidence = float(
                max(probabilities)
            )


        # ========================================================
        # RISK LEVEL
        # ========================================================

        if prediction == "None":

            if confidence is None or confidence >= 0.80:

                risk_level = "LOW"

            else:

                risk_level = "MODERATE"


        elif prediction in ["Insomnia", "Sleep Apnea"]:

            if confidence is None:

                risk_level = "MODERATE"

            elif confidence >= 0.80:

                risk_level = "HIGH"

            elif confidence >= 0.60:

                risk_level = "MODERATE"

            else:

                risk_level = "LOW"


        else:

            risk_level = "MODERATE"


        # ========================================================
        # PREDICTION STATUS
        # ========================================================

        if confidence is None:

            status = "Prediction generated"

        elif confidence >= 0.80:

            status = "High-confidence prediction"

        elif confidence >= 0.60:

            status = "Moderate-confidence prediction"

        else:

            status = "Low-confidence prediction"


        # ========================================================
        # PERSONALIZED RECOMMENDATION
        # ========================================================

        if prediction == "None":

            recommendation = (
                "Maintain healthy sleep habits and a consistent "
                "sleep schedule."
            )


        elif prediction == "Insomnia":

            recommendation = (
                "Maintain a consistent sleep schedule, reduce stress "
                "before bedtime, and avoid excessive screen use at night."
            )


        elif prediction == "Sleep Apnea":

            recommendation = (
                "Consider discussing your sleep pattern with a healthcare "
                "professional, especially if loud snoring or breathing "
                "interruptions occur."
            )


        else:

            recommendation = (
                "Maintain regular sleep habits and consider professional "
                "evaluation if sleep problems continue."
            )


        # ========================================================
        # PREDICTION RESULT
        # ========================================================

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)


        # --------------------------------------------------------
        # PREDICTED DISORDER
        # --------------------------------------------------------

        with result_col1:

            st.metric(
                "Predicted Sleep Disorder",
                str(prediction)
            )


        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        with result_col2:

            if confidence is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

            else:

                st.metric(
                    "Prediction Confidence",
                    "N/A"
                )


        # --------------------------------------------------------
        # RISK
        # --------------------------------------------------------

        with result_col3:

            st.metric(
                "Sleep Risk Level",
                risk_level
            )


        st.info(status)


        # ========================================================
        # CLASS PROBABILITIES
        # ========================================================

        if probabilities is not None:

            st.subheader("📊 Class Probabilities")


            class_names = label_encoder.inverse_transform(
                np.arange(len(probabilities))
            )


            probability_df = pd.DataFrame(
                {
                    "Sleep Disorder": class_names,
                    "Probability (%)": probabilities * 100
                }
            )


            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )


            # ----------------------------------------------------
            # BAR CHART
            # ----------------------------------------------------

            st.bar_chart(
                probability_df.set_index(
                    "Sleep Disorder"
                )["Probability (%)"]
            )


            # ----------------------------------------------------
            # PROBABILITY TABLE
            # ----------------------------------------------------

            display_df = probability_df.copy()


            display_df["Probability (%)"] = (
                display_df["Probability (%)"]
                .map(lambda x: f"{x:.2f}%")
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        # ========================================================
        # PERSONALIZED RECOMMENDATION
        # ========================================================

        st.subheader("💡 Personalized Recommendation")

        st.write(recommendation)


        # ========================================================
        # INPUT SUMMARY
        # ========================================================

        st.subheader("📋 Patient Input Summary")


        summary_df = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Occupation",
                    "BMI Category",
                    "Sleep Duration",
                    "Stress Level"
                ],

                "Value": [
                    age,
                    occupation,
                    bmi_category,
                    sleep_duration,
                    stress_level
                ]
            }
        )


        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ========================================================
        # DISCLAIMER
        # ========================================================

        st.warning(
            "This application is intended for educational and research "
            "purposes only and is not a medical diagnosis."
        )


    # ============================================================
    # FIXED EXCEPT BLOCK
    # ============================================================

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sleep Disorder Classification | Machine Learning Research Project"
)
