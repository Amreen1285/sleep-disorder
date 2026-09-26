import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sleep Disorder Classification",
    page_icon="😴",
    layout="wide"
)

# ------------------------------------------------------------
# Load trained project files
# ------------------------------------------------------------
MODEL_FILE = "final_sleep_disorder_svm_model.pkl"
PREPROCESSOR_FILE = "sleep_disorder_preprocessor.pkl"
LABEL_ENCODER_FILE = "sleep_disorder_label_encoder.pkl"
DATASET_FILE = "Sleep_health_and_lifestyle_dataset.csv"

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    return model, preprocessor, label_encoder

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)

try:
    model, preprocessor, label_encoder = load_model_files()
    df = load_dataset()
except Exception as e:
    st.error("Required project files could not be loaded.")
    st.code(str(e))
    st.info(
        "Keep app.py, the three .pkl files, and "
        "Sleep_health_and_lifestyle_dataset.csv in the same folder."
    )
    st.stop()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("😴 Sleep Disorder Classification System")
st.markdown(
    "### Machine Learning Research Project"
)
st.write(
    "This application uses the trained Optimized SVM model from the "
    "research pipeline to classify possible sleep-disorder categories "
    "from lifestyle and physiological information."
)

st.divider()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Project Information")
st.sidebar.write("**Model:** Optimized SVM")
st.sidebar.write("**Task:** Multiclass Classification")
st.sidebar.write(
    "**Input Features:** Age, Occupation, BMI Category, "
    "Sleep Duration, Stress Level"
)
st.sidebar.info(
    "This system provides a machine-learning prediction and is "
    "not a medical diagnosis."
)

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Enter Patient Information")

occupations = sorted(
    df["Occupation"].dropna().astype(str).unique().tolist()
)

bmi_categories = sorted(
    df["BMI Category"].dropna().astype(str).unique().tolist()
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1.0,
        max_value=120.0,
        value=25.0,
        step=1.0
    )

    occupation = st.selectbox(
        "Occupation",
        occupations
    )

    bmi_category = st.selectbox(
        "BMI Category",
        bmi_categories
    )

with col2:
    sleep_duration = st.number_input(
        "Sleep Duration (hours)",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.1
    )

    stress_level = st.number_input(
        "Stress Level",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

predict_button = st.button(
    "🔍 Predict Sleep Disorder",
    type="primary",
    use_container_width=True
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if predict_button:

    patient_data = pd.DataFrame({
        "Age": [age],
        "Occupation": [occupation],
        "BMI Category": [bmi_category],
        "Sleep Duration": [sleep_duration],
        "Stress Level": [stress_level]
    })

    try:
        patient_processed = preprocessor.transform(patient_data)

        prediction = model.predict(patient_processed)
        predicted_disorder = label_encoder.inverse_transform(
            prediction
        )[0]

        # Probability / confidence
        probabilities = None
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                patient_processed
            )[0]
            confidence = float(np.max(probabilities))

        # Status
        if str(predicted_disorder).lower() == "none":
            status = "No Sleep Disorder Detected"
        elif str(predicted_disorder).lower() == "insomnia":
            status = "Possible Insomnia Detected"
        elif str(predicted_disorder).lower() == "sleep apnea":
            status = "Possible Sleep Apnea Detected"
        else:
            status = f"Possible {predicted_disorder} Indicated"

        # Risk level follows the project logic
        if confidence is not None:
            if predicted_disorder == "None":
                risk_level = "LOW" if confidence >= 0.80 else "MODERATE"
            elif predicted_disorder in ["Insomnia", "Sleep Apnea"]:
                if confidence >= 0.80:
                    risk_level = "HIGH"
                elif confidence >= 0.60:
                    risk_level = "MODERATE"
                else:
                    risk_level = "LOW"
            else:
                risk_level = "MODERATE"
        else:
            risk_level = "NOT AVAILABLE"

        # Recommendation follows the project logic
        if predicted_disorder == "None":
            recommendation = (
                "Maintain your current healthy sleep habits and "
                "continue following a consistent sleep schedule."
            )
        elif predicted_disorder == "Insomnia":
            recommendation = (
                "Maintain a consistent sleep schedule, reduce stress "
                "before bedtime, and avoid excessive screen exposure "
                "at night."
            )
        elif predicted_disorder == "Sleep Apnea":
            recommendation = (
                "Consider discussing your sleep pattern with a "
                "healthcare professional, especially if you experience "
                "loud snoring or breathing interruptions during sleep."
            )
        else:
            recommendation = (
                "Maintain regular sleep habits and consider professional "
                "evaluation if sleep-related problems continue."
            )

        # --------------------------------------------------------
        # Result cards
        # --------------------------------------------------------
        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Predicted Sleep Disorder",
                str(predicted_disorder)
            )

        with r2:
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
                if confidence is not None else "N/A"
            )

        with r3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.success(status)

        # --------------------------------------------------------
        # Probability distribution
        # --------------------------------------------------------
        if probabilities is not None:
            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Sleep Disorder": label_encoder.classes_,
                "Probability (%)": probabilities * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Sleep Disorder")
            )

            display_df = probability_df.copy()
            display_df["Probability (%)"] = display_df[
                "Probability (%)"
            ].round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------
        st.subheader("Personalized Sleep Recommendation")
        st.info(recommendation)

        # --------------------------------------------------------
        # Input summary
        # --------------------------------------------------------
        with st.expander("View Patient Input"):
            st.dataframe(
                patient_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Note: This is a machine-learning prediction and not "
            "a medical diagnosis."
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Sleep Disorder Classification Using Machine Learning | "
    "Research Project"
)
