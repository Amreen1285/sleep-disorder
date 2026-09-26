import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sleep Disorder Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""<style>
.stApp {
    background-color: #080d19;
}

[data-testid="stSidebar"] {
    background-color: #0b1120;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

h1 {
    color: #f5f7ff !important;
    font-weight: 700 !important;
}

h2 {
    color: #f5f7ff !important;
}

h3 {
    color: #e8ebf5 !important;
}

p {
    color: #9aa6bf;
}

label {
    color: #d8deeb !important;
}

[data-testid="stMetric"] {
    background-color: #111a2d;
    border: 1px solid #202b43;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #7f8ba5 !important;
}

[data-testid="stMetricValue"] {
    color: #f5f7ff !important;
}

[data-baseweb="select"] > div {
    background-color: #111a2d !important;
    border-color: #283550 !important;
}

[data-testid="stNumberInput"] input {
    background-color: #111a2d !important;
    color: #f5f7ff !important;
    border-color: #283550 !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 10px;
    background-color: #6355c7;
    color: white;
    border: none;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #7466d8;
}

hr {
    border-color: #202a40;
}

</style>""", unsafe_allow_html=True)


# ============================================================
# FILE NAMES
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

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    return (
        model,
        preprocessor,
        label_encoder
    )


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATASET_FILE
    )


# ============================================================
# LOAD PROJECT
# ============================================================

try:

    model, preprocessor, label_encoder = (
        load_model_files()
    )

    df = load_dataset()

except Exception as e:

    st.error(
        "Required project files could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Sleep Disorder AI")

    st.caption(
        "Machine Learning Research Project"
    )

    st.divider()

    st.subheader("Project")

    st.write("**Task:** Classification")

    st.write("**Model:** Optimized SVM")

    st.write("**Status:** 🟢 Ready")

    st.divider()

    st.subheader("Input Features")

    st.write(
        """
        • Age

        • Occupation

        • BMI Category

        • Sleep Duration

        • Stress Level
        """
    )

    st.divider()

    st.subheader("Dataset")

    st.caption(
        "Sleep Health and Lifestyle Dataset"
    )

    st.divider()

    st.caption(
        "Educational and research use only."
    )


# ============================================================
# PROFESSIONAL HEADER
# ============================================================

st.caption(
    "MACHINE LEARNING  •  RESEARCH PROJECT"
)

st.title(
    "Sleep Disorder Classification"
)

st.write(
    "Machine-learning based classification of sleep disorders "
    "using lifestyle and physiological information."
)


st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader(
    "Project Overview"
)

st.caption(
    "Current machine-learning system configuration and status."
)


overview1, overview2, overview3, overview4 = st.columns(4)


with overview1:

    st.metric(
        "MODEL",
        "Optimized SVM"
    )

    st.caption(
        "Final trained classifier"
    )


with overview2:

    st.metric(
        "TASK",
        "Classification"
    )

    st.caption(
        "Sleep disorder prediction"
    )


with overview3:

    st.metric(
        "DATASET",
        "Sleep Health"
    )

    st.caption(
        "Lifestyle & health data"
    )


with overview4:

    st.metric(
        "STATUS",
        "Ready"
    )

    st.caption(
        "Model loaded successfully"
    )


# ============================================================
# MODEL AND TASK
# ============================================================

st.divider()

st.subheader(
    "Model & Task"
)

st.caption(
    "Select the configuration used for prediction."
)


model_col, task_col = st.columns(2)


with model_col:

    model_choice = st.selectbox(
        "🧠 Model",
        [
            "Optimized SVM"
        ]
    )


with task_col:

    task = st.selectbox(
        "🎯 Task",
        [
            "Classification"
        ]
    )


st.info(
    "Active model: Optimized Support Vector Machine (SVM)"
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Patient Information"
)

st.caption(
    "Enter the lifestyle and physiological information required "
    "for sleep disorder prediction."
)


occupations = sorted(
    df["Occupation"]
    .dropna()
    .unique()
    .tolist()
)


bmis = sorted(
    df["BMI Category"]
    .dropna()
    .unique()
    .tolist()
)


input_col1, input_col2 = st.columns(2)


# ============================================================
# LEFT INPUTS
# ============================================================

with input_col1:

    age = st.number_input(
        "👤 Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )


    occupation = st.selectbox(
        "💼 Occupation",
        occupations
    )


    bmi = st.selectbox(
        "⚖️ BMI Category",
        bmis
    )


# ============================================================
# RIGHT INPUTS
# ============================================================

with input_col2:

    sleep = st.number_input(
        "😴 Sleep Duration (hours)",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.1
    )


    stress = st.number_input(
        "🧠 Stress Level",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )


st.write("")


# ============================================================
# RUN PREDICTION
# ============================================================

if st.button(
    "RUN SLEEP DISORDER ANALYSIS",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Occupation": occupation,
                    "BMI Category": bmi,
                    "Sleep Duration": sleep,
                    "Stress Level": stress
                }
            ]
        )


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        transformed = preprocessor.transform(
            input_df
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction_encoded = model.predict(
            transformed
        )


        prediction = label_encoder.inverse_transform(
            prediction_encoded
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = None

        confidence = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                transformed
            )[0]

            confidence = float(
                max(probabilities)
            )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if prediction == "None":

            if (
                confidence is None
                or confidence >= 0.80
            ):

                risk = "LOW"

            else:

                risk = "MODERATE"


        elif prediction in [
            "Insomnia",
            "Sleep Apnea"
        ]:

            if confidence is None:

                risk = "MODERATE"

            elif confidence >= 0.80:

                risk = "HIGH"

            elif confidence >= 0.60:

                risk = "MODERATE"

            else:

                risk = "LOW"

        else:

            risk = "MODERATE"


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        if prediction == "None":

            recommendation = (
                "Maintain healthy sleep habits and "
                "follow a consistent sleep schedule."
            )

        elif prediction == "Insomnia":

            recommendation = (
                "Maintain a consistent sleep schedule, "
                "reduce stress before bedtime, and "
                "limit excessive screen use at night."
            )

        elif prediction == "Sleep Apnea":

            recommendation = (
                "Consider discussing your sleep pattern "
                "with a healthcare professional, especially "
                "if loud snoring or breathing interruptions occur."
            )

        else:

            recommendation = (
                "Maintain regular sleep habits and consider "
                "professional evaluation if sleep problems continue."
            )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Results"
        )

        st.caption(
            "Output generated by the trained Optimized SVM model."
        )


        result1, result2, result3 = st.columns(3)


        with result1:

            st.metric(
                "😴 Predicted Disorder",
                str(prediction)
            )


        with result2:

            if confidence is not None:

                confidence_text = (
                    f"{confidence * 100:.2f}%"
                )

            else:

                confidence_text = "N/A"


            st.metric(
                "🎯 Model Confidence",
                confidence_text
            )


        with result3:

            st.metric(
                "Risk Level",
                risk
            )


        # ====================================================
        # PROBABILITY ANALYSIS
        # ====================================================

        if probabilities is not None:

            st.divider()

            st.subheader(
                "Probability Analysis"
            )

            st.caption(
                "Probability distribution across the available "
                "sleep disorder classes."
            )


            class_names = label_encoder.inverse_transform(
                np.arange(
                    len(probabilities)
                )
            )


            probability_df = pd.DataFrame(
                {
                    "Sleep Disorder": class_names,
                    "Probability (%)":
                        probabilities * 100
                }
            ).sort_values(
                "Probability (%)",
                ascending=False
            )


            chart_col, table_col = st.columns(
                [1.3, 1]
            )


            with chart_col:

                st.bar_chart(
                    probability_df.set_index(
                        "Sleep Disorder"
                    )["Probability (%)"]
                )


            with table_col:

                display_df = probability_df.copy()

                display_df[
                    "Probability (%)"
                ] = display_df[
                    "Probability (%)"
                ].map(
                    lambda x:
                    f"{x:.2f}%"
                )


                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


        # ====================================================
        # SLEEP INSIGHT
        # ====================================================

        st.divider()

        st.subheader(
            "Sleep Insight"
        )

        st.info(
            recommendation
        )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.subheader(
            "Patient Summary"
        )


        summary1, summary2, summary3, summary4, summary5 = (
            st.columns(5)
        )


        with summary1:

            st.metric(
                "Age",
                str(age)
            )


        with summary2:

            st.metric(
                "Occupation",
                str(occupation)
            )


        with summary3:

            st.metric(
                "BMI",
                str(bmi)
            )


        with summary4:

            st.metric(
                "Sleep",
                f"{sleep:.1f} hrs"
            )


        with summary5:

            st.metric(
                "Stress",
                f"{stress}/10"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.write("")

        st.warning(
            "This application is intended for educational "
            "and research purposes only and is not a medical diagnosis."
        )


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
    "Sleep Disorder Classification • Machine Learning Research Project • Optimized SVM"
)
