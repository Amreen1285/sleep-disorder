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

st.markdown(
    """
    <style>

    /* ==============================
       GENERAL
       ============================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            linear-gradient(
                135deg,
                #070b14 0%,
                #0b1120 55%,
                #0d1426 100%
            );

        color: #f4f6ff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* ==============================
       SIDEBAR
       ============================== */

    [data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #060a13,
                #0a1020
            );

        border-right:
            1px solid rgba(150,160,190,0.13);
    }

    [data-testid="stSidebar"] * {
        color: #dce4fa !important;
    }


    /* ==============================
       HEADER
       ============================== */

    .project-header {

        padding:
            8px 4px 25px 4px;

        border-bottom:
            1px solid rgba(150,160,190,0.15);

        margin-bottom:
            28px;
    }

    .project-label {

        color: #9d8cff;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1.8px;

        text-transform: uppercase;

        margin-bottom: 8px;
    }

    .project-title {

        color: #f5f7ff;

        font-size: 34px;

        font-weight: 750;

        letter-spacing: -0.8px;

        margin-bottom: 7px;
    }

    .project-description {

        color: #8f9ab5;

        font-size: 14px;

        line-height: 1.6;

        max-width: 850px;
    }


    /* ==============================
       SECTION TITLES
       ============================== */

    .section-title {

        color: #f4f6ff;

        font-size: 21px;

        font-weight: 750;

        margin-top: 28px;

        margin-bottom: 5px;
    }

    .section-description {

        color: #808ca6;

        font-size: 12px;

        margin-bottom: 17px;
    }


    /* ==============================
       OVERVIEW CARDS
       ============================== */

    .overview-card {

        background:
            rgba(17,25,47,0.72);

        border:
            1px solid rgba(150,160,190,0.14);

        border-radius: 14px;

        padding: 18px;

        min-height: 112px;

        transition: 0.2s;
    }

    .overview-card:hover {

        border-color:
            rgba(150,140,255,0.35);

        transform:
            translateY(-2px);
    }

    .overview-label {

        color: #727f9a;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1.2px;

        margin-bottom: 8px;
    }

    .overview-value {

        color: #f4f6ff;

        font-size: 18px;

        font-weight: 700;
    }

    .overview-description {

        color: #727f9a;

        font-size: 11px;

        margin-top: 5px;
    }


    /* ==============================
       CONFIGURATION CARD
       ============================== */

    .configuration-card {

        background:
            linear-gradient(
                145deg,
                rgba(24,32,56,0.82),
                rgba(14,21,39,0.82)
            );

        border:
            1px solid rgba(150,160,190,0.15);

        border-radius: 16px;

        padding: 20px;

        margin-top: 8px;
    }

    .configuration-title {

        color: #737f9a;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }

    .configuration-model {

        color: #f5f6ff;

        font-size: 22px;

        font-weight: 750;

        margin-top: 7px;
    }

    .configuration-text {

        color: #808ca6;

        font-size: 12px;

        margin-top: 5px;
    }


    /* ==============================
       INPUT CONTAINER
       ============================== */

    .input-card {

        background:
            rgba(14,21,39,0.60);

        border:
            1px solid rgba(150,160,190,0.13);

        border-radius: 16px;

        padding: 5px 18px 18px 18px;
    }


    /* ==============================
       INPUTS
       ============================== */

    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {

        background:
            #111a2e !important;

        border:
            1px solid rgba(150,160,190,0.17) !important;

        color:
            #f2f5ff !important;

        border-radius:
            10px !important;
    }

    label {

        color:
            #cbd5e1 !important;

        font-weight:
            600 !important;
    }


    /* ==============================
       PREDICTION BUTTON
       ============================== */

    div.stButton > button {

        width: 100%;

        min-height: 55px;

        border-radius: 12px;

        border:
            1px solid rgba(160,145,255,0.45);

        background:
            linear-gradient(
                135deg,
                #6654cf,
                #485bb8
            );

        color: white;

        font-size: 14px;

        font-weight: 750;

        box-shadow:
            0 10px 25px rgba(70,80,180,0.20);

        transition: 0.2s;
    }

    div.stButton > button:hover {

        border-color:
            rgba(210,205,255,0.75);

        transform:
            translateY(-1px);
    }


    /* ==============================
       RESULT CARDS
       ============================== */

    .result-card {

        background:
            linear-gradient(
                145deg,
                rgba(24,32,57,0.90),
                rgba(12,18,33,0.94)
            );

        border:
            1px solid rgba(150,160,190,0.15);

        border-radius: 16px;

        padding: 23px;

        min-height: 155px;

        text-align: center;
    }

    .result-icon {

        font-size: 31px;

        margin-bottom: 5px;
    }

    .result-label {

        color: #75819c;

        font-size: 9px;

        font-weight: 700;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }

    .result-value {

        color: #f5f6ff;

        font-size: 25px;

        font-weight: 750;

        margin-top: 7px;
    }


    /* ==============================
       INSIGHT
       ============================== */

    .insight-card {

        background:
            rgba(26,25,52,0.60);

        border-left:
            3px solid #8170e8;

        border-radius:
            0 13px 13px 0;

        padding:
            19px 21px;

        color:
            #d5dced;

        line-height:
            1.7;
    }


    /* ==============================
       SUMMARY
       ============================== */

    .summary-card {

        background:
            rgba(17,25,47,0.70);

        border:
            1px solid rgba(150,160,190,0.13);

        border-radius:
            13px;

        padding:
            15px;

        min-height:
            105px;
    }

    .summary-icon {

        font-size:
            21px;
    }

    .summary-label {

        color:
            #707d98;

        font-size:
            9px;

        font-weight:
            700;

        letter-spacing:
            1px;

        text-transform:
            uppercase;

        margin-top:
            6px;
    }

    .summary-value {

        color:
            #f2f4ff;

        font-size:
            14px;

        font-weight:
            650;

        margin-top:
            5px;
    }


    /* ==============================
       FOOTER
       ============================== */

    .footer {

        text-align:
            center;

        color:
            #59657f;

        font-size:
            10px;

        padding:
            30px 0 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
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

    model = joblib.load(
        MODEL_FILE
    )

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
# LOAD FILES
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
# PROFESSIONAL HEADER
# ============================================================

st.markdown(
    """
    <div class="project-header">

        <div class="project-label">
            MACHINE LEARNING • RESEARCH PROJECT
        </div>

        <div class="project-title">
            Sleep Disorder Classification
        </div>

        <div class="project-description">
            Machine-learning based classification of sleep disorders
            using lifestyle and physiological information.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Sleep Disorder AI"
    )

    st.caption(
        "Research Project Dashboard"
    )

    st.divider()

    st.markdown(
        "### Project Configuration"
    )

    st.markdown(
        "**Task**  \n"
        "Classification"
    )

    st.markdown(
        "**Model**  \n"
        "Optimized SVM"
    )

    st.markdown(
        "**Status**  \n"
        "🟢 Model Ready"
    )

    st.divider()

    st.markdown(
        "### Input Features"
    )

    st.markdown(
        """
        - Age
        - Occupation
        - BMI Category
        - Sleep Duration
        - Stress Level
        """
    )

    st.divider()

    st.markdown(
        "### Dataset"
    )

    st.caption(
        "Sleep Health and Lifestyle Dataset"
    )

    st.divider()

    st.caption(
        "For educational and research purposes."
    )


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Project Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Current machine-learning system configuration and status.'
    '</div>',
    unsafe_allow_html=True
)


overview1, overview2, overview3, overview4 = st.columns(
    4,
    gap="medium"
)


overview_cards = [

    (
        overview1,
        "MODEL",
        "Optimized SVM",
        "Final trained classifier"
    ),

    (
        overview2,
        "TASK",
        "Classification",
        "Sleep disorder prediction"
    ),

    (
        overview3,
        "DATASET",
        "Sleep Health",
        "Lifestyle & health data"
    ),

    (
        overview4,
        "STATUS",
        "● Ready",
        "Model loaded successfully"
    )

]


for col, label, value, description in overview_cards:

    with col:

        st.markdown(
            f"""
            <div class="overview-card">

                <div class="overview-label">
                    {label}
                </div>

                <div class="overview-value">
                    {value}
                </div>

                <div class="overview-description">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MODEL & TASK
# ============================================================

st.markdown(
    '<div class="section-title">Model & Task</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Select the task and trained model used for prediction.'
    '</div>',
    unsafe_allow_html=True
)


config_col1, config_col2 = st.columns(
    2,
    gap="large"
)


with config_col1:

    task = st.selectbox(
        "🎯 Task",
        ["Classification"]
    )


with config_col2:

    model_choice = st.selectbox(
        "🧠 Model",
        ["Optimized SVM"]
    )


st.markdown(
    """
    <div class="configuration-card">

        <div class="configuration-title">
            Active Model
        </div>

        <div class="configuration-model">
            Optimized Support Vector Machine
        </div>

        <div class="configuration-text">
            Final trained SVM model used for sleep disorder classification.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the lifestyle and physiological information required for prediction.'
    '</div>',
    unsafe_allow_html=True
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


st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)


input_col1, input_col2 = st.columns(
    2,
    gap="large"
)


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


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


st.write("")


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "RUN SLEEP DISORDER ANALYSIS",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # INPUT DATA
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
        # PREPROCESSING
        # ----------------------------------------------------

        transformed = preprocessor.transform(
            input_df
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction_encoded = model.predict(
            transformed
        )


        prediction = label_encoder.inverse_transform(
            prediction_encoded
        )[0]


        # ----------------------------------------------------
        # CONFIDENCE
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
        # RISK LEVEL
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

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Output generated by the trained Optimized SVM model.'
            '</div>',
            unsafe_allow_html=True
        )


        result1, result2, result3 = st.columns(
            3,
            gap="large"
        )


        confidence_text = (

            f"{confidence * 100:.2f}%"

            if confidence is not None

            else "N/A"
        )


        if risk == "LOW":

            risk_icon = "🟢"

        elif risk == "MODERATE":

            risk_icon = "🟠"

        else:

            risk_icon = "🔴"


        result_cards = [

            (
                result1,
                "😴",
                "PREDICTED DISORDER",
                str(prediction)
            ),

            (
                result2,
                "🎯",
                "MODEL CONFIDENCE",
                confidence_text
            ),

            (
                result3,
                risk_icon,
                "RISK LEVEL",
                risk
            )

        ]


        for col, icon, label, value in result_cards:

            with col:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-icon">
                            {icon}
                        </div>

                        <div class="result-label">
                            {label}
                        </div>

                        <div class="result-value">
                            {value}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # PROBABILITY ANALYSIS
        # ====================================================

        if probabilities is not None:

            st.markdown(
                '<div class="section-title">'
                'Probability Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Probability distribution across the available sleep disorder classes.'
                '</div>',
                unsafe_allow_html=True
            )


            class_names = label_encoder.inverse_transform(
                np.arange(
                    len(probabilities)
                )
            )


            probability_df = pd.DataFrame(
                {
                    "Sleep Disorder": class_names,
                    "Probability (%)": probabilities * 100
                }
            ).sort_values(
                "Probability (%)",
                ascending=False
            )


            chart_col, table_col = st.columns(
                [1.3, 1],
                gap="large"
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

        st.markdown(
            '<div class="section-title">'
            'Sleep Insight'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="insight-card">

                <strong>
                    Personalized Research Output
                </strong>

                <br><br>

                {recommendation}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Patient Summary'
            '</div>',
            unsafe_allow_html=True
        )


        summary_cols = st.columns(5)


        summary = [

            (
                "👤",
                "AGE",
                str(age)
            ),

            (
                "💼",
                "OCCUPATION",
                str(occupation)
            ),

            (
                "⚖️",
                "BMI",
                str(bmi)
            ),

            (
                "😴",
                "SLEEP",
                f"{sleep:.1f} hrs"
            ),

            (
                "🧠",
                "STRESS",
                f"{stress}/10"
            )

        ]


        for col, (
            icon,
            label,
            value
        ) in zip(
            summary_cols,
            summary
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="summary-card">

                        <div class="summary-icon">
                            {icon}
                        </div>

                        <div class="summary-label">
                            {label}
                        </div>

                        <div class="summary-value">
                            {value}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
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

st.markdown(
    """
    <div class="footer">
        Sleep Disorder Classification • Machine Learning Research Project • Optimized SVM
    </div>
    """,
    unsafe_allow_html=True
)
