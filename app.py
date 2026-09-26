import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SleepAI • Sleep Disorder Classification",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NIGHT THEME UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 7% 8%, rgba(92,70,190,.20), transparent 25%),
        radial-gradient(circle at 92% 12%, rgba(48,117,210,.15), transparent 24%),
        linear-gradient(135deg,#050812 0%,#09101f 52%,#0c1426 100%);
    color: #f4f6ff;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#070b16,#0b1121);
    border-right: 1px solid rgba(157,170,207,.15);
}

[data-testid="stSidebar"] * {
    color: #dce4fa !important;
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(155,135,255,.24);
    border-radius: 30px;
    padding: 42px;
    background:
        radial-gradient(circle at 88% 20%,rgba(155,135,255,.18),transparent 23%),
        radial-gradient(circle at 65% 90%,rgba(103,164,255,.11),transparent 25%),
        rgba(12,18,35,.88);
    box-shadow: 0 25px 65px rgba(0,0,0,.28);
    margin-bottom: 28px;
}

/* MOON */
.moon {
    position: absolute;
    right: 65px;
    top: 38px;
    width: 105px;
    height: 105px;
    border-radius: 50%;
    background: #f4f0ff;
    box-shadow: 0 0 60px rgba(180,164,255,.35);
}

.moon:after {
    content: "";
    position: absolute;
    left: 30px;
    top: -7px;
    width: 105px;
    height: 105px;
    border-radius: 50%;
    background: #0d1426;
}

.stars {
    position: absolute;
    right: 185px;
    top: 28px;
    color: rgba(220,226,255,.7);
    font-size: 13px;
    letter-spacing: 14px;
}

.kicker {
    color: #a99aff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.2px;
    text-transform: uppercase;
}

.hero h1 {
    margin: 12px 0 10px;
    font-size: clamp(36px,5vw,58px);
    line-height: 1.03;
    letter-spacing: -2px;
    font-weight: 800;
}

.hero p {
    max-width: 820px;
    color: #aeb8d0;
    font-size: 15px;
    line-height: 1.7;
}

.badge {
    display: inline-block;
    margin: 14px 7px 0 0;
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid rgba(155,135,255,.22);
    background: rgba(155,135,255,.07);
    color: #c8c0ff;
    font-size: 11px;
    font-weight: 700;
}

/* SECTION */
.section-title {
    color: #f4f6ff;
    font-size: 24px;
    font-weight: 800;
    margin: 20px 0 5px;
}

.section-subtitle {
    color: #8f9ab5;
    font-size: 13px;
    margin-bottom: 18px;
}

/* CONFIG CARD */
.config-card {
    padding: 20px;
    border-radius: 22px;
    border: 1px solid rgba(155,135,255,.18);
    background:
        linear-gradient(
            145deg,
            rgba(27,35,65,.72),
            rgba(13,19,36,.82)
        );
}

/* RESULT */
.result-card {
    padding: 28px 20px;
    border-radius: 24px;
    text-align: center;
    border: 1px solid rgba(155,135,255,.22);
    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(155,135,255,.14),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            rgba(25,33,61,.92),
            rgba(11,17,32,.94)
        );
    min-height: 175px;
    box-shadow: 0 18px 45px rgba(0,0,0,.24);
}

.result-icon {
    font-size: 40px;
}

.result-label {
    color: #8793af;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-top: 8px;
}

.result-value {
    color: #f8f8ff;
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}

/* RECOMMENDATION */
.recommendation {
    border-left: 3px solid #9b87ff;
    border-radius: 0 17px 17px 0;
    padding: 20px 22px;
    background: rgba(155,135,255,.07);
    color: #d5dced;
    line-height: 1.7;
}

/* SUMMARY */
.summary-card {
    border: 1px solid rgba(157,170,207,.15);
    background: rgba(17,25,47,.72);
    border-radius: 18px;
    padding: 17px;
    min-height: 115px;
}

.summary-icon {
    font-size: 24px;
}

.summary-label {
    color: #7f8ba7;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 800;
    margin-top: 6px;
}

.summary-value {
    color: #f3f5ff;
    font-size: 16px;
    font-weight: 750;
    margin-top: 5px;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    min-height: 55px;
    border-radius: 15px;
    border: 1px solid rgba(179,163,255,.5);
    background: linear-gradient(135deg,#735de7,#4d62c7);
    color: white;
    font-size: 15px;
    font-weight: 800;
    box-shadow: 0 12px 30px rgba(76,91,200,.25);
}

div.stButton > button:hover {
    border-color: rgba(220,214,255,.75);
    transform: translateY(-1px);
}

/* INPUTS */
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background: #111a31 !important;
    border-color: rgba(157,170,207,.16) !important;
    color: #f2f5ff !important;
    border-radius: 12px !important;
}

label {
    color: #cbd5e1 !important;
    font-weight: 650 !important;
}

hr {
    border-color: rgba(157,170,207,.10);
}

.footer {
    text-align: center;
    color: #5f6b87;
    font-size: 11px;
    padding: 30px 0 8px;
}
</style>
""", unsafe_allow_html=True)


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

    return model, preprocessor, label_encoder


@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATASET_FILE
    )


try:

    model, preprocessor, label_encoder = load_model_files()

    df = load_dataset()

except Exception as e:

    st.error(
        "Required project files could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">

    <div class="stars">
        ✦ · ✧ · ✦
    </div>

    <div class="moon"></div>

    <div class="kicker">
        🌙 Intelligent Sleep Analytics
    </div>

    <h1>
        Sleep Disorder<br>
        Classification AI
    </h1>

    <p>
        An intelligent machine-learning system that analyzes
        lifestyle and physiological information to predict
        possible sleep disorder categories.
    </p>

    <span class="badge">
        AI / ML
    </span>

    <span class="badge">
        CLASSIFICATION
    </span>

    <span class="badge">
        OPTIMIZED SVM
    </span>

    <span class="badge">
        RESEARCH PROJECT
    </span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown("## 🌙 SleepAI")

    st.caption(
        "Research Project Dashboard"
    )

    st.divider()

    st.markdown(
        "### ⚙️ Current Configuration"
    )

    st.markdown(
        "**Task:** Classification"
    )

    st.markdown(
        "**Model:** Optimized SVM"
    )

    st.markdown(
        "**Dataset:** Sleep Health & Lifestyle"
    )

    st.divider()

    st.markdown(
        "### 📋 Input Features"
    )

    st.markdown("""
    👤 Age

    💼 Occupation

    ⚖️ BMI Category

    😴 Sleep Duration

    🧠 Stress Level
    """)

    st.divider()

    st.caption(
        "Educational and research use only."
    )


# ============================================================
# MODEL AND TASK
# ============================================================
st.markdown(
    '<div class="section-title">⚙️ Model & Task</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Choose the configuration for your analysis.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    2,
    gap="large"
)

with col1:

    task = st.selectbox(
        "🎯 Task",
        ["Classification"]
    )

with col2:

    model_choice = st.selectbox(
        "🧠 Model",
        ["Optimized SVM"]
    )


st.markdown("""
<div class="config-card">

    <b>🧠 Selected Model</b>

    <div style="
        font-size:24px;
        font-weight:800;
        margin-top:8px;
        color:#f5f3ff;
    ">
        Optimized SVM
    </div>

    <div style="
        color:#8996b3;
        font-size:12px;
        margin-top:6px;
    ">
        Final trained model • Classification task
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PATIENT INPUT
# ============================================================
st.markdown(
    '<div class="section-title">🧑‍💻 Patient Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the information used by the trained prediction pipeline.'
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


col1, col2 = st.columns(
    2,
    gap="large"
)


with col1:

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


with col2:

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
# PREDICT
# ============================================================
if st.button(
    "✨ RUN AI SLEEP ANALYSIS",
    type="primary"
):

    try:

        input_df = pd.DataFrame([
            {
                "Age": age,
                "Occupation": occupation,
                "BMI Category": bmi,
                "Sleep Duration": sleep,
                "Stress Level": stress
            }
        ])


        # PREPROCESS
        transformed = preprocessor.transform(
            input_df
        )


        # PREDICTION
        prediction_encoded = model.predict(
            transformed
        )


        prediction = label_encoder.inverse_transform(
            prediction_encoded
        )[0]


        # PROBABILITY
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


        # ====================================================
        # RISK
        # ====================================================
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


        # ====================================================
        # RECOMMENDATION
        # ====================================================
        if prediction == "None":

            recommendation = (
                "Maintain healthy sleep habits and "
                "a consistent sleep schedule."
            )

        elif prediction == "Insomnia":

            recommendation = (
                "Maintain a consistent sleep schedule, "
                "reduce stress before bedtime, and avoid "
                "excessive screen use at night."
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
        # RESULT
        # ====================================================
        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🔮 AI Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Prediction generated using the selected trained model.'
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


        risk_icon = (
            "🟢"
            if risk == "LOW"
            else "🟠"
            if risk == "MODERATE"
            else "🔴"
        )


        cards = [

            (
                result1,
                "😴",
                "Predicted Disorder",
                str(prediction)
            ),

            (
                result2,
                "🎯",
                "Model Confidence",
                confidence_text
            ),

            (
                result3,
                risk_icon,
                "Sleep Risk Level",
                risk
            )

        ]


        for col, icon, label, value in cards:

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
        # PROBABILITY
        # ====================================================
        if probabilities is not None:

            st.markdown(
                '<div class="section-title">'
                '📊 Model Probability Profile'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">'
                'Probability distribution across the available classes.'
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
                [1.25, 1],
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
        # RECOMMENDATION
        # ====================================================
        st.markdown(
            '<div class="section-title">'
            '💡 Sleep Insight'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="recommendation">

                <b>
                    Personalized research output
                </b>

                <br><br>

                {recommendation}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SUMMARY
        # ====================================================
        st.markdown(
            '<div class="section-title">'
            '📋 Patient Summary'
            '</div>',
            unsafe_allow_html=True
        )


        summary_cols = st.columns(5)


        summary = [

            (
                "👤",
                "Age",
                str(age)
            ),

            (
                "💼",
                "Occupation",
                str(occupation)
            ),

            (
                "⚖️",
                "BMI",
                str(bmi)
            ),

            (
                "😴",
                "Sleep",
                f"{sleep:.1f} hrs"
            ),

            (
                "🧠",
                "Stress",
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


        st.write("")


        st.warning(
            "⚕️ This application is intended for "
            "educational and research purposes only and "
            "is not a medical diagnosis."
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
        🌙 Sleep Disorder Classification AI
        • Machine Learning Research Project
        • Optimized SVM
    </div>
    """,
    unsafe_allow_html=True
)
