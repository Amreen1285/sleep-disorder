import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sleep Disorder AI", page_icon="🌙", layout="wide")

st.markdown('''
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");
*{font-family:Inter,sans-serif}
.stApp{background:radial-gradient(circle at 10% 5%,#21194a55,transparent 28%),radial-gradient(circle at 90% 10%,#183d6a44,transparent 25%),linear-gradient(135deg,#070b17,#0b1020 50%,#10152a);color:#eef2ff}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080c19,#0d1223);border-right:1px solid #94a3b81f}
[data-testid="stSidebar"] *{color:#dbe4ff!important}
.block-container{max-width:1450px;padding-top:2rem}
.hero{padding:42px 44px;border:1px solid #9382ff40;border-radius:28px;background:radial-gradient(circle at 82% 25%,#8b5cf633,transparent 25%),radial-gradient(circle at 15% 85%,#3b82f622,transparent 28%),#0d1226d9;box-shadow:0 25px 70px #00000059;margin-bottom:30px}
.kicker{color:#a78bfa;font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.hero h1{font-size:clamp(36px,5vw,58px);line-height:1.05;font-weight:800;letter-spacing:-2px;color:#f7f8ff;margin:12px 0}
.hero p{max-width:850px;color:#aeb9d5;font-size:16px;line-height:1.7}
.badge{display:inline-block;padding:8px 13px;margin:5px 5px 0 0;border-radius:999px;background:#7c6aff1c;border:1px solid #8b5cf640;color:#c8c0ff;font-size:12px;font-weight:600}
.section{color:#f2f4ff;font-size:25px;font-weight:750;margin:18px 0 5px}
.sub{color:#8995b3;font-size:14px;margin-bottom:20px}
.result{padding:30px;border-radius:24px;text-align:center;background:radial-gradient(circle at 85% 20%,#8b5cf633,transparent 30%),linear-gradient(145deg,#191f3ef0,#0c1122f0);border:1px solid #8b5cf647;box-shadow:0 20px 55px #00000047}
.ri{font-size:40px}.rl{color:#939fbd;font-size:12px;text-transform:uppercase;letter-spacing:1.4px;font-weight:700;margin-top:8px}.rv{color:#f8f7ff;font-size:32px;font-weight:800;margin-top:6px}
.card{background:#11182fbd;border:1px solid #94a3b81f;border-radius:20px;padding:20px;box-shadow:0 12px 35px #0003}
.reco{border-left:3px solid #8b5cf6;background:#8b5cf614;border-radius:0 16px 16px 0;padding:20px 22px;color:#d8def0;line-height:1.7}
.metric{background:linear-gradient(145deg,#1c2343e0,#0e1327e0);border:1px solid #8b5cf633;border-radius:20px;padding:18px;min-height:125px}
.ml{color:#8793b2;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:700}.mv{color:#f7f8ff;font-size:18px;font-weight:800;margin-top:9px}
.footer{text-align:center;color:#66728f;font-size:12px;padding:28px}
div.stButton>button{width:100%;min-height:54px;border-radius:15px;border:1px solid #a78bfa73;background:linear-gradient(135deg,#6d4de6,#4f46b8);color:white;font-weight:750;box-shadow:0 12px 30px #4f46e53f}
[data-testid="stNumberInput"] input,[data-baseweb="select"]>div{background:#131b34eb!important;border-color:#94a3b829!important;color:#eef2ff!important;border-radius:12px!important}
label{color:#cbd5e1!important;font-weight:600!important}
hr{border-color:#94a3b81a}
</style>
''', unsafe_allow_html=True)

MODEL_FILE="final_sleep_disorder_svm_model.pkl"
PREPROCESSOR_FILE="sleep_disorder_preprocessor.pkl"
LABEL_ENCODER_FILE="sleep_disorder_label_encoder.pkl"
DATASET_FILE="Sleep_health_and_lifestyle_dataset.csv"

@st.cache_resource
def load_model_files():
    return (joblib.load(MODEL_FILE), joblib.load(PREPROCESSOR_FILE), joblib.load(LABEL_ENCODER_FILE))

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)

try:
    model, preprocessor, label_encoder = load_model_files()
    df = load_dataset()
except Exception as e:
    st.error("Required project files could not be loaded. Make sure the .pkl files and CSV file are in the same folder as app.py.")
    st.exception(e)
    st.stop()

st.markdown('''
<div class="hero">
<div class="kicker">🌙 AI-Powered Sleep Analysis</div>
<h1>Sleep Disorder<br>Classification</h1>
<p>An intelligent machine-learning system that analyzes lifestyle and physiological information to predict possible sleep disorder categories.</p>
<span class="badge">MACHINE LEARNING</span><span class="badge">CLASSIFICATION</span><span class="badge">OPTIMIZED SVM</span><span class="badge">RESEARCH PROJECT</span>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌙 Sleep AI")
    st.caption("Research Project Dashboard")
    st.divider()
    st.markdown("### 🧠 Model")
    st.markdown("**Optimized SVM**")
    st.markdown("### 📊 Dataset")
    st.markdown("**Sleep Health and Lifestyle Dataset**")
    st.markdown("### 🔬 Task")
    st.markdown("Classification")
    st.divider()
    st.markdown("### Input Features")
    st.markdown("- 👤 Age\n- 💼 Occupation\n- ⚖️ BMI Category\n- 😴 Sleep Duration\n- 🧠 Stress Level")

occupations=sorted(df["Occupation"].dropna().unique().tolist())
bmis=sorted(df["BMI Category"].dropna().unique().tolist())

st.markdown('<div class="section">👤 Patient Information</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Enter the required information to analyze the sleep pattern.</div>',unsafe_allow_html=True)

c1,c2=st.columns(2,gap="large")
with c1:
    age=st.number_input("👤 Age",1,100,25,1)
    occupation=st.selectbox("💼 Occupation",occupations)
    bmi=st.selectbox("⚖️ BMI Category",bmis)
with c2:
    sleep=st.number_input("😴 Sleep Duration (hours)",1.0,15.0,7.0,0.1)
    stress=st.number_input("🧠 Stress Level",1,10,5,1)
    st.markdown(f'<div class="card">Age: <b>{age}</b> &nbsp; • &nbsp; Sleep: <b>{sleep:.1f} hrs</b> &nbsp; • &nbsp; Stress: <b>{stress}/10</b></div>',unsafe_allow_html=True)

st.write("")
if st.button("🔍  ANALYZE SLEEP PATTERN",type="primary"):
    try:
        inp=pd.DataFrame([{"Age":age,"Occupation":occupation,"BMI Category":bmi,"Sleep Duration":sleep,"Stress Level":stress}])
        transformed=preprocessor.transform(inp)
        pred_enc=model.predict(transformed)
        prediction=label_encoder.inverse_transform(pred_enc)[0]
        probs=model.predict_proba(transformed)[0] if hasattr(model,"predict_proba") else None
        confidence=float(max(probs)) if probs is not None else None

        if prediction=="None":
            risk="LOW" if confidence is None or confidence>=.80 else "MODERATE"
        elif prediction in ["Insomnia","Sleep Apnea"]:
            risk="MODERATE" if confidence is None else ("HIGH" if confidence>=.80 else "MODERATE" if confidence>=.60 else "LOW")
        else: risk="MODERATE"

        if prediction=="None":
            recommendation="Maintain healthy sleep habits and a consistent sleep schedule."
        elif prediction=="Insomnia":
            recommendation="Maintain a consistent sleep schedule, reduce stress before bedtime, and avoid excessive screen use at night."
        elif prediction=="Sleep Apnea":
            recommendation="Consider discussing your sleep pattern with a healthcare professional, especially if loud snoring or breathing interruptions occur."
        else:
            recommendation="Maintain regular sleep habits and consider professional evaluation if sleep problems continue."

        st.markdown("---")
        st.markdown('<div class="section">✨ AI Analysis Result</div>',unsafe_allow_html=True)
        st.markdown('<div class="sub">Prediction generated by the optimized SVM classification model.</div>',unsafe_allow_html=True)

        a,b,c=st.columns(3,gap="large")
        conf=f"{confidence*100:.2f}%" if confidence is not None else "N/A"
        icon="🟢" if risk=="LOW" else "🟠" if risk=="MODERATE" else "🔴"
        for col,ico,label,value in [(a,"😴","Predicted Disorder",prediction),(b,"🎯","Prediction Confidence",conf),(c,icon,"Sleep Risk Level",risk)]:
            with col:
                st.markdown(f'<div class="result"><div class="ri">{ico}</div><div class="rl">{label}</div><div class="rv">{value}</div></div>',unsafe_allow_html=True)

        if probs is not None:
            st.markdown('<div class="section">📊 Prediction Probabilities</div>',unsafe_allow_html=True)
            st.markdown('<div class="sub">Probability distribution across the model classes.</div>',unsafe_allow_html=True)
            names=label_encoder.inverse_transform(np.arange(len(probs)))
            pdf=pd.DataFrame({"Sleep Disorder":names,"Probability (%)":probs*100}).sort_values("Probability (%)",ascending=False)
            x,y=st.columns([1.25,1],gap="large")
            with x: st.bar_chart(pdf.set_index("Sleep Disorder")["Probability (%)"])
            with y:
                show=pdf.copy()
                show["Probability (%)"]=show["Probability (%)"].map(lambda v:f"{v:.2f}%")
                st.dataframe(show,use_container_width=True,hide_index=True)

        st.markdown('<div class="section">💡 AI Recommendation</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="reco"><b>Personalized guidance</b><br><br>{recommendation}</div>',unsafe_allow_html=True)

        st.markdown('<div class="section">📋 Patient Summary</div>',unsafe_allow_html=True)
        cols=st.columns(5)
        data=[("👤","Age",age),("💼","Occupation",occupation),("⚖️","BMI",bmi),("😴","Sleep",f"{sleep:.1f} hrs"),("🧠","Stress",f"{stress}/10")]
        for col,(ico,label,value) in zip(cols,data):
            with col: st.markdown(f'<div class="metric"><div style="font-size:25px">{ico}</div><div class="ml">{label}</div><div class="mv">{value}</div></div>',unsafe_allow_html=True)

        st.write("")
        st.warning("⚕️ This application is intended for educational and research purposes only and is not a medical diagnosis.")
    except Exception as e:
        st.error("Prediction could not be completed.")
        st.exception(e)

st.markdown('<div class="footer">🌙 Sleep Disorder Classification • Machine Learning Research Project • Optimized SVM</div>',unsafe_allow_html=True)
