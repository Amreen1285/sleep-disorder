# Sleep Disorder Classification App

This Streamlit app is built from the project's existing machine-learning pipeline.

## Files required in the same folder

- app.py
- final_sleep_disorder_svm_model.pkl
- sleep_disorder_preprocessor.pkl
- sleep_disorder_label_encoder.pkl
- Sleep_health_and_lifestyle_dataset.csv
- requirements.txt

## Run the app

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser should open the Streamlit application.

## What the app does

1. Accepts Age, Occupation, BMI Category, Sleep Duration, and Stress Level.
2. Uses the saved project preprocessor.
3. Uses the saved Optimized SVM research model.
4. Converts the predicted class back to the original sleep-disorder label.
5. Displays prediction confidence and class probabilities.
6. Displays the project risk-level logic.
7. Displays the project's personalized recommendation logic.

Important: the output is a machine-learning prediction, not a medical diagnosis.
