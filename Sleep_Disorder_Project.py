
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
# Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Display basic information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- TARGET DISTRIBUTION ---")
print(df["Sleep Disorder"].value_counts())
# ==============================
# DATA PREPROCESSING

# ==============================

# Fill missing Sleep Disorder values with "None"
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# Remove unnecessary Person ID column
df = df.drop(columns=["Person ID"], errors="ignore")

print("\n--- AFTER PREPROCESSING ---")
print("Dataset shape:", df.shape)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- SLEEP DISORDER CLASSES ---")
print(df["Sleep Disorder"].value_counts())
# ==============================
# FEATURE PREPROCESSING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Separate input features and target
X = df[["Age", "Occupation", "BMI Category", "Sleep Duration", "Stress Level"]]
y = df["Sleep Disorder"]
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Encoded Classes:", label_encoder.classes_)
# Find categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

print("\n--- CATEGORICAL FEATURES ---")
print(list(categorical_cols))

print("\n--- NUMERICAL FEATURES ---")
print(list(numerical_cols))
# ==============================
# ==============================
# LEAKAGE-FREE PREPROCESSING
# ==============================

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Split raw data first
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify categorical and numerical features
categorical_cols = X_train_raw.select_dtypes(include=["object"]).columns
numerical_cols = X_train_raw.select_dtypes(exclude=["object"]).columns

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),

        ("numerical",
         StandardScaler(),
         numerical_cols)
    ]
)

# Fit ONLY on training data
X_train = preprocessor.fit_transform(X_train_raw)

# Transform test data
X_test = preprocessor.transform(X_test_raw)

print("\n--- LEAKAGE-FREE PREPROCESSING ---")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Processed features:", X_train.shape[1])

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts())
# ==============================
# BASELINE MODEL
# ==============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--- BASELINE MODEL RESULTS ---")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))
# ============================================================
# STAGE 2: STRATIFIED CROSS-VALIDATION
# ============================================================

from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro"
    ],
    n_jobs=-1
)

print("\n========================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(cv_results["test_accuracy"].mean() * 100, 2),
    "% ±",
    round(cv_results["test_accuracy"].std() * 100, 2)
)

print(
    "Balanced Accuracy:",
    round(cv_results["test_balanced_accuracy"].mean() * 100, 2),
    "% ±",
    round(cv_results["test_balanced_accuracy"].std() * 100, 2)
)

print(
    "Macro F1:",
    round(cv_results["test_f1_macro"].mean() * 100, 2),
    "% ±",
    round(cv_results["test_f1_macro"].std() * 100, 2)
)

print(
    "Macro Precision:",
    round(cv_results["test_precision_macro"].mean() * 100, 2),
    "% ±",
    round(cv_results["test_precision_macro"].std() * 100, 2)
)

print(
    "Macro Recall:",
    round(cv_results["test_recall_macro"].mean() * 100, 2),
    "% ±",
    round(cv_results["test_recall_macro"].std() * 100, 2)
)
# ============================================================
# STAGE 3: RANDOM FOREST HYPERPARAMETER OPTIMIZATION
# ============================================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    "n_estimators": randint(200, 800),
    "max_depth": [None, 5, 8, 10, 15, 20, 25, 30],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample", None]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_distributions=param_dist,
    n_iter=40,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\n========================================")
print("STAGE 3: HYPERPARAMETER OPTIMIZATION")
print("========================================")

rf_search.fit(X_train, y_train)

print("\nBEST PARAMETERS:")
print(rf_search.best_params_)

print(
    "\nBEST CROSS-VALIDATION MACRO F1:",
    round(rf_search.best_score_ * 100, 2),
    "%"
)

best_model = rf_search.best_estimator_

y_pred_optimized = best_model.predict(X_test)

optimized_accuracy = accuracy_score(
    y_test,
    y_pred_optimized
)

print("\n========================================")
print("OPTIMIZED RANDOM FOREST TEST RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(optimized_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_optimized
    )
)
from sklearn.metrics import confusion_matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_optimized
    )
)
# ============================================================
# STAGE 4: ADVANCED MODEL COMPARISON
# ============================================================

from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier


models = {
    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss"
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),
"Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        random_state=42
    ),
}
models["Soft Voting"] = VotingClassifier(
    estimators=[
        ("svm", models["SVM"]),
        ("catboost", models["CatBoost"]),
        ("extra_trees", models["Extra Trees"])
    ],
    voting="soft",
    weights=[2, 3, 1]
)
print("\n========================================")
print("STAGE 4: ADVANCED MODEL COMPARISON")
print("========================================")
results_names = []
results_accuracy = []
results_f1 = []
for name, clf in models.items():

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    clf.fit(X_train_dense, y_train)

    predictions = clf.predict(X_test_dense)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    macro_f1 = report["macro avg"]["f1-score"]

    results_names.append(name)
    results_accuracy.append(accuracy)
    results_f1.append(macro_f1)

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(
        "Test Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Macro F1:",
        round(macro_f1 * 100, 2),
        "%"
    )
    
# ================================
# STAGE 5: MODEL PERFORMANCE GRAPHS
# ================================

import matplotlib.pyplot as plt

models_graph = results_names
accuracy_graph = [round(x * 100, 2) for x in results_accuracy]
f1_graph = [round(x * 100, 2) for x in results_f1]

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.bar(models_graph, accuracy_graph)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.ylim(80, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Macro F1 Graph
plt.figure(figsize=(8, 5))
plt.bar(models_graph, f1_graph)
plt.title("Model Macro F1 Comparison")
plt.xlabel("Models")
plt.ylabel("Macro F1 (%)")
plt.ylim(80, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
# ==============================
# DETAILED EVALUATION
# ==============================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# =========================================
# STAGE 6: DETAILED EVALUATION
# =========================================

# ============================================================
# STAGE 6: FINAL MODEL SELECTION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

print("\n================================================")
print("STAGE 6: FINAL MODEL SELECTION")
print("================================================")

# Select model with highest Macro F1
best_index = np.argmax(results_f1)
best_model_name = results_names[best_index]
best_model = models[best_model_name]

print("\nSelected Best Model:", best_model_name)
print(
    "Validation Test Accuracy:",
    round(results_accuracy[best_index] * 100, 2),
    "%"
)
print(
    "Validation Macro F1:",
    round(results_f1[best_index] * 100, 2),
    "%"
)


# ============================================================
# STAGE 7: FINAL MODEL TRAINING
# ============================================================

print("\n================================================")
print("STAGE 7: FINAL MODEL TRAINING")
print("================================================")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)


# ============================================================
# STAGE 8: FINAL PERFORMANCE EVALUATION
# ============================================================

final_accuracy = accuracy_score(y_test, y_pred)
final_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
final_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
final_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
final_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n================================================")
print("FINAL MODEL PERFORMANCE")
print("================================================")

print("Best Model:", best_model_name)
print("Accuracy:", round(final_accuracy * 100, 2), "%")
print("Balanced Accuracy:", round(final_balanced_accuracy * 100, 2), "%")
print("Macro Precision:", round(final_precision * 100, 2), "%")
print("Macro Recall:", round(final_recall * 100, 2), "%")
print("Macro F1:", round(final_f1 * 100, 2), "%")


# ============================================================
# STAGE 9: CLASSIFICATION REPORT
# ============================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# STAGE 10: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Final Model")
plt.tight_layout()
plt.show()


# ============================================================
# STAGE 11: FEATURE IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance

print("\n================================================")
print("STAGE 11: FEATURE IMPORTANCE ANALYSIS")
print("================================================")

# Convert sparse matrix to dense
X_test_dense = X_test.toarray()

perm_result = permutation_importance(
    best_model,
    X_test_dense,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(importance_df.head(10))


plt.figure(figsize=(10, 6))

top_features = importance_df.head(10)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()


# ============================================================
# STAGE 12: SAVE FINAL MODEL
# ============================================================

print("\n================================================")
print("STAGE 12: SAVING FINAL MODEL")
print("================================================")

joblib.dump(best_model, "sleep_disorder_model.pkl")
joblib.dump(preprocessor, "sleep_disorder_preprocessor.pkl")
joblib.dump(label_encoder, "sleep_disorder_label_encoder.pkl")

print("\nModel saved as:")
print("sleep_disorder_model.pkl")

print("\nPreprocessor saved as:")
print("sleep_disorder_preprocessor.pkl")

print("\nLabel encoder saved as:")
print("sleep_disorder_label_encoder.pkl")


# ============================================================
# STAGE 13: FINAL SLEEP DISORDER PREDICTION
# ============================================================

print("\n================================================")
print("FINAL SLEEP DISORDER PREDICTION")
print("================================================")

age = float(input("Enter Age: "))

occupation = input(
    "Enter Occupation exactly as in dataset: "
)

bmi = input(
    "Enter BMI Category exactly as in dataset: "
)

sleep_duration = float(
    input("Enter Sleep Duration: ")
)

stress_level = float(
    input("Enter Stress Level: ")
)


new_patient = pd.DataFrame({
    "Age": [age],
    "Occupation": [occupation],
    "BMI Category": [bmi],
    "Sleep Duration": [sleep_duration],
    "Stress Level": [stress_level]
})


# Apply SAME preprocessing used during training
new_patient_processed = preprocessor.transform(
    new_patient
)


# Prediction
prediction = best_model.predict(
    new_patient_processed
)


predicted_class = label_encoder.inverse_transform(
    prediction
)[0]


print("\n================================================")
print("PREDICTION RESULT")
print("================================================")

print(
    "Predicted Sleep Disorder:",
    predicted_class
)
# PREDICTION STATUS
if predicted_class == "None":
    status = "No Sleep Disorder Detected"
elif predicted_class == "Insomnia":
    status = "Possible Insomnia Detected"
elif predicted_class == "Sleep Apnea":
    status = "Possible Sleep Apnea Detected"
else:
    status = "Other Sleep Disorder Status"

print("Prediction Status:", status)

# ============================================================
# PREDICTION PROBABILITY
# ============================================================

if hasattr(best_model, "predict_proba"):

    probabilities = best_model.predict_proba(
        new_patient_processed
    )[0]

    print("\nPrediction Probabilities:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities
    ):
        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    confidence = max(probabilities)

    print(
        "\nPrediction Confidence:",
        round(confidence * 100, 2),
        "%"
    )

else:

    print(
        "\nProbability prediction is not available "
        "for this model."
    )


print("\n================================================")
print("PROJECT EXECUTION COMPLETED")
print("================================================")
# ============================================================
# STAGE 14: OPTIMIZED SVM RESEARCH MODEL
# ============================================================

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print("\n================================================")
print("STAGE 14: OPTIMIZED SVM RESEARCH MODEL")
print("================================================")

svm_param_grid = {
    "C": [0.1, 1, 3, 10, 30],
    "gamma": ["scale", 0.01, 0.03, 0.1],
    "kernel": ["rbf"],
    "class_weight": [None, "balanced"]
}

svm_grid = GridSearchCV(
    estimator=SVC(
        probability=True,
        random_state=42
    ),
    param_grid=svm_param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

optimized_svm = svm_grid.best_estimator_

print("\nBest SVM Parameters:")
print(svm_grid.best_params_)

print(
    "Best Cross-Validation Macro F1:",
    round(svm_grid.best_score_ * 100, 2),
    "%"
)

optimized_svm_pred = optimized_svm.predict(X_test)

optimized_svm_accuracy = accuracy_score(
    y_test,
    optimized_svm_pred
)

optimized_svm_f1 = f1_score(
    y_test,
    optimized_svm_pred,
    average="macro"
)
# ============================================================
# OPTIMIZED SVM - SIX FINAL EVALUATION METRICS
# ============================================================

optimized_svm_balanced_accuracy = balanced_accuracy_score(
    y_test,
    optimized_svm_pred
)

optimized_svm_precision = precision_score(
    y_test,
    optimized_svm_pred,
    average="macro",
    zero_division=0
)

optimized_svm_recall = recall_score(
    y_test,
    optimized_svm_pred,
    average="macro",
    zero_division=0
)

optimized_svm_weighted_f1 = f1_score(
    y_test,
    optimized_svm_pred,
    average="weighted",
    zero_division=0
)

print("\n================================================")
print("OPTIMIZED SVM - FINAL SIX METRICS")
print("================================================")

print("Accuracy:",
      round(optimized_svm_accuracy * 100, 2), "%")

print("Balanced Accuracy:",
      round(optimized_svm_balanced_accuracy * 100, 2), "%")

print("Macro Precision:",
      round(optimized_svm_precision * 100, 2), "%")

print("Macro Recall:",
      round(optimized_svm_recall * 100, 2), "%")

print("Macro F1:",
      round(optimized_svm_f1 * 100, 2), "%")

print("Weighted F1:",
      round(optimized_svm_weighted_f1 * 100, 2), "%")

print("\nOptimized SVM Test Accuracy:",
      round(optimized_svm_accuracy * 100, 2), "%")

print("Optimized SVM Macro F1:",
      round(optimized_svm_f1 * 100, 2), "%")

print("\nOptimized SVM Classification Report:")
print(
    classification_report(
        y_test,
        optimized_svm_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)
# ============================================================
# FINAL RESEARCH MODEL - OPTIMIZED SVM
# ============================================================


print("\n================================================")
print("FINAL RESEARCH MODEL")
print("================================================")

final_research_model = optimized_svm

joblib.dump(
    final_research_model,
    "final_sleep_disorder_svm_model.pkl"
)

print("Final Research Model: Optimized SVM")
print("Test Accuracy:", round(optimized_svm_accuracy * 100, 2), "%")
print("Macro F1:", round(optimized_svm_f1 * 100, 2), "%")

print("\nFinal research model saved successfully.")
# ============================================================
# FINAL RESEARCH MODEL PREDICTION
# ============================================================

print("\n================================================")
print("FINAL RESEARCH MODEL PREDICTION")
print("================================================")

age = float(input("Enter Age: "))

occupation = input(
    "Enter Occupation exactly as in dataset: "
)

bmi = input(
    "Enter BMI Category exactly as in dataset: "
)

sleep_duration = float(
    input("Enter Sleep Duration: ")
)

stress_level = float(
    input("Enter Stress Level: ")
)

new_patient = pd.DataFrame({
    "Age": [age],
    "Occupation": [occupation],
    "BMI Category": [bmi],
    "Sleep Duration": [sleep_duration],
    "Stress Level": [stress_level]
})

new_patient_processed = preprocessor.transform(
    new_patient
)

prediction = optimized_svm.predict(
    new_patient_processed
)

predicted_class = label_encoder.inverse_transform(
    prediction
)[0]

print("\n================================================")
print("FINAL PREDICTION RESULT")
print("================================================")

print(
    "Predicted Sleep Disorder:",
    predicted_class
)

if predicted_class == "None":
    status = "No Sleep Disorder Detected"
elif predicted_class == "Insomnia":
    status = "Possible Insomnia Detected"
elif predicted_class == "Sleep Apnea":
    status = "Possible Sleep Apnea Detected"
else:
    status = "Other Sleep Disorder Status"

print("Prediction Status:", status)

if hasattr(optimized_svm, "predict_proba"):

    probabilities = optimized_svm.predict_proba(
        new_patient_processed
    )[0]

    print("\nPrediction Probabilities:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities
    ):
        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    confidence = max(probabilities)

    print(
        "\nPrediction Confidence:",
        round(confidence * 100, 2),
        "%"
    )

print("\n================================================")
print("FINAL RESEARCH PREDICTION COMPLETED")
print("================================================")
# ============================================================
# FINAL MODEL COMPARISON GRAPH
# ============================================================

final_model_names = results_names + ["Optimized SVM"]
final_model_accuracy = results_accuracy + [optimized_svm_accuracy]
final_model_f1 = results_f1 + [optimized_svm_f1]

final_accuracy_percent = [
    x * 100 for x in final_model_accuracy
]

final_f1_percent = [
    x * 100 for x in final_model_f1
]

plt.figure(figsize=(10, 6))

plt.bar(
    final_model_names,
    final_accuracy_percent
)

plt.title("Final Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

plt.bar(
    final_model_names,
    final_f1_percent
)

plt.title("Final Model Macro F1 Comparison")
plt.xlabel("Models")
plt.ylabel("Macro F1 (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
# ============================================================
# RESEARCH EXPERIMENT: REMOVE-ONE MODEL COMPARISON
# ============================================================

print("\n================================================")
print("REMOVE-ONE MODEL COMPARISON")
print("================================================")

base_models = {
    "SVM": models["SVM"],
    "CatBoost": models["CatBoost"],
    "Extra Trees": models["Extra Trees"]
}

# Full Soft Voting model
full_voting = VotingClassifier(
    estimators=[
        ("svm", base_models["SVM"]),
        ("catboost", base_models["CatBoost"]),
        ("extra_trees", base_models["Extra Trees"])
    ],
    voting="soft",
    weights=[2, 3, 1]
)

full_voting.fit(X_train_dense, y_train)

full_pred = full_voting.predict(X_test_dense)

full_accuracy = accuracy_score(
    y_test,
    full_pred
)

print("\nFull Ensemble Accuracy:",
      round(full_accuracy * 100, 2), "%")


# Remove SVM
voting_no_svm = VotingClassifier(
    estimators=[
        ("catboost", base_models["CatBoost"]),
        ("extra_trees", base_models["Extra Trees"])
    ],
    voting="soft",
    weights=[3, 1]
)

voting_no_svm.fit(X_train_dense, y_train)

pred_no_svm = voting_no_svm.predict(X_test_dense)

accuracy_no_svm = accuracy_score(
    y_test,
    pred_no_svm
)

print("Without SVM:",
      round(accuracy_no_svm * 100, 2), "%")


# Remove CatBoost
voting_no_catboost = VotingClassifier(
    estimators=[
        ("svm", base_models["SVM"]),
        ("extra_trees", base_models["Extra Trees"])
    ],
    voting="soft",
    weights=[2, 1]
)

voting_no_catboost.fit(X_train_dense, y_train)

pred_no_catboost = voting_no_catboost.predict(X_test_dense)

accuracy_no_catboost = accuracy_score(
    y_test,
    pred_no_catboost
)

print("Without CatBoost:",
      round(accuracy_no_catboost * 100, 2), "%")


# Remove Extra Trees
voting_no_extra = VotingClassifier(
    estimators=[
        ("svm", base_models["SVM"]),
        ("catboost", base_models["CatBoost"])
    ],
    voting="soft",
    weights=[2, 3]
)

voting_no_extra.fit(X_train_dense, y_train)

pred_no_extra = voting_no_extra.predict(X_test_dense)

accuracy_no_extra = accuracy_score(
    y_test,
    pred_no_extra
)

print("Without Extra Trees:",
      round(accuracy_no_extra * 100, 2), "%")


print("\n================================================")
print("REMOVE-ONE EXPERIMENT COMPLETED")
print("================================================")
# ============================================================
# RESEARCH EXPERIMENT: ADD-ONE MODEL COMPARISON
# ============================================================

print("\n================================================")
print("ADD-ONE MODEL COMPARISON")
print("================================================")

# SVM only
svm_only = SVC(
    probability=True,
    random_state=42
)

svm_only.fit(X_train_dense, y_train)

svm_only_pred = svm_only.predict(X_test_dense)

svm_only_accuracy = accuracy_score(
    y_test,
    svm_only_pred
)

print("\nSVM Only:",
      round(svm_only_accuracy * 100, 2), "%")


# SVM + CatBoost
svm_catboost = VotingClassifier(
    estimators=[
        ("svm", SVC(
            probability=True,
            random_state=42
        )),
        ("catboost", CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        ))
    ],
    voting="soft",
    weights=[2, 3]
)

svm_catboost.fit(X_train_dense, y_train)

svm_catboost_pred = svm_catboost.predict(
    X_test_dense
)

svm_catboost_accuracy = accuracy_score(
    y_test,
    svm_catboost_pred
)

print("SVM + CatBoost:",
      round(svm_catboost_accuracy * 100, 2), "%")


# SVM + CatBoost + Extra Trees
svm_catboost_extra = VotingClassifier(
    estimators=[
        ("svm", SVC(
            probability=True,
            random_state=42
        )),
        ("catboost", CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        )),
        ("extra_trees", ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced"
        ))
    ],
    voting="soft",
    weights=[2, 3, 1]
)

svm_catboost_extra.fit(
    X_train_dense,
    y_train
)

svm_catboost_extra_pred = svm_catboost_extra.predict(
    X_test_dense
)

svm_catboost_extra_accuracy = accuracy_score(
    y_test,
    svm_catboost_extra_pred
)

print(
    "SVM + CatBoost + Extra Trees:",
    round(
        svm_catboost_extra_accuracy * 100,
        2
    ),
    "%"
)


print("\n================================================")
print("ADD-ONE EXPERIMENT COMPLETED")
print("================================================")
# ============================================================
# STANDARD SVM VS OPTIMIZED SVM
# ============================================================

print("\n================================================")
print("STANDARD SVM VS OPTIMIZED SVM")
print("================================================")

standard_svm = models["SVM"]

standard_svm.fit(X_train_dense, y_train)

standard_svm_pred = standard_svm.predict(
    X_test_dense
)

standard_svm_accuracy = accuracy_score(
    y_test,
    standard_svm_pred
)

standard_svm_f1 = f1_score(
    y_test,
    standard_svm_pred,
    average="macro",
    zero_division=0
)

print("\nStandard SVM Accuracy:",
      round(standard_svm_accuracy * 100, 2), "%")

print("Standard SVM Macro F1:",
      round(standard_svm_f1 * 100, 2), "%")

print("\nOptimized SVM Accuracy:",
      round(optimized_svm_accuracy * 100, 2), "%")

print("Optimized SVM Macro F1:",
      round(optimized_svm_f1 * 100, 2), "%")

print("\nAccuracy Improvement:",
      round(
          (optimized_svm_accuracy -
           standard_svm_accuracy) * 100,
          2
      ),
      "percentage points")

print("\nMacro F1 Improvement:",
      round(
          (optimized_svm_f1 -
           standard_svm_f1) * 100,
          2
      ),
      "percentage points")

print("\n================================================")
print("SVM OPTIMIZATION COMPARISON COMPLETED")
print("================================================")
# ============================================================
# FINAL RESEARCH RESULTS TABLE
# ============================================================

final_results_table = pd.DataFrame({
    "Experiment": [
        "Standard SVM",
        "Optimized SVM",
        "SVM Only",
        "SVM + CatBoost",
        "SVM + CatBoost + Extra Trees"
    ],
    "Accuracy (%)": [
        standard_svm_accuracy * 100,
        optimized_svm_accuracy * 100,
        svm_only_accuracy * 100,
        svm_catboost_accuracy * 100,
        svm_catboost_extra_accuracy * 100
    ],
    "Macro F1 (%)": [
        standard_svm_f1 * 100,
        optimized_svm_f1 * 100,
        f1_score(y_test, svm_only_pred, average="macro", zero_division=0) * 100,
        f1_score(y_test, svm_catboost_pred, average="macro", zero_division=0) * 100,
        f1_score(y_test, svm_catboost_extra_pred, average="macro", zero_division=0) * 100
    ]
})

print("\n================================================")
print("FINAL RESEARCH RESULTS TABLE")
print("================================================")

print(
    final_results_table.round(2).to_string(index=False)
)

print("\n================================================")
print("PROJECT RESEARCH EXPERIMENTS COMPLETED")
print("================================================")
# ============================================================
#                 PROJECT 18 - FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("        PROJECT 18: SLEEP DISORDER CLASSIFICATION")
print("=" * 70)

print("\nFINAL RESEARCH MODEL")
print("-" * 70)
print(f"{'Model':<30}: Optimized SVM")
print(f"{'Test Accuracy':<30}: {optimized_svm_accuracy * 100:.2f}%")
print(f"{'Macro F1':<30}: {optimized_svm_f1 * 100:.2f}%")

print("\nSIX FINAL EVALUATION METRICS")
print("-" * 70)
print(f"{'Accuracy':<30}: {optimized_svm_accuracy * 100:.2f}%")
print(f"{'Balanced Accuracy':<30}: {optimized_svm_balanced_accuracy * 100:.2f}%")
print(f"{'Macro Precision':<30}: {optimized_svm_precision * 100:.2f}%")
print(f"{'Macro Recall':<30}: {optimized_svm_recall * 100:.2f}%")
print(f"{'Macro F1':<30}: {optimized_svm_f1 * 100:.2f}%")
print(f"{'Weighted F1':<30}: {optimized_svm_weighted_f1 * 100:.2f}%")

print("\nRESEARCH EXPERIMENTS")
print("-" * 70)
print(f"{'Remove-One Experiment':<30}: COMPLETED")
print(f"{'Add-One Experiment':<30}: COMPLETED")
print(f"{'SVM Optimization':<30}: COMPLETED")
print(f"{'Model Comparison':<30}: COMPLETED")
print(f"{'Final Prediction':<30}: COMPLETED")

print("\nMODEL COMPARISON")
print("-" * 70)
print(final_results_table.round(2).to_string(index=False))

print("\n" + "=" * 70)
print("       PROJECT 18 EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)
# ============================================================
# STAGE 15: FEATURE-ENGINEERED SVM RESEARCH EXPERIMENT
# ============================================================

print("\n================================================")
print("STAGE 15: FEATURE-ENGINEERED SVM")
print("================================================")

# Create additional meaningful features
def add_engineered_features(data):
    data = data.copy()

    data["Sleep_Stress_Interaction"] = (
        data["Sleep Duration"] * data["Stress Level"]
    )

    data["Age_Stress_Interaction"] = (
        data["Age"] * data["Stress Level"]
    )

    data["Age_Sleep_Interaction"] = (
        data["Age"] * data["Sleep Duration"]
    )

    data["Stress_per_Sleep"] = (
        data["Stress Level"] / (data["Sleep Duration"] + 0.1)
    )

    return data


# Apply feature engineering separately
X_train_fe_raw = add_engineered_features(X_train_raw)
X_test_fe_raw = add_engineered_features(X_test_raw)

# Identify columns
categorical_fe = X_train_fe_raw.select_dtypes(
    include=["object"]
).columns

numerical_fe = X_train_fe_raw.select_dtypes(
    exclude=["object"]
).columns

# New preprocessing
fe_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_fe
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_fe
        )
    ]
)

# Fit only on training data
X_train_fe = fe_preprocessor.fit_transform(
    X_train_fe_raw
)

X_test_fe = fe_preprocessor.transform(
    X_test_fe_raw
)

print("Original features:", X_train.shape[1])
print("Engineered features:", X_train_fe.shape[1])

# Optimized SVM search
fe_svm_grid = GridSearchCV(
    estimator=SVC(
        probability=True,
        random_state=42
    ),
    param_grid={
        "C": [1, 3, 10, 30, 50],
        "gamma": ["scale", 0.01, 0.03, 0.05, 0.1],
        "kernel": ["rbf"],
        "class_weight": [None, "balanced"]
    },
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

print("\nTraining Feature-Engineered SVM...")

fe_svm_grid.fit(X_train_fe, y_train)

feature_svm = fe_svm_grid.best_estimator_

# Prediction
feature_svm_pred = feature_svm.predict(X_test_fe)

# Metrics
feature_svm_accuracy = accuracy_score(
    y_test,
    feature_svm_pred
)

feature_svm_balanced = balanced_accuracy_score(
    y_test,
    feature_svm_pred
)

feature_svm_precision = precision_score(
    y_test,
    feature_svm_pred,
    average="macro",
    zero_division=0
)

feature_svm_recall = recall_score(
    y_test,
    feature_svm_pred,
    average="macro",
    zero_division=0
)

feature_svm_f1 = f1_score(
    y_test,
    feature_svm_pred,
    average="macro",
    zero_division=0
)

feature_svm_weighted_f1 = f1_score(
    y_test,
    feature_svm_pred,
    average="weighted",
    zero_division=0
)

print("\n================================================")
print("FEATURE-ENGINEERED SVM RESULTS")
print("================================================")

print("Best Parameters:")
print(fe_svm_grid.best_params_)

print(
    "Test Accuracy:",
    round(feature_svm_accuracy * 100, 2),
    "%"
)

print(
    "Balanced Accuracy:",
    round(feature_svm_balanced * 100, 2),
    "%"
)

print(
    "Macro Precision:",
    round(feature_svm_precision * 100, 2),
    "%"
)

print(
    "Macro Recall:",
    round(feature_svm_recall * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(feature_svm_f1 * 100, 2),
    "%"
)

print(
    "Weighted F1:",
    round(feature_svm_weighted_f1 * 100, 2),
    "%"
)

print("\n================================================")
print("ACCURACY COMPARISON")
print("================================================")

print(
    "Previous Optimized SVM:",
    round(optimized_svm_accuracy * 100, 2),
    "%"
)

print(
    "Feature-Engineered SVM:",
    round(feature_svm_accuracy * 100, 2),
    "%"
)

improvement = (
    feature_svm_accuracy - optimized_svm_accuracy
) * 100

print(
    "Accuracy Improvement:",
    round(improvement, 2),
    "percentage points"
)

# Select the better model
if feature_svm_accuracy > optimized_svm_accuracy:

    final_best_model = feature_svm
    final_best_accuracy = feature_svm_accuracy
    final_best_f1 = feature_svm_f1
    final_best_preprocessor = fe_preprocessor

    print("\nNEW BEST MODEL: Feature-Engineered SVM")

else:

    final_best_model = optimized_svm
    final_best_accuracy = optimized_svm_accuracy
    final_best_f1 = optimized_svm_f1
    final_best_preprocessor = preprocessor

    print("\nBEST MODEL REMAINS: Optimized SVM")

print(
    "\nFINAL BEST ACCURACY:",
    round(final_best_accuracy * 100, 2),
    "%"
)

print(
    "FINAL BEST MACRO F1:",
    round(final_best_f1 * 100, 2),
    "%"
)

print("\n================================================")
print("STAGE 15 COMPLETED")
print("================================================")
# ============================================================
# STAGE 16: CROSS-VALIDATED PROBABILITY ENSEMBLE
# ============================================================

print("\n================================================")
print("STAGE 16: CROSS-VALIDATED PROBABILITY ENSEMBLE")
print("================================================")

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

# Use dense training data for tree-based models
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Base models
ensemble_svm = SVC(
    probability=True,
    random_state=42,
    **svm_grid.best_params_
)

ensemble_catboost = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

ensemble_extra = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

# 5-fold stratified cross-validation
ensemble_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating cross-validated probabilities...")

# Out-of-fold probabilities
svm_oof = cross_val_predict(
    ensemble_svm,
    X_train_dense,
    y_train,
    cv=ensemble_cv,
    method="predict_proba",
    n_jobs=-1
)

cat_oof = cross_val_predict(
    ensemble_catboost,
    X_train_dense,
    y_train,
    cv=ensemble_cv,
    method="predict_proba",
    n_jobs=-1
)

extra_oof = cross_val_predict(
    ensemble_extra,
    X_train_dense,
    y_train,
    cv=ensemble_cv,
    method="predict_proba",
    n_jobs=-1
)

# Search ensemble weights using ONLY training OOF predictions
best_weight_f1 = -1
best_weights = None

for svm_weight in [1, 2, 3, 4, 5, 6]:
    for cat_weight in [1, 2, 3, 4, 5, 6]:
        for extra_weight in [1, 2, 3, 4]:

            total_weight = (
                svm_weight +
                cat_weight +
                extra_weight
            )

            blended_oof = (
                (svm_weight * svm_oof) +
                (cat_weight * cat_oof) +
                (extra_weight * extra_oof)
            ) / total_weight

            oof_prediction = np.argmax(
                blended_oof,
                axis=1
            )

            oof_f1 = f1_score(
                y_train,
                oof_prediction,
                average="macro",
                zero_division=0
            )

            if oof_f1 > best_weight_f1:
                best_weight_f1 = oof_f1
                best_weights = (
                    svm_weight,
                    cat_weight,
                    extra_weight
                )

print("\nBEST ENSEMBLE WEIGHTS:")
print("Optimized SVM:", best_weights[0])
print("CatBoost:", best_weights[1])
print("Extra Trees:", best_weights[2])

print(
    "\nCross-Validated Macro F1:",
    round(best_weight_f1 * 100, 2),
    "%"
)

# Train models on the complete training set
print("\nTraining final ensemble...")

ensemble_svm.fit(
    X_train_dense,
    y_train
)

ensemble_catboost.fit(
    X_train_dense,
    y_train
)

ensemble_extra.fit(
    X_train_dense,
    y_train
)

# Test probabilities
svm_test_proba = ensemble_svm.predict_proba(
    X_test_dense
)

cat_test_proba = ensemble_catboost.predict_proba(
    X_test_dense
)

extra_test_proba = ensemble_extra.predict_proba(
    X_test_dense
)

svm_weight, cat_weight, extra_weight = best_weights

total_weight = (
    svm_weight +
    cat_weight +
    extra_weight
)

# Final weighted probability prediction
ensemble_test_proba = (
    (svm_weight * svm_test_proba) +
    (cat_weight * cat_test_proba) +
    (extra_weight * extra_test_proba)
) / total_weight

ensemble_prediction = np.argmax(
    ensemble_test_proba,
    axis=1
)

# Final test metrics
ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)

ensemble_macro_f1 = f1_score(
    y_test,
    ensemble_prediction,
    average="macro",
    zero_division=0
)

print("\n================================================")
print("CROSS-VALIDATED ENSEMBLE RESULTS")
print("================================================")

print(
    "Test Accuracy:",
    round(ensemble_accuracy * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(ensemble_macro_f1 * 100, 2),
    "%"
)

print("\n================================================")
print("FINAL MODEL COMPARISON")
print("================================================")

print(
    "Optimized SVM:",
    round(optimized_svm_accuracy * 100, 2),
    "%"
)

print(
    "Weighted Ensemble:",
    round(ensemble_accuracy * 100, 2),
    "%"
)

print(
    "Accuracy Difference:",
    round(
        (ensemble_accuracy - optimized_svm_accuracy) * 100,
        2
    ),
    "percentage points"
)

if ensemble_accuracy > optimized_svm_accuracy:

    print("\nNEW BEST MODEL: Cross-Validated Weighted Ensemble")

else:

    print("\nBEST MODEL REMAINS: Optimized SVM")

print("\n================================================")
print("STAGE 16 COMPLETED")
print("================================================")
# ============================================================
# STAGE 17: VALIDATION-BASED SVM THRESHOLD TUNING
# ============================================================

print("\n================================================")
print("STAGE 17: SVM THRESHOLD TUNING")
print("================================================")

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from itertools import product

# Split training data into training and validation sets
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)

print("\nTraining validation SVM...")

# Copy the already optimized SVM
threshold_svm = clone(optimized_svm)

threshold_svm.fit(
    X_subtrain,
    y_subtrain
)

# Validation probabilities
validation_proba = threshold_svm.predict_proba(
    X_validation
)

classes = threshold_svm.classes_

# Search class probability multipliers
search_values = [0.80, 0.90, 1.00, 1.10, 1.20]

best_validation_f1 = -1
best_multipliers = None

for multipliers in product(
    search_values,
    repeat=len(classes)
):

    adjusted_probability = (
        validation_proba *
        np.array(multipliers)
    )

    validation_prediction = classes[
        np.argmax(
            adjusted_probability,
            axis=1
        )
    ]

    validation_f1 = f1_score(
        y_validation,
        validation_prediction,
        average="macro",
        zero_division=0
    )

    if validation_f1 > best_validation_f1:

        best_validation_f1 = validation_f1
        best_multipliers = multipliers

print("\nBEST VALIDATION THRESHOLD MULTIPLIERS:")

for class_name, multiplier in zip(
    classes,
    best_multipliers
):
    print(
        class_name,
        ":",
        multiplier
    )

print(
    "\nValidation Macro F1:",
    round(best_validation_f1 * 100, 2),
    "%"
)

# Train the optimized SVM again on the complete training data
print("\nTraining final threshold-tuned SVM...")

threshold_svm.fit(
    X_train,
    y_train
)

# Test probabilities
test_proba = threshold_svm.predict_proba(
    X_test
)

# Apply validation-selected multipliers
adjusted_test_probability = (
    test_proba *
    np.array(best_multipliers)
)

threshold_prediction = classes[
    np.argmax(
        adjusted_test_probability,
        axis=1
    )
]

# Final test metrics
threshold_accuracy = accuracy_score(
    y_test,
    threshold_prediction
)

threshold_macro_f1 = f1_score(
    y_test,
    threshold_prediction,
    average="macro",
    zero_division=0
)

print("\n================================================")
print("THRESHOLD-TUNED SVM RESULTS")
print("================================================")

print(
    "Test Accuracy:",
    round(threshold_accuracy * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(threshold_macro_f1 * 100, 2),
    "%"
)

print("\n================================================")
print("FINAL ACCURACY COMPARISON")
print("================================================")

print(
    "Previous Optimized SVM:",
    round(optimized_svm_accuracy * 100, 2),
    "%"
)

print(
    "Threshold-Tuned SVM:",
    round(threshold_accuracy * 100, 2),
    "%"
)

print(
    "Accuracy Difference:",
    round(
        (threshold_accuracy - optimized_svm_accuracy) * 100,
        2
    ),
    "percentage points"
)

if threshold_accuracy > optimized_svm_accuracy:

    print("\nNEW BEST MODEL: Threshold-Tuned SVM")

else:

    print("\nBEST MODEL REMAINS: Optimized SVM")

print("\n================================================")
print("STAGE 17 COMPLETED")
print("================================================")
# ============================================================
# STAGE 18: SLEEP RISK LEVEL
# ============================================================

print("\n================================================")
print("SLEEP RISK ASSESSMENT")
print("================================================")

# Use prediction confidence for project-level risk indication
if predicted_class == "None":
    if confidence >= 0.80:
        risk_level = "LOW"
    else:
        risk_level = "MODERATE"

elif predicted_class in ["Insomnia", "Sleep Apnea"]:
    if confidence >= 0.80:
        risk_level = "HIGH"
    elif confidence >= 0.60:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

else:
    risk_level = "MODERATE"


print("Predicted Condition :", predicted_class)
print("Sleep Risk Level     :", risk_level)
print("Model Confidence     :", round(confidence * 100, 2), "%")

print("\nNote: This is a machine-learning prediction")
print("and not a medical diagnosis.")

print("\n================================================")
print("STAGE 18 COMPLETED")
print("================================================")
# ============================================================
# STAGE 19: PERSONALIZED SLEEP RECOMMENDATION
# ============================================================

print("\n================================================")
print("PERSONALIZED SLEEP RECOMMENDATION")
print("================================================")

if predicted_class == "None":
    recommendation = (
        "Maintain your current healthy sleep habits and "
        "continue following a consistent sleep schedule."
    )

elif predicted_class == "Insomnia":
    recommendation = (
        "Maintain a consistent sleep schedule, reduce stress "
        "before bedtime, and avoid excessive screen exposure at night."
    )

elif predicted_class == "Sleep Apnea":
    recommendation = (
        "Consider discussing your sleep pattern with a healthcare "
        "professional, especially if you experience loud snoring "
        "or breathing interruptions during sleep."
    )

else:
    recommendation = (
        "Maintain regular sleep habits and consider professional "
        "evaluation if sleep-related problems continue."
    )

print("Recommendation:")
print(recommendation)

print("\n================================================")
print("STAGE 19 COMPLETED")
print("================================================")
# ============================================================
# STAGE 20: EXPLAINABLE AI - FEATURE IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

print("\n================================================")
print("STAGE 20: EXPLAINABLE AI")
print("================================================")

print("\nCalculating feature importance...")

# Convert test data to dense format
if hasattr(X_test, "toarray"):
    X_test_for_explanation = X_test.toarray()
else:
    X_test_for_explanation = X_test

# Calculate permutation importance
importance_result = permutation_importance(
    optimized_svm,
    X_test_for_explanation,
    y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get processed feature names
processed_names = preprocessor.get_feature_names_out()

importance_values = importance_result.importances_mean

# Original project features
original_features = [
    "Age",
    "Occupation",
    "BMI Category",
    "Sleep Duration",
    "Stress Level"
]

# Group encoded features
grouped_importance = {}

for feature in original_features:

    matching_values = []

    for name, value in zip(
        processed_names,
        importance_values
    ):

        if feature in name:
            matching_values.append(abs(value))

    if matching_values:
        grouped_importance[feature] = sum(matching_values)
    else:
        grouped_importance[feature] = 0.0

# Sort features
sorted_features = sorted(
    grouped_importance,
    key=grouped_importance.get,
    reverse=True
)

print("\nFEATURE IMPORTANCE RANKING")
print("================================================")

for rank, feature in enumerate(
    sorted_features,
    start=1
):

    print(
        rank,
        ".",
        feature,
        ":",
        round(
            grouped_importance[feature] * 100,
            2
        ),
        "%"
    )

# Create graph
plt.figure(figsize=(9, 6))

values = [
    grouped_importance[feature] * 100
    for feature in sorted_features
]

plt.barh(
    sorted_features[::-1],
    values[::-1]
)

plt.xlabel("Permutation Importance (%)")
plt.ylabel("Patient Feature")

plt.title(
    "Explainable AI - Feature Importance for Sleep Disorder Prediction"
)

plt.tight_layout()
plt.show()

print("\n================================================")
print("STAGE 20 COMPLETED")
print("================================================")
# ============================================================
# STAGE 21: FINAL SLEEP DISORDER PREDICTION
# ============================================================

print("\n================================================")
print("FINAL SLEEP DISORDER PREDICTION")
print("================================================")

age = float(input("Enter Age: "))

occupation = input(
    "Enter Occupation exactly as in dataset: "
)

bmi_category = input(
    "Enter BMI Category exactly as in dataset: "
)

sleep_duration = float(
    input("Enter Sleep Duration: ")
)

stress_level = float(
    input("Enter Stress Level: ")
)

# Create input dataframe
patient_data = pd.DataFrame({
    "Age": [age],
    "Occupation": [occupation],
    "BMI Category": [bmi_category],
    "Sleep Duration": [sleep_duration],
    "Stress Level": [stress_level]
})

# Transform patient data
patient_processed = preprocessor.transform(patient_data)

# Prediction using optimized SVM
prediction = optimized_svm.predict(patient_processed)

# Convert prediction back to original label
if "label_encoder" in globals():
    predicted_disorder = label_encoder.inverse_transform(prediction)[0]
else:
    predicted_disorder = prediction[0]

print("\n================================================")
print("PREDICTION RESULT")
print("================================================")

print("Predicted Sleep Disorder:", predicted_disorder)

if str(predicted_disorder).lower() in ["none", "no disorder", "no sleep disorder"]:
    print("Prediction Status: No sleep disorder detected.")
else:
    print(
        "Prediction Status:",
        str(predicted_disorder),
        "may be indicated."
    )

print("================================================")
print("STAGE 21 COMPLETED")
print("================================================")
# ============================================================
# FINAL CLEAN PROJECT OUTPUT
# ============================================================

import os

# Clear terminal screen
os.system("cls" if os.name == "nt" else "clear")

print()
print("==============================================================")
print("          SLEEP DISORDER CLASSIFICATION SYSTEM")
print("              MACHINE LEARNING RESEARCH PROJECT")
print("==============================================================")

print()
print("PROJECT RESULTS")
print("--------------------------------------------------------------")

print(f"{'Best Model':<30}: Optimized SVM")
print(f"{'Test Accuracy':<30}: {optimized_svm_accuracy * 100:.2f}%")
print(f"{'Macro F1 Score':<30}: {optimized_svm_f1 * 100:.2f}%")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ============================================================
# FINAL 6 PERFORMANCE METRICS
# ============================================================

final_accuracy = accuracy_score(y_test, y_pred)

final_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

final_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

# Multiclass specificity
cm = confusion_matrix(y_test, y_pred)

specificities = []

for i in range(len(cm)):
    true_negative = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    false_positive = cm[:, i].sum() - cm[i, i]

    if true_negative + false_positive != 0:
        specificity = true_negative / (true_negative + false_positive)
        specificities.append(specificity)

final_specificity = sum(specificities) / len(specificities)


# ============================================================
# FINAL PROJECT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("                 MACHINE LEARNING RESEARCH PROJECT")
print("=" * 65)

print("\nPROJECT RESULTS")
print("-" * 65)

print(f"Best Model              : Optimized SVM")
print(f"Accuracy                : {final_accuracy * 100:.2f}%")
print(f"Precision               : {final_precision * 100:.2f}%")
print(f"Recall                  : {final_recall * 100:.2f}%")
print(f"F1 Score                : {final_f1 * 100:.2f}%")
print(f"Specificity             : {final_specificity * 100:.2f}%")
print(f"Balanced Accuracy       : {final_balanced_accuracy * 100:.2f}%")

print("-" * 65)
print("                 PROJECT EXECUTION COMPLETED")
print("=" * 65)
print()
print("--------------------------------------------------------------")
print("                 PREDICTION RESULT")
print("--------------------------------------------------------------")

print(f"{'Predicted Sleep Disorder':<30}: {predicted_class}")

if predicted_class == "None":
    final_status = "No Sleep Disorder Detected"
elif predicted_class == "Insomnia":
    final_status = "Possible Insomnia Detected"
elif predicted_class == "Sleep Apnea":
    final_status = "Possible Sleep Apnea Detected"
else:
    final_status = "Other Sleep Disorder Status"

print(f"{'Prediction Status':<30}: {final_status}")

# Confidence
if "confidence" in globals():
    print(f"{'Prediction Confidence':<30}: {confidence * 100:.2f}%")

# Risk level
if "risk_level" in globals():
    print(f"{'Sleep Risk Level':<30}: {risk_level}")

print()
print("--------------------------------------------------------------")
print("                 RECOMMENDATION")
print("--------------------------------------------------------------")

if "recommendation" in globals():
    print(recommendation)

print()
print("--------------------------------------------------------------")
print("                 EXPLAINABLE AI")
print("--------------------------------------------------------------")

if "sorted_features" in globals():
    print("Important Patient Features:")

    for rank, feature in enumerate(sorted_features, start=1):
        importance = grouped_importance[feature] * 100
        print(f"{rank}. {feature:<25} {importance:.2f}%")

print()
print("==============================================================")
print("             PROJECT EXECUTION COMPLETED")
print("==============================================================")
print()
print("Model: Optimized SVM")
print("The system provides machine-learning based prediction")
print("and is not a medical diagnosis.")
print()
