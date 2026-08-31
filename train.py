import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. CONVERT TOTALCHARGES TO NUMERIC
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges values:")
print(df["TotalCharges"].isnull().sum())

print("\nFirst 5 TotalCharges values:")
print(df["TotalCharges"].head())


# ============================================================
# 4. HANDLE MISSING TOTALCHARGES VALUES
# ============================================================

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing TotalCharges values after cleaning:")
print(df["TotalCharges"].isnull().sum())


# ============================================================
# 5. CHECK DUPLICATE ROWS
# ============================================================

duplicates = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicates)


# ============================================================
# 6. CHECK UNIQUE CATEGORIES
# ============================================================

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())


# ============================================================
# 7. EDA - CHURN BY CONTRACT
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by contract type:")
print(contract_churn)

contract_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 8. EDA - TENURE VS CHURN
# ============================================================

average_tenure = df.groupby("Churn")["tenure"].mean()

print("\nAverage tenure by churn:")
print(average_tenure)

df.boxplot(
    column="tenure",
    by="Churn",
    figsize=(10, 6)
)

plt.title("Customer Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()


# ============================================================
# 9. EDA - MONTHLY CHARGES VS CHURN
# ============================================================

average_monthly_charges = df.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage monthly charges by churn:")
print(average_monthly_charges)

df.boxplot(
    column="MonthlyCharges",
    by="Churn",
    figsize=(10, 6)
)

plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()


# ============================================================
# 10. EDA - INTERNET SERVICE
# ============================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by internet service:")
print(internet_churn)

internet_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 11. EDA - PAYMENT METHOD
# ============================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by payment method:")
print(payment_churn)

payment_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=20)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 12. EDA - GENDER
# ============================================================

gender_churn = pd.crosstab(
    df["gender"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by gender:")
print(gender_churn)

gender_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Gender")
plt.xlabel("Gender")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 13. EDA - SENIOR CITIZEN
# ============================================================

df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({
    0: "Non-Senior Citizen",
    1: "Senior Citizen"
})

senior_churn = pd.crosstab(
    df["SeniorCitizenLabel"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by senior citizen status:")
print(senior_churn)

senior_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Senior Citizen Status")
plt.xlabel("Customer Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 14. EDA - PARTNER
# ============================================================

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by partner status:")
print(partner_churn)

partner_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Partner Status")
plt.xlabel("Has Partner")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 15. EDA - DEPENDENTS
# ============================================================

dependents_churn = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by dependents status:")
print(dependents_churn)

dependents_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Dependents Status")
plt.xlabel("Has Dependents")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 16. EDA - ONLINE SECURITY
# ============================================================

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Online Security:")
print(security_churn)

security_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 17. EDA - ONLINE BACKUP
# ============================================================

backup_churn = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Online Backup:")
print(backup_churn)

backup_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Online Backup")
plt.xlabel("Online Backup")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 18. EDA - DEVICE PROTECTION
# ============================================================

device_churn = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Device Protection:")
print(device_churn)

device_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Device Protection")
plt.xlabel("Device Protection")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 19. EDA - TECH SUPPORT
# ============================================================

tech_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Tech Support:")
print(tech_churn)

tech_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 20. EDA - STREAMING TV
# ============================================================

streaming_tv_churn = pd.crosstab(
    df["StreamingTV"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Streaming TV:")
print(streaming_tv_churn)

streaming_tv_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Streaming TV")
plt.xlabel("Streaming TV")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 21. EDA - STREAMING MOVIES
# ============================================================

streaming_movies_churn = pd.crosstab(
    df["StreamingMovies"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Streaming Movies:")
print(streaming_movies_churn)

streaming_movies_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Streaming Movies")
plt.xlabel("Streaming Movies")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 22. EDA - PHONE SERVICE
# ============================================================

phone_churn = pd.crosstab(
    df["PhoneService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Phone Service:")
print(phone_churn)

phone_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Phone Service")
plt.xlabel("Phone Service")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 23. EDA - MULTIPLE LINES
# ============================================================

multiple_lines_churn = pd.crosstab(
    df["MultipleLines"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by multiple lines:")
print(multiple_lines_churn)

multiple_lines_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Churn Percentage by Multiple Lines")
plt.xlabel("Multiple Lines")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 24. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    ["Churn", "SeniorCitizenLabel", "customerID"],
    axis=1
)

y = df["Churn"]


print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# ============================================================
# 25. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 26. CHECK TRAIN / TEST DATA
# ============================================================

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# ============================================================
# 27. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["str"]
).columns

numerical_features = X_train.select_dtypes(
    exclude=["str"]
).columns


# ============================================================
# 28. CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 29. TRANSFORM TRAINING AND TESTING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)


# ============================================================
# 30. TRAIN MACHINE LEARNING MODELS
# ============================================================

# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

print("\nLogistic Regression training completed.")


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_model.fit(
    X_train_processed,
    y_train
)

print("Decision Tree training completed.")


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

print("Random Forest training completed.")

# ============================================================
# 31. MAKE PREDICTIONS AND EVALUATE MODELS
# ============================================================


# ------------------------------------------------------------
# Logistic Regression Predictions
# ------------------------------------------------------------

logistic_predictions = logistic_model.predict(
    X_test_processed
)

logistic_probabilities = logistic_model.predict_proba(
    X_test_processed
)[:, 1]


# ------------------------------------------------------------
# Decision Tree Predictions
# ------------------------------------------------------------

decision_tree_predictions = decision_tree_model.predict(
    X_test_processed
)

decision_tree_probabilities = decision_tree_model.predict_proba(
    X_test_processed
)[:, 1]


# ------------------------------------------------------------
# Random Forest Predictions
# ------------------------------------------------------------

random_forest_predictions = random_forest_model.predict(
    X_test_processed
)

random_forest_probabilities = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 32. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, predictions, probabilities):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        pos_label="Yes"
    )

    recall = recall_score(
        y_true,
        predictions,
        pos_label="Yes"
    )

    f1 = f1_score(
        y_true,
        predictions,
        pos_label="Yes"
    )

    roc_auc = roc_auc_score(
        (y_true == "Yes").astype(int),
        probabilities
    )

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=["No", "Yes"]
        )
    )


# ============================================================
# 33. EVALUATE ALL THREE MODELS
# ============================================================

evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)

evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_predictions,
    decision_tree_probabilities
)

evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions,
    random_forest_probabilities
)

# ============================================================
# 34. HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# ============================================================

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(
    X_train_processed,
    y_train
)

print("\nBest Logistic Regression parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(grid_search.best_score_)

# ============================================================
# 35. EVALUATE TUNED LOGISTIC REGRESSION
# ============================================================

best_logistic_model = grid_search.best_estimator_


tuned_predictions = best_logistic_model.predict(
    X_test_processed
)


tuned_probabilities = best_logistic_model.predict_proba(
    X_test_processed
)[:, 1]


evaluate_model(
    "Tuned Logistic Regression",
    y_test,
    tuned_predictions,
    tuned_probabilities
)

# ============================================================
# 36. CHURN PROBABILITY THRESHOLD ANALYSIS
# ============================================================

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


print("\nChurn Threshold Analysis:")

for threshold in thresholds:

    threshold_predictions = np.where(
        logistic_probabilities >= threshold,
        "Yes",
        "No"
    )

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        pos_label="Yes"
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        pos_label="Yes"
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_predictions,
        pos_label="Yes"
    )

    print(
        f"\nThreshold: {threshold}"
    )

    print(
        f"Precision: {threshold_precision:.4f}"
    )

    print(
        f"Recall: {threshold_recall:.4f}"
    )

    print(
        f"F1 Score: {threshold_f1:.4f}"
    )

    # ============================================================
# 37. FINAL THRESHOLD ANALYSIS - 0.30
# ============================================================

final_threshold = 0.30


final_predictions = np.where(
    logistic_probabilities >= final_threshold,
    "Yes",
    "No"
)


# ------------------------------------------------------------
# Calculate final metrics
# ------------------------------------------------------------

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    pos_label="Yes"
)

final_recall = recall_score(
    y_test,
    final_predictions,
    pos_label="Yes"
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    pos_label="Yes"
)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

final_cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=["No", "Yes"]
)


print("\n" + "=" * 50)
print("FINAL THRESHOLD ANALYSIS")
print("=" * 50)

print(f"\nSelected threshold: {final_threshold}")

print(f"\nAccuracy : {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall   : {final_recall:.4f}")
print(f"F1 Score : {final_f1:.4f}")


print("\nConfusion Matrix:")
print(final_cm)


# ------------------------------------------------------------
# Extract confusion matrix values
# ------------------------------------------------------------

true_negative = final_cm[0][0]
false_positive = final_cm[0][1]
false_negative = final_cm[1][0]
true_positive = final_cm[1][1]


print("\nConfusion Matrix Breakdown:")

print(f"True Negatives : {true_negative}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")
print(f"True Positives : {true_positive}")


# ------------------------------------------------------------
# Number of customers predicted to churn
# ------------------------------------------------------------

predicted_churners = np.sum(
    final_predictions == "Yes"
)

actual_churners = np.sum(
    y_test == "Yes"
)


print(f"\nCustomers predicted to churn: {predicted_churners}")
print(f"Actual churners: {actual_churners}")

# ============================================================
# 38. CREATE FINAL MODEL PIPELINE
# ============================================================

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", logistic_model)
    ]
)


# ============================================================
# 39. TRAIN FINAL MODEL PIPELINE
# ============================================================

final_model.fit(
    X_train,
    y_train
)


# ============================================================
# 40. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "models/churn_model.pkl"
)


# ============================================================
# 41. SAVE CHURN THRESHOLD
# ============================================================

joblib.dump(
    final_threshold,
    "models/churn_threshold.pkl"
)


print("\nFinal model saved successfully.")
print("Model file: models/churn_model.pkl")
print("Threshold file: models/churn_threshold.pkl")

# ============================================================
# 42. VERIFY SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    "models/churn_model.pkl"
)

loaded_threshold = joblib.load(
    "models/churn_threshold.pkl"
)

print("\nSaved model loaded successfully.")
print("Loaded threshold:", loaded_threshold)

# ============================================================
# 43. TEST LOADED MODEL
# ============================================================

loaded_probabilities = loaded_model.predict_proba(
    X_test
)[:, 1]

loaded_predictions = np.where(
    loaded_probabilities >= loaded_threshold,
    "Yes",
    "No"
)

print("\nLoaded model prediction test:")
print("Number of predictions:", len(loaded_predictions))
print("First 10 predictions:")
print(loaded_predictions[:10])

# ============================================================
# 44. VERIFY SAVED MODEL PREDICTIONS
# ============================================================

original_probabilities = logistic_model.predict_proba(
    X_test_processed
)[:, 1]

saved_model_probabilities = loaded_model.predict_proba(
    X_test
)[:, 1]


predictions_match = np.allclose(
    original_probabilities,
    saved_model_probabilities
)


print("\nPrediction verification:")
print("Original and saved model probabilities match:", predictions_match)