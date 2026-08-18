# CREDIT RISK ANALYSIS

# Models:
#   1. Logistic Regression
#   2. Support Vector Machine with RBF kernel

# Datasets:
#   1. UCI Default of Credit Card Clients
#   2. UCI South German Credit

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import sklearn

from ucimlrepo import fetch_ucirepo

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

TEST_SIZE = 0.25

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

print("\nLoading Taiwan Default of Credit Card Clients...")

taiwan_data = fetch_ucirepo(id=350)

X_taiwan = taiwan_data.data.features.copy()
y_taiwan = taiwan_data.data.targets.copy().squeeze()


taiwan_rename = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}

X_taiwan = X_taiwan.rename(columns=taiwan_rename)


y_taiwan = pd.to_numeric(y_taiwan).astype(int)

print("Taiwan dataset loaded successfully.")



print("\nLoading South German Credit...")

GERMAN_FILE = Path("SouthGermanCredit.asc")

if not GERMAN_FILE.exists():
    raise FileNotFoundError(
        "\nSouthGermanCredit.asc was not found.\n"
        "Place SouthGermanCredit.asc in the same folder as analysis.py."
    )

german_raw = pd.read_csv(
    GERMAN_FILE,
    sep=r"\s+",
    engine="python",
)

print("\nOriginal South German columns:")
print(german_raw.columns.tolist())


german_rename = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

german_raw = german_raw.rename(columns=german_rename)

if "credit_risk" not in german_raw.columns:
    raise ValueError(
        "Could not find credit_risk target.\n"
        f"Columns found: {german_raw.columns.tolist()}"
    )

if german_raw.shape[0] != 1000:
    raise ValueError(
        f"Expected 1000 South German records, found {german_raw.shape[0]}."
    )

X_german = german_raw.drop(columns=["credit_risk"]).copy()
y_german_raw = german_raw["credit_risk"].copy()

print("South German dataset loaded successfully.")


print("\nOriginal South German target distribution:")
print(y_german_raw.value_counts(dropna=False))

german_counts = y_german_raw.value_counts()

if len(german_counts) != 2:
    raise ValueError(
        "Expected exactly two classes in South German Credit."
    )

# Official dataset contains 700 good and 300 bad cases.
if sorted(german_counts.tolist()) != [300, 700]:
    raise ValueError(
        "Expected 700 good and 300 bad credit cases.\n"
        f"Found:\n{german_counts}"
    )

bad_credit_label = german_counts.idxmin()

y_german = (y_german_raw == bad_credit_label).astype(int)

print("\nOriginal label corresponding to BAD credit:")
print(bad_credit_label)

print("\nRe-encoded South German target:")
print(y_german.value_counts().sort_index())

print("0 = good / lower risk")
print("1 = bad / higher risk")



def describe_dataset(name, X, y):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {X.shape[0]}")
    print(f"Predictors: {X.shape[1]}")
    print(f"Missing predictor values: {X.isna().sum().sum()}")
    print(f"Missing target values: {y.isna().sum()}")

    print("\nTarget counts:")
    print(y.value_counts().sort_index())

    print("\nTarget proportions:")
    print(
        y.value_counts(normalize=True)
        .sort_index()
        .round(4)
    )


describe_dataset(
    "TAIWAN DEFAULT OF CREDIT CARD CLIENTS",
    X_taiwan,
    y_taiwan,
)

describe_dataset(
    "SOUTH GERMAN CREDIT",
    X_german,
    y_german,
)



taiwan_categorical = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

taiwan_numeric = [
    column
    for column in X_taiwan.columns
    if column not in taiwan_categorical
]


german_categorical = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

missing_columns = [
    column
    for column in german_categorical
    if column not in X_german.columns
]

if missing_columns:
    raise ValueError(
        "Expected South German columns were not found:\n"
        f"{missing_columns}"
    )

german_numeric = [
    column
    for column in X_german.columns
    if column not in german_categorical
]


print("\nTaiwan categorical variables:")
print(taiwan_categorical)

print("\nTaiwan numeric/ordinal variables:")
print(taiwan_numeric)

print("\nSouth German categorical variables:")
print(german_categorical)

print("\nSouth German numeric/ordinal variables:")
print(german_numeric)

def make_preprocessor(categorical_columns, numeric_columns):

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ],
        remainder="drop",
    )


taiwan_preprocessor = make_preprocessor(
    taiwan_categorical,
    taiwan_numeric,
)

german_preprocessor = make_preprocessor(
    german_categorical,
    german_numeric,
)


def make_models(preprocessor):

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessing",
                clone(preprocessor),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    svm_model = Pipeline(
        steps=[
            (
                "preprocessing",
                clone(preprocessor),
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=1000,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_model,
        "RBF SVM": svm_model,
    }



all_results = []
german_cost_results = []

saved_models = {}
saved_splits = {}


def evaluate_dataset(
    dataset_name,
    X,
    y,
    preprocessor,
    calculate_german_cost=False,
):

    print("\n" + "#" * 70)
    print(f"ANALYSING: {dataset_name}")
    print("#" * 70)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    print("\nTraining target proportions:")
    print(
        y_train.value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    print("\nTesting target proportions:")
    print(
        y_test.value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    models = make_models(preprocessor)

    dataset_scores = {}

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Training {dataset_name} - {model_name}")
        print("-" * 70)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_score = model.decision_function(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            y_score,
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        result = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "ROC_AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }

        all_results.append(result)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nConfusion matrix:")
        print(cm)

        print(f"TN = {tn}")
        print(f"FP = {fp}")
        print(f"FN = {fn}")
        print(f"TP = {tp}")


        if calculate_german_cost:


            total_cost = (5 * fn) + fp

            german_cost_results.append(
                {
                    "Model": model_name,
                    "False_Positives": fp,
                    "False_Negatives": fn,
                    "FP_Cost": 1,
                    "FN_Cost": 5,
                    "Total_Cost": total_cost,
                }
            )

            print(
                "\nSouth German cost "
                f"(5 × FN + 1 × FP): {total_cost}"
            )



        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Lower risk",
                "Higher risk",
            ],
        ).plot(
            ax=ax,
            values_format="d",
        )

        ax.set_title(
            f"{dataset_name}\n{model_name}"
        )

        fig.tight_layout()

        filename = (
            "confusion_"
            + dataset_name.lower().replace(" ", "_")
            + "_"
            + model_name.lower().replace(" ", "_")
            + ".png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        dataset_scores[model_name] = y_score

        saved_models[
            (dataset_name, model_name)
        ] = model



    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    for model_name, y_score in dataset_scores.items():

        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name=model_name,
            ax=ax,
        )

    ax.set_title(
        f"ROC Curves - {dataset_name}"
    )

    fig.tight_layout()

    roc_filename = (
        "roc_"
        + dataset_name.lower().replace(" ", "_")
        + ".png"
    )

    fig.savefig(
        OUTPUT_DIR / roc_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    saved_splits[dataset_name] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }



evaluate_dataset(
    dataset_name="Taiwan",
    X=X_taiwan,
    y=y_taiwan,
    preprocessor=taiwan_preprocessor,
    calculate_german_cost=False,
)

evaluate_dataset(
    dataset_name="South German",
    X=X_german,
    y=y_german,
    preprocessor=german_preprocessor,
    calculate_german_cost=True,
)



def calculate_feature_importance(dataset_name):

    print("\n" + "=" * 70)
    print(f"FEATURE IMPORTANCE - {dataset_name}")
    print("=" * 70)

    model = saved_models[
        (dataset_name, "Logistic Regression")
    ]

    X_test = saved_splits[
        dataset_name
    ]["X_test"]

    y_test = saved_splits[
        dataset_name
    ]["y_test"]

    importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance_Mean":
                importance_result.importances_mean,
            "Importance_SD":
                importance_result.importances_std,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance_Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nTop features:")

    print(
        importance_df
        .head(15)
        .round(4)
        .to_string(index=False)
    )


    csv_name = (
        "feature_importance_"
        + dataset_name.lower().replace(" ", "_")
        + ".csv"
    )

    importance_df.to_csv(
        OUTPUT_DIR / csv_name,
        index=False,
    )



    top_features = (
        importance_df
        .head(12)
        .copy()
        .sort_values(
            "Importance_Mean",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.barh(
        top_features["Feature"],
        top_features["Importance_Mean"],
    )

    ax.set_xlabel(
        "Decrease in ROC-AUC when shuffled"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"Permutation Feature Importance - {dataset_name}"
    )

    fig.tight_layout()

    graph_name = (
        "feature_importance_"
        + dataset_name.lower().replace(" ", "_")
        + ".png"
    )

    fig.savefig(
        OUTPUT_DIR / graph_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


calculate_feature_importance(
    "Taiwan"
)

calculate_feature_importance(
    "South German"
)



results_df = pd.DataFrame(
    all_results
)

results_df = results_df[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
]

results_df.to_csv(
    OUTPUT_DIR / "model_results.csv",
    index=False,
)



cost_df = pd.DataFrame(
    german_cost_results
)

cost_df.to_csv(
    OUTPUT_DIR / "south_german_cost_results.csv",
    index=False,
)



with open(
    OUTPUT_DIR / "run_information.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Credit Risk Analysis\n"
    )

    file.write(
        "====================\n\n"
    )

    file.write(
        f"Random state: {RANDOM_STATE}\n"
    )

    file.write(
        f"Training proportion: {1 - TEST_SIZE}\n"
    )

    file.write(
        f"Testing proportion: {TEST_SIZE}\n"
    )

    file.write(
        f"scikit-learn version: {sklearn.__version__}\n"
    )

    file.write(
        "Positive class: "
        "1 = default / bad credit / higher risk\n"
    )

    file.write(
        "Train-test split: stratified\n"
    )

    file.write(
        "Categorical variables: one-hot encoded\n"
    )

    file.write(
        "Numeric/ordinal variables: standardised\n"
    )

    file.write(
        "Preprocessing fitted on training data only\n"
    )

    file.write(
        "Logistic Regression class_weight: balanced\n"
    )

    file.write(
        "SVM kernel: RBF\n"
    )

    file.write(
        "SVM C: 1.0\n"
    )

    file.write(
        "SVM gamma: scale\n"
    )

    file.write(
        "SVM class_weight: balanced\n"
    )

    file.write(
        "South German cost: 5 × FN + 1 × FP\n"
    )

    file.write(
        "Feature importance: permutation importance "
        "using Logistic Regression and ROC-AUC\n"
    )



print("\n\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(
    results_df
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("SOUTH GERMAN COST-SENSITIVE RESULTS")
print("=" * 70)

print(
    cost_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nAll files were saved in:")
print(OUTPUT_DIR.resolve())