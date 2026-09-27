import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

SCREENSHOT_DIR = os.path.join(
    BASE_DIR,
    "screenshots"
)

os.makedirs(
    SCREENSHOT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

prediction_path = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)

predictions = pd.read_csv(
    prediction_path
)


y_test = predictions["Actual"]

y_pred = predictions["Predicted"]


# --------------------------------------------------
# Model accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# Random baseline
# --------------------------------------------------

number_of_classes = y_test.nunique()

random_baseline = (
    1 / number_of_classes
)


# --------------------------------------------------
# Majority baseline
# --------------------------------------------------

majority_class = (
    y_test
    .value_counts()
    .idxmax()
)

majority_predictions = [
    majority_class
] * len(y_test)

majority_baseline = accuracy_score(
    y_test,
    majority_predictions
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"Number of classes: {number_of_classes}"
)

print(
    f"Model accuracy: {accuracy:.4f}"
)

print(
    f"Random baseline: {random_baseline:.4f}"
)

print(
    f"Majority baseline: {majority_baseline:.4f}"
)

print(
    f"Majority class: {majority_class}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(14, 11)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)

ax.set_title(
    "Ticket Category Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    SCREENSHOT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved:",
    confusion_path
)