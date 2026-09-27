import os
import sys
import joblib


# --------------------------------------------------
# Import preprocessing
# --------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from preprocessing import clean_text


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VECTOR_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "ticket_classifier.pkl"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

tfidf = joblib.load(
    VECTOR_PATH
)

model = joblib.load(
    MODEL_PATH
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_ticket(description):

    cleaned_text = clean_text(
        description
    )

    vector = tfidf.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    ranked_predictions = sorted(
        zip(
            model.classes_,
            probabilities
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return (
        prediction,
        ranked_predictions[:5]
    )


# --------------------------------------------------
# Terminal interface
# --------------------------------------------------

if __name__ == "__main__":

    description = input(
        "\nEnter ticket description: "
    )

    prediction, top_predictions = (
        predict_ticket(description)
    )

    print(
        "\nPredicted Category:",
        prediction
    )

    print("\nTop 5 predictions:")

    for category, probability in top_predictions:

        print(
            f"{category}: "
            f"{probability:.2%}"
        )