import os
import sys

import joblib
import streamlit as st


# --------------------------------------------------
# Import preprocessing
# --------------------------------------------------

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from preprocessing import clean_text


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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
# Header
# --------------------------------------------------

st.title(
    "🎫 AI Customer Support Ticket Classifier"
)

st.write(
    "Enter a customer support ticket description "
    "to predict its ticket category."
)


# --------------------------------------------------
# Check model
# --------------------------------------------------

if not os.path.exists(VECTOR_PATH):

    st.error(
        "TF-IDF vectorizer not found. "
        "Run `python src/train.py` first."
    )

    st.stop()


if not os.path.exists(MODEL_PATH):

    st.error(
        "Trained model not found. "
        "Run `python src/train.py` first."
    )

    st.stop()


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
# Input
# --------------------------------------------------

description = st.text_area(
    "Ticket Description",
    placeholder=(
        "Example: My laptop battery is draining "
        "very quickly and does not last long."
    ),
    height=150
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Category",
    type="primary"
):

    if not description.strip():

        st.warning(
            "Please enter a ticket description."
        )

    else:

        cleaned = clean_text(
            description
        )

        vector = tfidf.transform(
            [cleaned]
        )

        prediction = model.predict(
            vector
        )[0]

        probabilities = model.predict_proba(
            vector
        )[0]

        ranked = sorted(
            zip(
                model.classes_,
                probabilities
            ),
            key=lambda x: x[1],
            reverse=True
        )

        # Main result
        st.success(
            f"Predicted Category: {prediction}"
        )

        # Top predictions
        st.subheader(
            "Top 5 Model Predictions"
        )

        for category, probability in ranked[:5]:

            st.write(
                f"**{category}** — "
                f"{probability:.2%}"
            )

        st.info(
            "The supplied dataset contains substantial "
            "text/label inconsistency. Therefore, these "
            "predictions should be treated as baseline "
            "model outputs rather than production-level "
            "classifications."
        )