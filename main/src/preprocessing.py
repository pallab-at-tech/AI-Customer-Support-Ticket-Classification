import re
import pandas as pd


def clean_text(text):
    """
    Clean a customer support ticket description.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove template placeholders such as:
    # {product_purchased}
    # {error_message}
    text = re.sub(r"\{[^}]*\}", " ", text)

    # Normalize common contractions
    contractions = {
        "I'm": "Im",
        "I've": "Ive",
        "I'll": "Ill",
        "I'd": "Id",
        "it's": "its",
        "don't": "dont",
        "doesn't": "doesnt",
        "didn't": "didnt",
        "can't": "cant",
        "couldn't": "couldnt",
        "wouldn't": "wouldnt",
        "shouldn't": "shouldnt",
        "isn't": "isnt",
        "aren't": "arent",
        "wasn't": "wasnt",
        "weren't": "werent",
        "you're": "youre",
        "they're": "theyre",
        "we're": "were"
    }

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove newlines, tabs etc.
    text = re.sub(
        r"[\n\r\t]+",
        " ",
        text
    )

    # Keep letters, numbers and spaces
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.lower().strip()


def prepare_dataframe(df):
    """
    Add cleaned ticket description column.
    """

    df = df.copy()

    if "Ticket Description" not in df.columns:
        raise ValueError(
            "Dataset must contain 'Ticket Description' column."
        )

    df["Cleaned Description"] = (
        df["Ticket Description"]
        .apply(clean_text)
    )

    return df