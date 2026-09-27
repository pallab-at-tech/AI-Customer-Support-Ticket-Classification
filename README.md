# AI Customer Support Ticket Classification

## Project Overview

This project analyzes customer support tickets and develops a machine-learning baseline for predicting the ticket category from the ticket description.

The project covers:

* Exploratory Data Analysis
* Missing-value analysis
* Duplicate detection
* Text preprocessing
* Keyword analysis
* Label consistency analysis
* TF-IDF feature extraction
* Logistic Regression
* Model evaluation
* Confusion matrix
* Terminal-based prediction
* Real-time prediction using Streamlit

---

## Dataset

The dataset was downloaded from **Kaggle** and contains **8,469 customer support tickets** and **17 columns**.

### Dataset Source

* **Source:** [Kaggle – Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
* **Records:** 8,469
* **Columns:** 17
* **Ticket Categories:** 16

### Important Fields

* Customer Name
* Customer Email
* Customer Age
* Customer Gender
* Product Purchased
* Date of Purchase
* Ticket Type
* Ticket Category
* Ticket Description
* Ticket Status
* Resolution
* Ticket Priority
* Ticket Channel
* First Response Time
* Time to Resolution
* Customer Satisfaction Rating

The primary prediction target is:

`Ticket Category`

---

## Data Preprocessing

The ticket description is cleaned by:

1. Removing template placeholders
2. Normalizing contractions
3. Removing URLs
4. Removing special characters
5. Removing extra whitespace
6. Converting text to lowercase

---

## Exploratory Data Analysis

The dataset was analyzed for:

* Missing values
* Duplicate records
* Category distribution
* Ticket type distribution
* Repeated descriptions
* Description/category consistency
* Keyword/category relationships

---

## Data Quality Findings

The analysis identified several inconsistencies.

There are **461 duplicate cleaned descriptions**.

Additionally, **64 cleaned descriptions are associated with more than one ticket category**.

Keyword analysis also shows weak semantic relationships between descriptions and their assigned categories.

For example, the keyword `battery` appears across many unrelated categories instead of being concentrated in `Battery life`.

The keyword `refund` also appears across multiple unrelated categories.

These findings indicate that the dataset contains substantial text/label noise.

---

## Machine Learning Pipeline

The baseline pipeline is:

```text
Ticket Description
        ↓
Text Cleaning
        ↓
TF-IDF
        ↓
Logistic Regression
        ↓
Ticket Category
```

### TF-IDF

The model uses:

* Maximum features: 10000
* Stop words: English
* N-grams: unigrams and bigrams

### Logistic Regression

Configuration:

* `c = 1000`
* `max_iter = 2000`
* `random_state = 42`

---

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The model is also compared against:

* Random baseline
* Majority-class baseline

For 16 categories, the random baseline is approximately **6.25%**.

The previously obtained baseline model accuracy was approximately **6.79%**, which is very close to both the random and majority-class baselines.

---

## Interpretation

The low accuracy should not be interpreted only as a model-selection problem.

The exploratory analysis indicates that the supplied dataset itself has weak relationships between ticket descriptions and their assigned categories.

Changing the classification algorithm alone is therefore unlikely to resolve the underlying data-quality problem.

A production-quality classifier would require a dataset in which ticket descriptions and target labels are semantically consistent.

---

# Project Structure

```text
AI_Ticket_Classification/
│
├── data/
│   └── tickets.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── model/
│   ├── tfidf_vectorizer.pkl
│   ├── ticket_classifier.pkl
│   └── test_predictions.csv
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation and Setup

## 1. Clone or download the project

Open the project folder in your terminal or VS Code.

Make sure the dataset is located at:

```text
data/tickets.csv
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

Run:

```bash
pip install -r requirements.txt
```

The main libraries used in this project are:

* pandas
* NumPy
* scikit-learn
* joblib
* matplotlib
* Streamlit
* Jupyter

---

# Running the Project

The recommended order is:

```text
1. Train the model
       ↓
2. Evaluate the model
       ↓
3. Test terminal prediction
       ↓
4. Run Streamlit application
```

---

## 1. Train the Model

From the project root directory, run:

```bash
python src/train.py
```

The training script will:

1. Load `data/tickets.csv`
2. Clean the ticket descriptions
3. Split the data into training and testing sets
4. Create TF-IDF features
5. Train the Logistic Regression model
6. Generate predictions
7. Display training and testing performance
8. Save the trained model

After successful training, the following files will be created:

```text
model/
├── tfidf_vectorizer.pkl
├── ticket_classifier.pkl
└── test_predictions.csv
```

The testing result will be displayed similar to:

```text
============================================================
TESTING PERFORMANCE
============================================================

Accuracy: 0.0667
Accuracy percentage: 6.67%
```

The exact value may vary slightly depending on the dataset and environment.

---

## 2. Evaluate the Model

After training, run:

```bash
python src/evaluate.py
```

This script evaluates the trained model on the test data and compares its performance with baseline performance.

It displays:

* Number of classes
* Model accuracy
* Random baseline
* Majority-class baseline
* Majority class
* Classification report

Example:

```text
============================================================
MODEL EVALUATION
============================================================

Number of classes: 16

Model accuracy: 0.0667

Random baseline: 0.0625

Majority baseline: 0.0680

Majority class: Refund request
```

The evaluation script also generates the confusion matrix:

```text
screenshots/
└── confusion_matrix.png
```

The confusion matrix shows the relationship between the actual ticket categories and the categories predicted by the model.

---

## 3. Terminal Prediction

Once the model has been trained, you can classify a new support ticket directly from the terminal.

Run:

```bash
python src/predict.py
```

The program will ask:

```text
Enter ticket description:
```

For example:

```text
My laptop battery is draining very quickly.
```

The program will return the predicted ticket category:

```text
Predicted Category: Battery life
```

It will also display the top five predicted categories and their probabilities:

```text
Top 5 predictions:

Battery life: 15.20%
Hardware issue: 12.31%
Software bug: 10.84%
Product setup: 9.72%
Display issue: 8.91%
```

> **Note:** The actual prediction and probability values depend on the trained model. Because the supplied dataset contains substantial label noise, predictions should be treated as baseline model outputs rather than production-level classifications.

---

## 4. Run the Streamlit Application / Web Interface

To launch the web interface, run:

```bash
streamlit run app.py
```

Streamlit will start a local web server.

The terminal will normally show an address similar to:

```text
Local URL: http://localhost:8501
```

Open that address in your browser.

The application provides:

* Ticket description input
* Predict Category button
* Predicted ticket category
* Top five model predictions
* Prediction probabilities

The application uses the same trained:

```text
TF-IDF Vectorizer
+
Logistic Regression Model
```

that was generated during training.

### Sample Input

Enter a customer support ticket description in the text input field:

```text
My laptop battery is draining very quickly and does not last long.
```

Then click the **Predict Category** button.

### Sample Output

The application displays the predicted category:

```text
Predicted Category: Battery life
```

It also displays the top five predictions with their probabilities:

```text
Top 5 Predictions

Battery life          <probability>%
Hardware issue        <probability>%
Software bug          <probability>%
Product setup         <probability>%
Display issue         <probability>%
```

The actual prediction and probability values depend on the trained model and the input ticket description.

> **Note:** Because the supplied dataset contains substantial label noise and inconsistencies between ticket descriptions and categories, the web application's predictions should be considered baseline model outputs rather than production-level classifications.


---

# Running the Jupyter Notebook

The notebook contains the exploratory data analysis and data-quality investigation.

Start Jupyter with:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open:

```text
notebooks/analysis.ipynb
```

The notebook covers:

* Dataset shape
* Dataset information
* Missing values
* Duplicate records
* Ticket category distribution
* Ticket type distribution
* Text preprocessing
* Duplicate descriptions
* Description-to-category consistency
* Keyword analysis
* Ticket Type × Ticket Category analysis
* Baseline calculations

---

# Complete Execution Flow

For a fresh setup, execute the following commands from the project root:

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train

```bash
python src/train.py
```

### Step 3 — Evaluate

```bash
python src/evaluate.py
```

### Step 4 — Test prediction

```bash
python src/predict.py
```

### Step 5 — Launch web application

```bash
streamlit run app.py
```

---

# Important Note About the Dataset

The analysis found substantial inconsistencies between ticket descriptions and their assigned categories.

For example, common keywords such as `battery` and `refund` are distributed across multiple unrelated categories.

There are also repeated descriptions that appear with different category labels.

Therefore, the low classification accuracy reflects not only the limitations of the baseline model but also the quality and consistency of the supplied training data.

For a production-quality classification system, the dataset should be reviewed and relabeled so that ticket descriptions have a consistent relationship with their target categories.

---

# Conclusion

This project demonstrates a complete machine-learning classification workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Quality Analysis
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit Application
```

The project also demonstrates an important machine-learning principle: **model performance depends heavily on the quality and consistency of the training data.**

The supplied dataset shows substantial inconsistencies between ticket descriptions and category labels. Therefore, improving dataset quality and labeling consistency would be an important step before deploying a reliable production classification system.
