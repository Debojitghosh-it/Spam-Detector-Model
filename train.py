import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


print("1. Loading dataset...")

# Get the directory where train.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset path
csv_path = os.path.join(current_dir, "spam.csv")

# Load dataset
df = pd.read_csv(csv_path, encoding="latin-1")

print("Actual columns:", df.columns.tolist())

# Handle the common spam.csv formats
if "Category" in df.columns and "Message" in df.columns:
    df = df[["Category", "Message"]]
    df.columns = ["label", "message"]

elif "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

else:
    raise ValueError(
        "Could not find the expected columns. "
        "Expected Category/Message or v1/v2."
    )

# Remove missing values
df = df.dropna()

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove rows where label conversion failed
df = df.dropna()

df["label"] = df["label"].astype(int)

print("Dataset loaded successfully!")
print("Total messages:", len(df))

# Separate features and labels
X = df["message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("2. Creating TF-IDF vectorizer...")

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("3. Training Logistic Regression model...")

# Create model
model = LogisticRegression(
    max_iter=1000
)

# Train
model.fit(X_train_vectorized, y_train)

print("4. Evaluating model...")

# Predictions
y_pred = model.predict(X_test_vectorized)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model_path = os.path.join(
    current_dir,
    "logistic_spam_model.pkl"
)

vectorizer_path = os.path.join(
    current_dir,
    "tfidf_vectorizer.pkl"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("5. Model saved successfully!")
print("Model:", model_path)
print("Vectorizer:", vectorizer_path)