import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("1. Loading dataset...")

# Dynamically get the absolute path to the directory this script is in
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "spam.csv")

# Use the absolute path to load the CSV
df = pd.read_csv(csv_path, encoding="latin-1")

# DEBUG: This will print the actual column names in your CSV to the terminal
print(f"Actual columns in CSV: {df.columns.tolist()}")

# UPDATE THESE NAMES: Replace "Column1" and "Column2" with the actual names printed above
# "Column1" should be the label (spam/ham) and "Column2" should be the message text.
df = df[["Category", "Message"]]
df.columns = ["label", "message"]

# Encode labels: ham -> 0, spam -> 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Dataset loaded successfully!")

# --- The rest of your training code goes here ---
# X_train, X_test, y_train, y_test = train_test_split(...)
# model = LogisticRegression()
# model.fit(...)
