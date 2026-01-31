# =========================
# Email Spam Classifier
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# -------------------------
# Load Dataset
# -------------------------
# File is inside data/ folder
data = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)
# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("Dataset Preview:")
print(data.head())
# -------------------------
# Convert Labels to Binary
# -------------------------
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)
# -------------------------
# TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# -------------------------
# Naive Bayes Model
# -------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))
# -------------------------
# Logistic Regression Model
# -------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

lr_pred = lr_model.predict(X_test_tfidf)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
# -------------------------
# Support Vector Machine Model
# -------------------------
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
# -------------------------
# Confusion Matrix (SVM)
# -------------------------
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - SVM")
plt.show()
# -------------------------
# Prediction Function
# -------------------------
def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = svm_model.predict(text_vector)
    return "SPAM" if prediction[0] == 1 else "HAM"
# -------------------------
# Example Prediction
# -------------------------
email = "Congratulations! You won a free gift card. Click now!"
print("\nEmail Text:", email)
print("Prediction:", predict_email(email))
