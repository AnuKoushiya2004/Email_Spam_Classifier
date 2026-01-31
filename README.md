# Email_Spam_Classifier
📧 Email Spam Classifier

A Machine Learning–based Email Spam Classification project that detects whether an email is Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques and multiple ML models.

🚀 Project Overview

This project uses TF-IDF vectorization to convert email text into numerical features and applies different machine learning algorithms to classify emails:

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Among these, SVM provides the best performance and is used for final predictions.

📂 Repository Structure
Email-Spam-Classifier/
│
├── data/
│   └── spam.csv              # Dataset
│
├── spam_classifier.py        # Main Python script
├── requirements.txt          # Required libraries
└── README.md                 # Project documentation

📊 Dataset

Source: SMS/Email spam dataset

Columns Used:

label → ham (0) or spam (1)

message → email/text content

Dataset is stored inside the data/ folder.

🛠️ Technologies Used

Python 🐍

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

NLP (TF-IDF Vectorization)

🧠 Machine Learning Models

Multinomial Naive Bayes

Logistic Regression

Support Vector Machine (LinearSVC) ✅ (Best Performing Model)

📈 Evaluation Metrics

Accuracy Score

Precision, Recall, F1-Score

Confusion Matrix (visualized using Seaborn)

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/Email-Spam-Classifier.git
cd Email-Spam-Classifier


Install dependencies

pip install -r requirements.txt


Run the project

python spam_classifier.py

🧪 Example Output
Email Text: Congratulations! You won a free gift card. Click now!
Prediction: SPAM

🔍 Prediction Function

The project includes a function to classify new emails:

def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = svm_model.predict(text_vector)
    return "SPAM" if prediction[0] == 1 else "HAM"

📌 Key Highlights

Uses TF-IDF for feature extraction

Compares multiple ML models

Visualizes confusion matrix

Easy-to-use prediction function

Beginner-friendly & interview-ready project

🎯 Future Improvements

Add deep learning models (LSTM, BERT)

Deploy as a web app using Flask/Streamlit

Add email preprocessing (lemmatization, stemming)

Save and load trained models
