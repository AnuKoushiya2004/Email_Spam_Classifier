# Email_Spam_Classifier
📧 Email Spam Classifier & Analysis System

A Machine Learning–based application that classifies emails/messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques and visualizes results using evaluation metrics.

I. Overview

The Email Spam Classifier is designed to automatically detect unwanted spam emails by analyzing the content of messages. The system preprocesses text data, extracts features using TF-IDF, and applies multiple machine learning models to achieve accurate classification.

This project demonstrates the complete ML workflow from data preprocessing to model evaluation and prediction.

II. Project Modules

This project is divided into 6 major modules:

Data Loading Module

Loads the spam dataset from CSV

Selects and renames required columns

Data Preprocessing Module

Label encoding (Spam / Ham)

Text cleaning and preparation

Feature Extraction Module

TF-IDF Vectorization

Stop-word removal and feature limiting

Model Training Module

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Model Evaluation Module

Accuracy score

Classification report

Confusion matrix visualization

Prediction Module

Classifies new/unseen email text

Outputs Spam or Ham result

III. Technologies Used
Backend & Core

Python 🐍

Pandas, NumPy

Scikit-learn

NLP

TF-IDF Vectorizer

Text preprocessing techniques

Visualization

Matplotlib

Seaborn

IV. Features
1️⃣ Email Spam Detection

Classifies messages into Spam or Ham

Works on real-world text input

2️⃣ Multiple ML Models

Naive Bayes

Logistic Regression

Support Vector Machine (Best Accuracy)

3️⃣ Performance Evaluation

Accuracy comparison across models

Precision, Recall, F1-score

4️⃣ Confusion Matrix Visualization

Graphical representation of predictions

Easy interpretation of model performance

5️⃣ Custom Email Prediction

User can test any email message

Instant classification output

V. How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/Email-Spam-Classifier.git
cd Email-Spam-Classifier

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Program
python spam_classifier.py

VI. Sample Output
SVM Accuracy: 0.98

Email Text: Congratulations! You won a free gift card. Click now!
Prediction: SPAM


✔ Confusion Matrix is displayed using a heatmap.

VII. Testing

Train–test split validation

Model comparison testing

Prediction testing with custom inputs

VIII. Future Enhancements

Deploy as a web application using Flask/Streamlit

Integrate Deep Learning models (LSTM, BERT)

Save trained models using pickle

Add real-time email integration

📜 License

This project is licensed under the MIT License — free for academic and personal use.

🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to fork the repository and submit a pull request.

⭐ Acknowledgements

Scikit-learn for ML models

Matplotlib & Seaborn for visualization

Open-source NLP community
