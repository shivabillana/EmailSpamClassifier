# 📧 Spam Email Detection

This project implements a **machine learning model** to classify emails as **spam** or **ham (not spam)** using **Naive Bayes** and **Logistic Regression** with **TF-IDF text features**.

## 🚀 Features

- Preprocessing of email text with **TF-IDF Vectorizer**
- Models trained:

  - **Multinomial Naive Bayes**
  - **Logistic Regression**

- Evaluation using:

  - **Classification Report**
  - **Confusion Matrix**

## 📂 Dataset

The project uses a dataset of spam and non-spam (ham) emails.

> ⚠️ Note: The dataset (`spam_Emails_data.csv`) is large (350MB) and tracked with **Git LFS**.

## 🛠️ Installation

Clone the repository and install dependencies:

git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt

## ▶️ Usage

Run the script to train and evaluate the models:

python spam_detector.py

Or open the notebook:

jupyter notebook

## 📊 Results

Both **Naive Bayes** and **Logistic Regression** models are evaluated using precision, recall, F1-score, and confusion matrix to measure performance.

## 📦 Requirements

See requirements.txt for dependencies.
