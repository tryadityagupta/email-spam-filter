# 📧 Email/SMS Spam Classifier

A machine learning web application that classifies messages as **Spam** or **Ham (Not Spam)** using **Logistic Regression** and **Flask**.

---

## 🚀 Overview

This project uses the **SMS Spam Collection Dataset** to train a text classification model.
The trained model is deployed using a **Flask web application** that allows users to input messages and get real-time predictions.

---

## ✨ Features

* Text preprocessing (lowercasing, regex cleaning)
* TF-IDF vectorization with **n-grams (1,2)**
* Logistic Regression with **class balancing**
* Real-time predictions via Flask web app
* Automatic browser launch on server start
* Clean and modular `.py` pipeline (no notebooks)

---

## 🧠 Model Details

* **Algorithm:** Logistic Regression
* **Vectorizer:** TF-IDF (`ngram_range=(1,2)`)
* **Preprocessing:** Lowercasing + special character removal
* **Class Imbalance Handling:** `class_weight='balanced'`
* **Dataset:** SMS Spam Collection (UCI)

---

## 📁 Project Structure

```
email-spam-filter/
├── app.py              # Flask web app
├── train.py            # Model training script
├── templates/
│   └── index.html      # Frontend UI
├── SMSSpamCollection   # Dataset
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/tryadityagupta/email-spam-filter.git
cd email-spam-filter
```

---

### 2. Create virtual environment

**Windows:**

```
python -m venv spambot_env
spambot_env\Scripts\activate
```

**macOS/Linux:**

```
python3 -m venv spambot_env
source spambot_env/bin/activate
```

---

### 3. Install dependencies

```
pip install pandas scikit-learn flask joblib
```

---

## ▶️ Usage

### 1. Train the model

```
python train.py
```

---

### 2. Run the Flask app

```
python app.py
```

👉 The browser will open automatically at:
`http://127.0.0.1:5000`

---

## 🧪 Example Predictions

| Input Message             | Prediction |
| ------------------------- | ---------- |
| "free lottery win cash"   | Spam       |
| "hey let's meet tomorrow" | Ham        |
| "urgent! call now"        | Spam       |

---

## 🔧 How It Works

1. Text is cleaned using regex
2. Converted into numerical features using TF-IDF
3. Logistic Regression model predicts spam/ham
4. Flask app displays the result in real-time

---

## 🔮 Future Improvements

* Improve UI/UX design
* Deploy online (Render / Railway)
* Add model evaluation metrics in UI
* Try advanced models (SVM, Deep Learning)
* Support email datasets

---

## 🛠 Technologies Used

* Python 3
* pandas
* scikit-learn
* Flask
* joblib

---

## 📊 Dataset

* **Name:** SMS Spam Collection
* **Size:** 5,574 messages
* **Source:** UCI Machine Learning Repository

---

## 👨‍💻 Author

**Aditya Gupta**
GitHub: https://github.com/tryadityagupta

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
