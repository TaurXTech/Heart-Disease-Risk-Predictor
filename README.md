# ❤️ Heart Disease Risk Predictor

A machine learning web application that predicts the likelihood of heart disease using patient clinical measurements, lifestyle factors, biomarkers, and family history.

The project was developed using Python, Scikit-learn, and Streamlit, with a complete machine learning pipeline from data preprocessing to deployment.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early risk prediction can assist healthcare professionals in identifying high-risk individuals for further medical evaluation.

This project applies supervised machine learning to predict whether a patient is at risk of heart disease based on demographic information, medical history, lifestyle habits, and laboratory measurements.

---

## 🚀 Live Demo

**Streamlit App:**

(Add your Streamlit link here)

Example:

https://your-app-name.streamlit.app

---

## 📊 Dataset

Dataset: Heart Disease Prediction Dataset

Features include:

- Age
- Gender
- Blood Pressure
- Cholesterol Level
- BMI
- Exercise Habits
- Smoking
- Diabetes
- Family Heart Disease
- Alcohol Consumption
- Stress Level
- Sleep Hours
- Sugar Consumption
- High Blood Pressure
- High LDL Cholesterol
- Low HDL Cholesterol
- Triglyceride Level
- Fasting Blood Sugar
- CRP Level
- Homocysteine Level

Target:

- Heart Disease Status

---

## 🧹 Data Preprocessing

The dataset was cleaned before training by:

- Handling missing values
- Removing duplicate records
- Encoding categorical variables
- Feature engineering
- Data preprocessing using Scikit-learn Pipeline

---

## ⚙️ Feature Engineering

Several additional features were created to improve model performance.

- BMI Category
- Age × Blood Pressure
- Cholesterol × BMI
- Inflammation Score
- Lifestyle Risk Score
- Cardiovascular Risk Score
- Family Risk Score

---

## 🤖 Machine Learning Models Evaluated

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

After evaluation, **Gradient Boosting** achieved the best performance and was selected as the final model.

The model was further optimized using **GridSearchCV**.

---

## 📈 Model Evaluation

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

## 🌐 Streamlit Application

The web application allows users to:

- Enter patient information
- Predict heart disease risk
- Display prediction probabilities
- View raw model inputs
- Interact with an intuitive interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

---

## 📂 Project Structure

```
Heart-Disease-Predictor/
│
├── app.py
├── heart_disease_model.pkl
├── Heart_Disease_Prediction.ipynb
├── heart_disease.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Heart-Disease-Predictor.git
```

Move into the project folder

```bash
cd Heart-Disease-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Add screenshots of your Streamlit application here.

Example:

- Home Page
- Prediction Result
- Probability Chart

---

## ⚠️ Disclaimer

This application is intended solely for educational and demonstration purposes.

It should not be used as a substitute for professional medical diagnosis or treatment.

Always consult a qualified healthcare professional regarding medical concerns.

---

## 👨‍💻 Author

**Taurai Mvundusi**

GitHub:
https://github.com/yourusername

LinkedIn:
(Add your LinkedIn profile)

---

## ⭐ If you found this project useful

Please consider giving the repository a ⭐ to support the project.
