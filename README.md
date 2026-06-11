# Telco-Customer-Churn-Analysis
#https://telco-customer-churn-analysis-bacaj2rwr4odga9658bzok.streamlit.app/
# 📡 Telco Customer Churn Analysis
### End-to-End Machine Learning Web Application

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.20-purple?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## 🔗 Live Demo

> 🌐 **[Launch App →](https://telco-customer-churn-analysis.streamlit.app)**

---

## 📌 Project Overview

Customer churn — the phenomenon where subscribers discontinue a service — is one of the most critical business problems in the telecommunications industry. Acquiring a new customer costs **5–7× more** than retaining an existing one, making churn prediction a high-value machine learning problem.

This project presents a **full-stack, production-ready data science application** built entirely from scratch. Starting from raw data acquisition on Kaggle, through exploratory analysis, feature engineering, model training, and finally a deployed interactive web application — every step was independently designed and implemented.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Records** | 7,043 customers |
| **Features** | 21 attributes (demographics, services, billing) |
| **Target** | `Churn` — Yes / No |
| **Class Imbalance** | ~26.5% churned, ~73.5% retained |

The dataset was manually downloaded from Kaggle, inspected for quality issues, and preprocessed without relying on any pre-cleaned versions.

---

## ⚙️ Technical Pipeline

### 1. Data Acquisition & Cleaning
- Downloaded raw CSV from Kaggle
- Identified and handled **`TotalCharges`** column stored as string (whitespace-encoded nulls)
- Imputed missing values using **median strategy**
- Removed non-predictive identifier columns (`customerID`)

### 2. Feature Engineering & Preprocessing
- Applied **Label Encoding** to all categorical variables (11 columns)
- Scaled all numerical features using **StandardScaler** for model compatibility
- Ensured zero data leakage — preprocessing fitted only on training split

### 3. Exploratory Data Analysis (EDA)
- Analyzed churn distribution across contract types, internet services, and payment methods
- Generated correlation heatmaps to identify multicollinearity
- Visualized tenure vs. monthly charges scatter patterns segmented by churn status

### 4. Model Training & Evaluation
Three classifiers were trained and evaluated with cross-validation:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~80% | ~0.84 |
| Random Forest | ~82% | ~0.87 |
| Gradient Boosting | ~81% | ~0.86 |

Evaluation metrics include **Confusion Matrix**, **ROC Curve**, **Classification Report**, and **Feature Importance** rankings.

### 5. Customer Segmentation (Unsupervised)
- Applied **K-Means Clustering** (k = 2–8, user-adjustable)
- Reduced dimensionality with **PCA (2 components)** for visualization
- Identified cluster-wise churn rates to reveal high-risk customer segments

### 6. Real-Time Prediction Interface
- Interactive form collects all 20 customer attributes
- Trained Random Forest model returns **churn probability** in real time
- Results displayed with a gauge chart and color-coded risk indicator

---

## 🖥️ Application Features

| Tab | Description |
|---|---|
| **Overview** | KPI cards, churn distribution, contract analysis, tenure histogram |
| **EDA** | Boxplots, bar charts, correlation heatmap, scatter plots |
| **Models** | Train any of 3 classifiers, view ROC curve, confusion matrix, feature importance |
| **Clustering** | K-Means with PCA visualization and per-cluster churn rate |
| **Predict** | Single-customer real-time churn probability with gauge meter |

---

## 🏗️ Project Structure

```
telco-customer-churn-analysis/
│
├── churn_app.py                        # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset (Kaggle)
├── OEL_daniyal.ipynb                   # Jupyter notebook (EDA & modeling)
└── README.md                           # Project documentation
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/daniyal-hussain2528/telco-customer-churn-analysis.git
cd telco-customer-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run churn_app.py
```

> Python **3.11** recommended. The app also works with the built-in demo dataset — no CSV upload required.

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.20.0
```

---

## 🧠 Key Learnings

- Handling real-world **messy data** (type mismatches, encoded nulls, class imbalance)
- Building **reusable preprocessing pipelines** that avoid data leakage
- Designing **multi-page Streamlit applications** with dynamic state management
- Implementing both **supervised** (classification) and **unsupervised** (clustering) ML workflows in a single application
- Deploying a live ML application via **Streamlit Cloud + GitHub CI/CD**

---

## 👨‍💻 Author

**Daniyal Hussain**
OEL Project — Machine Learning & Data Science
📧 GitHub: [@daniyal-hussain2528](https://github.com/daniyal-hussain2528)

---

> *"The goal is to turn data into information, and information into insight."*
> — Carly Fiorina
