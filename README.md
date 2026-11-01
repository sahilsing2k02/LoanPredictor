# 🤖 Loan Default Prediction – Automating Bank Credit Processes Using AI

**Author / Maintainer:** Sahil Singh  

This project applies AI and machine learning to predict whether a customer is likely to default on a loan, enabling banks to make informed and automated credit decisions.

---

## 📊 Problem Statement

Banks often struggle to accurately assess credit risk and loan default probability. This project automates that process by using classification models to predict default based on applicant data.

---

## 🧠 Key Features

- Predicts loan defaults using historical banking data
- Compares multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, etc.)
- Evaluates models via ROC-AUC, F1 score, Precision, and Recall
- Provides explainability with **SHAP** (SHapley Additive exPlanations)
- Visualization of feature importance and model decisions
- End-to-end pipeline from data preprocessing to model interpretation

---

## 🧰 Tech Stack

- **Python**
- **scikit-learn**
- **XGBoost**
- **SHAP**
- **pandas**, **matplotlib**, **seaborn**

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/sahilsing2k02/LoanPredictor.git
cd LoanPredictor
```

2. **Install Dependencies**

```bash
pip install -r back_end/requirements.txt
```

3. **Run the Full Application (Backend + Frontend)**

Run the backend server (serves both API and frontend on port 5000):

```bash
python back_end/main.py
```

Then visit: `http://127.0.0.1:5000/`

Or run separately:

```bash
# Terminal 1: Backend API
python back_end/main.py

# Terminal 2: Frontend UI
python -m http.server 8000 --directory front_end
```

Then visit: `http://127.0.0.1:8000/`

---

## 📈 Model Evaluation

* **Best Model:** LightGBM (100-tree ensemble with compiled pure Python AST inference)
* **ROC-AUC:** High discriminative power
* **SHAP Values:** Used to interpret individual predictions and global feature impact

---

## 📌 Highlights

* Automated model selection and tuning
* Transparent credit risk scoring using SHAP
* Business-focused evaluation for banking applications
* Zero-dependency pure Python inference engine for instant serverless cloud deployment

---

## 📎 Resources

* 📚 [Comprehensive Study Review & Viva Guide](STUDY_REVIEW.md)
* 📄 [Project Report (PDF)](Automating_bank_credit_processes_using_AI.pdf)

---

## 📜 License & Attribution

This project is distributed under the MIT License.
* Adapted and extended by **Sahil Singh** for academic coursework.
* Original base implementation by **aimaster-dev**.

---

## 🙌 Acknowledgments

* Thanks to `aimaster-dev` for the baseline architecture and financial ML pipeline.
* Dataset provided by Burak Ergun via Kaggle ([Loan Data Set](https://www.kaggle.com/datasets/burak3ergun/loan-data-set)).
