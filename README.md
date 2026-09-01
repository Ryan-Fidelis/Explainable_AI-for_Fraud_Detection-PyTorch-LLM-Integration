# Explainable AI for Fraud Detection: PyTorch & LLM Integration

## Overview
This project is an end-to-end machine learning pipeline designed to detect corporate financial fraud and explain the reasoning behind its predictions using Generative AI.

In real-world financial systems, black-box neural networks are often insufficient because auditors and managers need to know why a transaction was blocked. This repository solves that by combining a PyTorch Deep Learning classifier (optimized for highly imbalanced datasets) with the Google Gemini API to generate human-readable, business-contextualized fraud analysis reports (Explainable AI - XAI).

## Architecture & Workflow
### Before you continue, it`s mandatory to understand that the entire dataset is sintetic with a semi-realistic normal-case to fraud-case imbalance

**Synthetic Fraud Injection:** Injects realistic fraud patterns (Smurfing, Out-of-Hours anomalies, Policy Violations via Z-Score, and Shell Merchant Collusion) into a clean corporate dataset, creating a realistic ~2.4% fraud imbalance.

**Feature Engineering:** Makes temporal data Cyclical (Sine/Cosine), Z-score calculations based on department averages (without separating transaction per month, a way to simplify the code), and One-Hot Encoding for categorical variables.

**Deep Learning Classification:** A Feed-Forward Neural Network built with PyTorch. It uses BCEWithLogitsLoss with a calculated positive weight (pos_weight) to heavily penalize missed frauds, bypassing the need for synthetic oversampling (like SMOTE).

**Explainable AI (XAI):** Anomalous tensors predicted as fraud are parsed and sent to the Gemini LLM, which explain the mathematical anomalies in plain English.


## Model Performance & Business Metrics
In fraud detection, Recall (capturing as many real frauds as possible) is often prioritized over Precision (avoiding false alarms). The model's decision threshold was mathematically adjusted (Threshold = 0.35) to ensure a high capture rate, successfully stopping over 92% of malicious transactions.

**Test Set Results:**
``` bash
Accuracy: 83.16%

Precision: 0.1202

Recall: 0.9215 (Out of all real frauds, the model successfully blocked 92.15%)

F1-Score: 0.2127
```
**Confusion Matrix:**
```Plaintext
[[12947  2664]   <-- [True Negatives, False Positives]
 [   31   364]]  <-- [False Negatives, True Positives]
 ```
## Explainable AI (XAI) Output Examples
When the PyTorch model flags a transaction (Prediction = 1.0), the data is routed to the Gemini API, as a dictionary, for contextual analysis.

```Plaintext
--- GEMINI FRAUD ANALYSIS ---
### **FRAUD ALERT: High Probability (Score: 1.0)**

#### **1. Likely Fraud Type**
**Unauthorized Personal Expense / Corporate Card Misuse** (Micro-Expense Policy Evasion)

---

#### **2. Suspicious Indicators & Justification**

* **Non-Business Merchant Category (`Category_Clothing: 1.0`):**  
  The corporate payment was processed under the *Clothing* category, which generally represents personal, non-operational expenditure unaligned with standard business procurement.

* **Weekend / Off-Hours Timing (`day_sin: -0.78` / `day_cos: 0.62`):**  
  The cyclical day encoding indicates the transaction occurred over the **weekend**, outside regular corporate operational hours.

* **Low-Amount Threshold Evasion (`amount_z_score: -0.42`):**  
  The transaction value is below the average spend (`z-score < 0`). Sub-threshold amounts are frequently used by bad actors to circumvent manual managerial sign-offs or automated approval limits.

* **Risk-Prone User Profile (`Role_Level_Júnior: 1.0`):**  
  The transaction was initiated by a Junior-level employee, a role tier that typically has strict spending controls and limited authorization for discretionary or off-hours purchasing.

---

#### **3. Recommended Action**
* **Immediate Action:** Freeze transaction settlement / place temporary hold on card.
* **Audit Request:** Flag transaction to the employee's direct supervisor to submit itemized receipts and business justification within 24 hours.
========================================

--- GEMINI FRAUD ANALYSIS ---
**FRAUD ALERT ANALYSIS REPORT**

**Likely Fraud Type:** **Unauthorized Personal Expense / Corporate Card Misuse**
**Model Confidence:** High (Prediction: 1.0)

---

### **Executive Summary**
This transaction exhibits key indicators of unauthorized personal expenditure by a low-level employee. The combination of a non-business merchant category, weekend timing, and a low transaction value suggests potential policy violation or intentional misuse of corporate funds.

---

### **Key Data Justifications**

1. **Non-Compliant Merchant Category:**
   * **`Category_Health & Beauty: 1.0`** — Expenses in personal care/beauty rarely serve a legitimate corporate purpose, contrasting with standard business categories (e.g., IT, Services, Travel).

2. **User Profile & Authorization Risk:**
   * **`Role_Level_Júnior: 1.0`** — Junior-level employees typically have restricted spending privileges and lower policy thresholds, making personal expense claims higher risk for compliance.

3. **Off-Hours / Weekend Timing:**
   * **`day_sin: -0.7818` / `day_cos: 0.6235`** — Cyclical features map to a weekend/non-business day, indicating the transaction occurred outside normal operating hours.

4. **Micro-Spending Pattern:**
   * **`amount_z_score: -0.9266`** — Transaction amount is significantly below average. This low-value profile is characteristic of policy-testing ("flying under the radar") or casual personal charges aimed at avoiding manager approval limits.

---

### **Recommended Action**
* **Flag for Audit:** Request itemized receipt and business justification from the employee.
* **Temporary Restriction:** Restrict corporate card use for non-standard categories pending review.
========================================

--- GEMINI FRAUD ANALYSIS ---
### **FRAUD ALERT ANALYSIS REPORT**

**Likely Fraud Type:** **Personal Expense Misappropriation / Out-of-Hours Policy Breach**

---

### **Executive Summary**
The model has flagged this transaction due to a combination of **non-compliant merchant categorization**, **anomalous transaction timing**, and an **elevated spend amount** associated with a mid-level staff role (*Pleno*).

---

### **Key Risk Indicators & Data Justification**

1. **Non-Standard Merchant Category (`Category_Health & Beauty = 1.0`)**
   * **Analysis:** The purchase was made under *Health & Beauty*, a category with no obvious connection to standard corporate procurement or operational needs.
   * **Risk:** High probability of personal misuse of corporate funds or purchasing cards.

2. **Temporal Anomaly (Weekend & Early Morning)**
   * **Analysis:**
     * `day_sin/cos` values (`-0.7818`, `0.6234`) indicate a **weekend** transaction.
     * `hour_sin/cos` values (`0.9659`, `-0.2588`) map to approximately **07:00 AM**.
   * **Risk:** High correlation with non-working-hour personal activity rather than legitimate business travel or emergency operations.

3. **Role Level vs. Spend Amount (`amount_z_score = 0.5613`, `Role_Level_Pleno = 1.0`)**
   * **Analysis:** The transaction amount is above average ($z\text{-score} > 0.5$) for a mid-level (*Pleno*) employee baseline.
   * **Risk:** Spending exceeds normal operational thresholds for mid-level discretionary authorization without explicit pre-approval.

---

### **Recommended Action**
* **Status:** **HOLD / REJECT**
* **Action:** Request itemized tax receipts and explicit business justification from the employee and their line manager before clearing the disbursement.
========================================
```
# How to Run Locally
#### 1. Clone this repository.
```bash
git clone [https://github.com/Ryan-Fidelis/Explainable_AI-for_Fraud_Detection-PyTorch-LLM-Integration.git](https://github.com/Ryan-Fidelis/Explainable_AI-for_Fraud_Detection-PyTorch-LLM-Integration.git)
```
#### 2. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn torch matplotlib python-dotenv google-generativeai
```

#### 3. Create a .env file in the root directory and add your Google Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```
#### 4. Run the main pipeline:
```bash
python classif_model.py
```

## Author
Developed by Ryan Gabriel de Souza Lopes Fidelis, currently pursuing a degree in Artificial Intelligence (Superior de Tecnologia em Inteligência Artificial) at Centro Universitário Senac.
