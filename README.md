# Explainable AI for Fraud Detection: PyTorch & LLM Integration

## Overview
This project is an end-to-end machine learning pipeline designed to detect corporate financial fraud and explain the reasoning behind its predictions using Generative AI.

In real-world financial systems, black-box neural networks are often insufficient because auditors and managers need to know why a transaction was blocked. This repository solves that by combining a PyTorch Deep Learning classifier (optimized for highly imbalanced datasets) with the Google Gemini API to generate human-readable, business-contextualized fraud analysis reports (Explainable AI - XAI).

## Architecture & Workflow
**Synthetic Fraud Injection:** Injects realistic fraud patterns (Smurfing, Out-of-Hours anomalies, Policy Violations via Z-Score, and Shell Merchant Collusion) into a clean corporate dataset, creating a realistic ~2.4% fraud imbalance.

**Feature Engineering:** Applies Cyclical Encoding (Sine/Cosine) for temporal data, Z-score calculations based on department averages, and One-Hot Encoding for categorical variables.

**Deep Learning Classification:** A Feed-Forward Neural Network built with PyTorch. It uses BCEWithLogitsLoss with a calculated positive weight (pos_weight) to heavily penalize missed frauds, bypassing the need for synthetic oversampling (like SMOTE).

**Explainable AI (XAI):** Anomalous tensors predicted as fraud are parsed and sent to the Gemini LLM, which acts as a virtual fraud analyst to explain the mathematical anomalies in plain business English.


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
When the PyTorch model flags a transaction (Prediction = 1.0), the data is routed to the Gemini API for contextual analysis.

Note: Two additional generated case explanations will be added shortly to this documentation.

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
