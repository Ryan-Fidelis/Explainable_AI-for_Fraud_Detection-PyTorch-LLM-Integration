# Explainable_AI-for_Fraud_Detection-PyTorch-LLM-Integration

## Code Output
```bash
Starting Training...
Epoch [10/100], Average Loss: 0.5588
Epoch [20/100], Average Loss: 0.5309
Epoch [30/100], Average Loss: 0.5065
Epoch [40/100], Average Loss: 0.5050
Epoch [50/100], Average Loss: 0.4898
Epoch [60/100], Average Loss: 0.4843
Epoch [70/100], Average Loss: 0.4815
Epoch [80/100], Average Loss: 0.4843
Epoch [90/100], Average Loss: 0.4856
Epoch [100/100], Average Loss: 0.4731

Test Set Accuracy (Threshold=0.35): 83.16%
------------------------------
EXTRA METRICS (Fraud Detection)
------------------------------
Precision: 0.1202 (When the model predicts Fraud, how often is it right?)
Recall:    0.9215 (Out of all real Frauds, how many did the model find?)
F1-Score:  0.2127 (Harmonic mean of Precision and Recall)

Confusion Matrix:
[[12947  2664]
 [   31   364]]

Total of detected frauds: 3028
Generating AI explanations for three random cases...


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
