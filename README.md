# **AI-Powered Student Dropout Prediction System**


## **Overview**
This repository contains an AI system for predicting high school students at risk of dropping out within the next academic year. The solution enables early interventions by educators and counselors through automated risk scoring. The implementation follows a comprehensive workflow from problem scoping to deployment, with special attention to ethical considerations and regulatory compliance.



## **Part 1: Student Dropout Prediction Framework**
### Problem Definition
**Objective**: Develop predictive model to identify at-risk students with:
- ≥90% recall (minimize false negatives)
- ≤15% false positive rate
- 40% reduction in dropout rates among flagged students

  

**Core KPI**:  
**Dropout Prevention Rate (DPR)**  
`DPR = (1 - (Predicted At-Risk Dropouts / Total Predicted At-Risk)) × 100%`  
*Target: ≥85% within first year*

### Data Strategy
**Sources**:
1. Student Information Systems (SIS)
   - Academic records, attendance, demographics
2. Support Services Databases
   - Counseling logs, behavioral incidents, interventions

**Preprocessing Pipeline**:

A[Raw Data] --> B[Missing Value Handling]
B --> C[Normalization]
C --> D[Categorical Encoding]
D --> E[Feature Engineering]

## **Part 2: Patient Readmission Case Study**
Implementation Workflow
Data Collection: EHRs, ADT systems, claims data

Feature Engineering:

Temporal features (days since last admission)

Comorbidity indices (Elixhauser)

Social determinants (Area Deprivation Index)

Model: XGBoost with SHAP explainability

Compliance: HIPAA-compliant data handling

Performance Metrics
Actual \ Predicted	Readmit	Not Readmit
Readmit	80 (TP)	30 (FN)
Not Readmit	20 (FP)	170 (TN)
Precision: 80%

Recall: 72.7%

## **Ethical Framework**
Bias Mitigation Strategies
Underrepresented Groups:

Stratified sampling during training

Loss re-weighting (1:5 minority ratio)

Proxy Variables:

Avoid ZIP code as race indicator

Area Deprivation Index instead of raw demographics

Compliance Measures
PHI anonymization before processing

Role-based access controls (RBAC)

Audit trails for data access

On-premise deployment option


## **Installation**

git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt

## **Usage**
python
from src.preprocessing import preprocess_data
from src.train import train_model

# Preprocess dataset
X_train, y_train = preprocess_data('data/student_records.csv')

# Train model
model = train_model(X_train, y_train, max_depth=10, class_weight={0:1, 1:5})


