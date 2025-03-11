Postpartum Depression Prediction
A machine learning-based approach to predict postpartum depression (PPD) using maternal mental health data. This project explores key risk factors and develops predictive models to aid early detection and intervention.

Project Overview
Postpartum depression is a serious mental health concern affecting new mothers, yet it is often underdiagnosed. This project applies machine learning models to identify key predictors of PPD and develop an accurate classification system.

We compare multiple models, including Logistic Regression, Random Forest, and XGBoost, and evaluate their performance based on accuracy, precision, recall, and F1-score.

Dataset
Source: Kaggle's Maternal Mental Health and Infant Sleep Dataset
Size: Includes multiple maternal health indicators
Preprocessing Applied:
Handling missing values
Feature scaling & encoding
Data splitting (70% Train, 15% Validation, 15% Test)
Key Features (Input Variables)
Demographics: Age, marital status
Mental Health Indicators: Feeling sad, irritability, anxiety, guilt, concentration issues
Sleep Factors: Nighttime sleep duration, trouble sleeping
Behavioral Factors: Appetite changes, bonding issues with baby
Target Variable (Output)
Postpartum Depression Risk (Yes/No)
Machine Learning Models Used
Logistic Regression (Baseline)
Random Forest
XGBoost (Best Model - 98.2% Accuracy)
Model Performance Comparison
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	58.2%	57.9%	58.2%	57.4%
Random Forest	97.7%	97.8%	97.7%	97.7%
XGBoost (Best)	98.2%	98.3%	98.2%	98.2%
XGBoost outperformed other models and showed high reliability in classifying postpartum depression cases.

Results & Findings
Top Predictors of PPD:

Trouble sleeping at night
Feelings of guilt
Problems concentrating or making decisions
Confusion Matrix & ROC Curve:

The model correctly identifies high-risk cases with minimal false positives.
AUC Score: 0.79, showing strong discrimination capability.

Challenges:
Data Imbalance (fewer positive PPD cases).
Lack of direct race/ethnicity data, requiring proxy variables.
Future Improvements
Access PRAMS data to include race/ethnicity for demographic fairness.
Use deep learning models (Neural Networks) for improved accuracy.
Expand dataset with additional maternal health factors.
Deploy as an AI tool for use in hospitals & maternal healthcare centers.

How to Run the Project

1. Clone the Repository
git clone https://github.com/yourusername/postpartum-depression-prediction.git
cd postpartum-depression-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Run the Model
python train_model.py

4. View Results & Metrics
python evaluate_model.py

Contributors
Dawnena Key - Data Science Researcher
University of Denver

Contact: dawnena.key@du.edu

License
This project is licensed under the MIT License.


