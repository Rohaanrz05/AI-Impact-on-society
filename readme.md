PROJECT REPORT
AI Social Impact Analysis & Prediction System

Date: January 2026
Subject: Data Science & Machine Learning Implementation Report

--------------------------------------------------
1. Executive Summary
--------------------------------------------------
This project was initiated to analyze the public's perception of Artificial Intelligence (AI) and its perceived impact on society. By leveraging a dataset of survey responses, we developed an end-to-end analytical pipeline that includes data cleaning, exploratory data analysis (EDA), and a dual-target machine learning system.

The final deliverable is a high-performance interactive dashboard named "AI Nexus" that allows users to visualize trends and predict user characteristics with high accuracy.

--------------------------------------------------
2. Project Objectives
--------------------------------------------------
The primary objective of this project was to move beyond basic data visualization and develop a predictive analytical system. The key objectives included:

- Data Standardization:
  Transform raw and inconsistent survey data into a clean, structured dataset suitable for analysis.

- Public Sentiment Analysis:
  Analyze how demographics such as age, gender, and education influence trust in AI and fear of job displacement.

- Predictive Modeling:
  Objective A: Predict a user's Employment Status based on behavioral and demographic features.
  Objective B: Predict a user's Level of Trust in AI based on knowledge and demographics.

- Model Benchmarking:
  Compare Random Forest and XGBoost models to identify the optimal solution.

--------------------------------------------------
3. Methodology
--------------------------------------------------

Phase 1: Data Acquisition & Cleaning
-----------------------------------
The initial dataset consisted of raw survey responses containing inconsistent labels and languages. Using Python (Pandas), the following preprocessing steps were performed:

- Column Renaming:
  Converted long survey questions into concise, code-friendly column names.

- Translation & Standardization:
  Standardized inconsistent values.
  Examples:
  - "Emekliyim" converted to "Retired"
  - "Ön Bachelor's degree" standardized to "Associate Degree"
  - Sentiment labels mapped (e.g., "Definitely Becomes" → "Yes")

- Outlier Handling:
  Applied Winsorization to numerical features such as AI_Usage_Scale to reduce the impact of extreme values.

- Feature Reduction:
  Removed irrelevant columns such as ID and Occupation to minimize noise.

Phase 2: Exploratory Data Analysis (EDA)
---------------------------------------
Visualization libraries such as Plotly and Seaborn were used to identify trends:

- Trust Dynamics:
  A positive correlation was observed between AI knowledge and trust.

- Job Security:
  Fear of job loss due to AI was present across all education levels.

Phase 3: Machine Learning Implementation
----------------------------------------
A dual-target prediction system was implemented:

- Preprocessing:
  LabelEncoder was used to convert categorical features into numerical format.

- Data Splitting:
  Dataset was divided into 80% training and 20% testing sets.

--------------------------------------------------
4. Analysis of Models Used
--------------------------------------------------

Model 1: Random Forest Classifier
--------------------------------
Description:
An ensemble bagging algorithm that constructs multiple decision trees and aggregates their predictions.

Advantages:
- Robust against overfitting
- Handles non-linear relationships effectively
- Feature importance is easily interpretable

Disadvantages:
- Larger model size
- Slower predictions with very large forests

Model 2: XGBoost (Extreme Gradient Boosting)
-------------------------------------------
Description:
A boosting algorithm that builds trees sequentially, correcting previous errors.

Advantages:
- State-of-the-art accuracy
- Built-in regularization
- Optimized for speed and performance

Disadvantages:
- More sensitive to outliers
- Requires careful hyperparameter tuning

Model Comparison:
-----------------
XGBoost consistently outperformed Random Forest by approximately 2–5% accuracy. This justified its selection as the primary model for the Prediction Lab.

--------------------------------------------------
5. Success & Conclusion
--------------------------------------------------
All project objectives were successfully achieved:

- A fully operational Streamlit dashboard (AI Nexus) with modern glassmorphism UI.
- High-accuracy predictions for Employment Status and Trust in AI.
- Actionable insights revealing that fear of AI is not limited to low-skilled or low-education groups.

This project demonstrates the effective integration of data science, machine learning, and interactive visualization to address real-world societal questions.
