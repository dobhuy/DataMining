# 🧠 Understanding and Predicting Depression to Enhance Mental Health Interventions

This project was developed as **Data Mining and Application**, University of Information Technology (VNU-HCM).  
It focuses on **using machine learning techniques to identify and predict depression** based on survey data, contributing to early mental health intervention.

---

## 🧩 Introduction

Mental health is a major global issue, with depression being one of the most prevalent disorders.  
This project aims to leverage **data mining and machine learning** to detect individuals at risk of depression based on their **biological characteristics and lifestyle habits**.

The predictive model provides insights for **mental health organizations, policymakers, and healthcare providers** to design targeted interventions and awareness programs.

---

## 📊 Dataset

- **Source:** Synthetic mental health survey dataset  
- **Number of features:** 19  
- **Target variable:** `Depression` (Binary classification)  
- **Key features include:** Age, Profession, Work Pressure, Financial Stress, Sleep Duration, etc.  
- **Data preprocessing:**  
  - Handle missing values (median/mode imputation)  
  - Label encoding for categorical features  
  - Normalization using `StandardScaler`  
  - Outlier detection using IQR method  

---

## ⚙️ Approach

1. **Data Cleaning & Preprocessing**
   - Dropped columns with >70% missing values  
   - Imputed missing values  
   - Normalized numerical data  

2. **Exploratory Data Analysis (EDA)**
   - Visualized feature correlations  
   - Analyzed depression distribution by age, gender, job type, etc.  

3. **Feature Selection**
   - Used feature importance metrics from tree-based models  

## 💻 Demo

The project demo is built using **Streamlit** for an interactive user experience
