# Solving the Problem of Jaya Jaya Maju Company

## Business Understanding

**Jaya Jaya Maju** is an international-scale edutech company with over 1,000 employees. Despite rapid growth, the company is currently facing a serious challenge: an **employee attrition rate exceeding 10%**.

This project aims to **identify the factors causing attrition**, build a **machine learning-based prediction model**, and present an **interactive dashboard** to assist the HR division in making informed and proactive decisions.

### Business Problem

* Identify the main factors affecting the company's attrition rate.
* Determine behavioral patterns of employees likely to resign.
* Develop a business dashboard as an employee monitoring tool to support HR decision-making.

### Project Scope

* Identify variables that influence employee attrition.
* Build a machine learning model to predict which employees are likely to resign.
* Provide interactive visualization dashboards using Looker Studio.
* Offer recommendations based on insights from the business dashboard to help guide company decisions.

### Preparation

**Data source:** [Employee Dataset - Dicoding GitHub](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

**Project structure:**

```
.
├── datasets/
│   ├── clean_data.csv
│   └── employee_data.csv
├── models/
│   ├── rf_model.pkl
│   └── xgb_model.pkl
├── README.md
├── nurrizkyarumjatmiko-dashboard.png
├── notebook.ipynb
├── prediction.py
└── requirements.txt
```

```bash
# Set up Python environment
python -m venv .env

# Activate environment (Linux/Mac)
source .env/bin/activate      

# Activate environment (Windows)
.env\Scripts\activate       
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the model from the prediction script:

```bash
python prediction.py
```

## Business Dashboard

Link: [Business Dashboard](https://lookerstudio.google.com/reporting/5cfdfc59-2481-4de8-a6ed-21eb2e9715c1)

**Description:**
This business dashboard was developed using **Looker Studio** as an interactive data visualization platform. Its main objectives are:

* To monitor key factors contributing to attrition, such as:

  * JobLevel
  * OverTime
  * StockOptionLevel
  * MaritalStatus
  * YearsAtCompany
* To provide clear insights enabling the HR team to make quicker and data-driven decisions.

## Conclusion

From the exploration and modeling results, the following findings were obtained:

The business dashboard was built based on **feature importance** extracted from the **XGBoost model**, identifying several attributes with the most influence, such as **JobLevel**, **OverTime**, **StockOptionLevel**, **MaritalStatus**, **YearsAtCompany**, **Age**, **TotalWorkingYears**, **JobInvolvement**, **EnvironmentSatisfaction**, and **MonthlyIncome**.

* Employees with high workloads such as overtime, higher job levels, low income, and dissatisfaction with their work environment tend to have a higher chance of resigning, as these factors significantly impact attrition.
* The best-performing model is the **XGBoost Classifier**, with an accuracy of 85%.

### Recommended Action Items

Some recommendations for management to reduce the attrition rate:

* Reduce unnecessary overtime.
* Offer additional incentives for employees with high workloads.
* Support **work-life balance** policies through flexible work arrangements.
* Create a supportive and appreciative work environment.
* Provide clear career development paths and training opportunities.
