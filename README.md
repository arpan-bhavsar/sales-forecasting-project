# 🛒 DMart Sales Intelligence & Forecasting Dashboard

An interactive, PowerBI-style analytics and predictive application built using **Streamlit** and **Scikit-Learn**. This system forecasts retail sales trends and evaluates machine learning performance using historical DMart transaction data.

🔗 **[Live Dashboard Link](https://sales-forecasting-project-fvp9hdsch3rrng5v8as5bu.streamlit.app/)** 

## 👥 Team Members
* [Arpan Bhavsar](https://github.com/arpan-bhavsar)
* [Kritarth Singh](https://github.com/KritarthSingh19)
* [Janvi Soni](https://github.com/Janvi-Soni214)

---

## 📊 Project Architecture & Features
- **Interactive UI/UX:** Built a dynamic dashboard featuring tracking cards for high-level KPIs, clean evaluation graphs, and side-by-side model comparisons.
- **Predictive Engine:** Predicts exact sales transactions based on real-time parameters (Category, City, Region, Discount, Profit, and Seasonality).
- **Comprehensive Error Breakdown:** Provides transparent regression metric visualization ($MAE$, $RMSE$, $R^2$) to isolate model performance drops during high-variability festive quarters.

---

## 🛠️ Tech Stack & Dependencies
* **Frontend:** Streamlit, Altair 
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformers, OneHotEncoder)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

All core configurations and cloud requirements are tracked dynamically via `requirements.txt`.

---

## 🚀 How to Run Locally

If you want to pull this project down and run it on your machine, follow these steps:

### 1. Clone the Workspace
```bash
git clone [https://github.com/arpan-bhavsar/sales-forecasting-project](https://github.com/arpan-bhavsar/sales-forecasting-project)
cd sales-forecasting-project

pip install -r requirements.txt

streamlit run dashboard.py
