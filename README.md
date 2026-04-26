# 🏆 Amazon Buy Box % Prediction System

### From Data to Actionable E-Commerce Decisions

---

## 🚀 Business Impact

This project transforms raw Amazon Seller data into a **decision-support system** that helps optimize Buy Box performance.

* 📈 Predict Buy Box % for each product (ASIN)
* ⚠️ Identify high-risk products likely to lose Buy Box visibility
* 🧠 Understand key drivers influencing Buy Box ownership
* 🎯 Support pricing, advertising, and conversion optimization strategies

👉 Goal: **Turn machine learning predictions into actionable business decisions**

---

## 🎯 Use Case

This system is designed for:

* 🛒 **E-commerce Managers** → optimize pricing & campaigns
* 📦 **Operations Teams** → manage inventory & product performance
* 📊 **Data Analysts** → understand Buy Box drivers
* 💰 **Marketing Teams** → evaluate advertising impact

### Example

A product predicted with **LOW Buy Box % + HIGH risk**
→ triggers:

* price adjustment
* advertising activation
* listing optimization

---

## 🧠 Project Overview

This project predicts the **Amazon Buy Box win rate (%)** using advanced machine learning models and feature engineering.

It combines:

* Business Reports (core performance data)
* Advertising data (ads impact)
* Time-based features (seasonality & trends)
* Product-level behavioral patterns

---

## 📊 Dashboard Preview

![Dashboard](images/dashboard.png)

---

## ⚙️ Features

* 📊 Interactive Streamlit dashboard
* 🤖 6 machine learning models
* 🧩 Advanced feature engineering (~70+ features)
* 📈 Model comparison (MAE, RMSE, R²)
* 🔮 Buy Box forecasting (7–28 days)
* ⚠️ Risk classification (HIGH / MEDIUM / LOW)
* 📉 Feature importance analysis
* 📦 Multi-source data integration

---

## 🧬 Data Sources

The system integrates multiple Amazon datasets:

* 📁 Business Reports (sessions, sales, Buy Box %)
* 📢 Advertising Data (impressions, clicks, spend)
* 🔍 Search Term Data (optional)

⚠️ **Note:**
Data used in this repository is **synthetic** and transformed for portfolio purposes.

---

## 🧪 Feature Engineering

Key feature groups:

* 📅 Time features (month, quarter, seasonality)
* 📊 Conversion metrics (conversion rate, order rate)
* 📱 User behavior (mobile vs browser ratios)
* 💰 Sales intensity (sales per session/view)
* 🏢 B2B metrics
* 📉 Log transformations
* 🔁 Lag features (1, 7, 14, 30 days)
* 📈 Rolling averages
* 📦 Product-level statistics
* 📢 Advertising impact (CTR, conversion, spend)

👉 Total: **70+ engineered features**

---

## 🤖 Modeling Approach

We compare multiple machine learning models:

| Model             | Type               | Purpose               |
| ----------------- | ------------------ | --------------------- |
| Linear Regression | Baseline           | Benchmark             |
| Ridge             | Regularized        | Reduce overfitting    |
| Random Forest     | Ensemble           | Capture non-linearity |
| Gradient Boosting | Boosting           | Improve residuals     |
| XGBoost           | Advanced boosting  | High performance      |
| LightGBM          | Efficient boosting | Speed & scalability   |

---

## 🧠 Ensemble Strategy

Final predictions are generated using:

👉 **Ensemble (average of multiple models)**

This ensures:

* more stable predictions
* reduced variance
* better generalization

---

## 📈 Model Evaluation

Models are evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

👉 Best model is automatically selected

---

## 🔮 Forecasting & Risk Analysis

The system predicts Buy Box % and classifies products into:

| Risk Level     | Meaning                |
| -------------- | ---------------------- |
| 🔴 HIGH RISK   | Likely to lose Buy Box |
| 🟡 MEDIUM RISK | Moderate stability     |
| 🟢 LOW RISK    | Strong Buy Box control |

👉 Includes confidence intervals (uncertainty estimation)

---

## 🔑 Key Insights

* 📢 Advertising has a **strong impact** on Buy Box ownership
* 📊 Buy Box distribution is often **bimodal (0% or 100%)**
* 📈 Conversion rate strongly correlates with Buy Box %
* 🎯 Mid-range products (30–70%) are the biggest optimization opportunity

---

## 🛠️ Installation

```bash
git clone https://github.com/Mst-KrgZ/amazon-buybox-prediction-system.git
cd amazon-buybox-prediction-system

pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.11
* Streamlit
* Pandas / NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Plotly

---

## 👨‍💻 Author

**Mesut Karagöz**
Data Scientist

🔗 GitHub: https://github.com/Mst-KrgZ
🔗 LinkedIn: https://www.linkedin.com/in/mesut-karagöz-181733260/

---

## ⚡ Final Note

> This project goes beyond prediction —
> it delivers a **data-driven decision system for real-world e-commerce optimization.**
