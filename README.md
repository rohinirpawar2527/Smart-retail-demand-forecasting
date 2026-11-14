# Smart Retail Demand Forecasting System

An end-to-end Machine Learning and Time Series Forecasting application built to analyze retail sales patterns, forecast product demand, and support inventory planning.

**Live Demo:** https://rohinirpawar2527-smart-retail-demand-forecasting-app-g1yp7a.streamlit.app/

**GitHub Repository:** https://github.com/rohinirpawar2527/Smart-retail-demand-forecasting

## Project Overview

The Smart Retail Demand Forecasting System is an interactive dashboard designed to help understand retail demand trends and make data-driven inventory decisions.

The dashboard works with retail data covering 2024–2025, representing 5 stores, 20 products, and 73,100 daily records.

## Key Features

- **Overview Dashboard:** View total sales, revenue, daily demand and demand trends.
- **Data Explorer:** Explore and analyze retail datasets.
- **Demand Forecasting:** Generate demand predictions using machine learning.
- **Product Analytics:** Analyze product-level sales and performance.
- **Inventory Insights:** Explore demand information to support inventory planning.
- **Model Performance:** Review forecasting model performance.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Statsmodels
- Plotly
- Joblib

## Project Structure

```text
smart-retail-demand-forecasting/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
├── data/
├── models/
├── notebooks/
├── src/
└── tests/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rohinirpawar2527/Smart-retail-demand-forecasting.git
```

### 2. Open the project folder

```bash
cd Smart-retail-demand-forecasting
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:**  
https://rohinirpawar2527-smart-retail-demand-forecasting-app-g1yp7a.streamlit.app/

## Project Objective

To apply machine learning and data analysis techniques to retail demand forecasting and provide an interactive platform for exploring sales trends, product performance, and inventory-related insights.

## Author

**Rohini Pawar**

GitHub: https://github.com/rohinirpawar2527

---

If you find this project useful, feel free to explore the repository and the live application.
