# Smart Retail Demand Forecasting System

An end-to-end Machine Learning and Time Series forecasting application designed to predict product demand, analyze sales patterns, and optimize inventory replenishment for retail operations.

---

## 📁 Project Structure

```text
smart-retail-demand-forecasting/
├── app.py              # Streamlit web application entry point
├── requirements.txt    # Project Python dependencies
├── README.md           # Project documentation and guide
├── .gitignore          # Git ignore rules for data, models, and environments
├── data/               # Raw, processed, and sample datasets
├── models/             # Trained models, encoders, and scaler weights
├── src/                # Reusable source code (data processing, feature engineering, models)
├── assets/             # Static assets (images, logos, charts)
└── notebooks/          # Jupyter notebooks for exploratory data analysis (EDA) and experimentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd smart-retail-demand-forecasting
```

### 2. Create and activate a virtual environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

---

## 📌 Project Roadmap
- [x] Initial project scaffolding & welcome dashboard
- [ ] Exploratory Data Analysis (EDA) on retail sales and seasonality
- [ ] Feature engineering (lag features, rolling statistics, calendar indicators)
- [ ] Baseline & ML forecasting models (LightGBM, XGBoost, Statsmodels)
- [ ] Interactive Streamlit forecasting and inventory planning dashboard
