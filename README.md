# Financial Crisis Prediction System

This project is a comprehensive, machine learning-based early warning system designed to predict financial crises using macroeconomic and financial indicators. The foundation of the project is based on the research by **Chris Reimann**, extended with additional data sources, new model architectures, and a modular data pipeline.

---

## 📊 Project Purpose

The aim is to build an **interpretable**, robust predictive system that:

1. **Forecasts** systemic financial crises at a country level.  
2. **Integrates** real-world economic data from multiple global sources.  
3. **Compares** classical ML models with advanced methods (XGBoost, LSTM).  
4. Supports **interpretability** with ALE plots and robustness testing.

---

## 🔗 Key Features

### ✈️ Data Pipeline
- **World Bank Data Ingestion**: Automatic ingestion of macroeconomic indicators from 1960 to 2023.  
- **IMF WEO Dataset Integration**: Forecast-oriented data (2024–2028) such as GDP, inflation, debt.  
- **Crisis Label Sources**:  
  - JST Macrohistory Database  
  - Laeven & Valencia (IMF banking crises)  
  - ESRB (European Systemic Risk Board)  

### 📊 Models Supported
- **Logistic Regression (Logit)**  
- **Random Forest**  
- **Extra Trees**  
- **Support Vector Machine (SVM)**  
- **Neural Networks (MLP)**  
- **K-Nearest Neighbors (KNN)**  
- **XGBoost** ✨ *(new)*  
- **LSTM** ✨ *(new, with time-series reshaping)*

### 📈 Experiment Modes
- **In-Sample**: Full dataset fitting.  
- **Cross-Validation**: Repeated temporal group validation.  
- **Forecast**: True out-of-sample performance starting from 1980.

---

## 🔍 Project Structure

financialCrisisML/ ├── code/ │ ├── data_pipeline/ │ │ ├── ingest_worldbank.py │ │ ├── ingest_imf.py │ │ └── prepareData.py │ ├── doExperiment.py │ ├── model_new_perform.py │ ├── new_models_experiment.py │ └── utils.py ├── visualizations/ ├── logs/ └── README.md


---

## 🌐 Data Sources

- **World Bank API** (via `wbdata` and manual fetches)  
- **IMF WEO Forecast Dataset** (April 2024 release)  
- **JSTdatasetR6.xlsx** (historical macrofinancial data)

---

## ✨ Recent Contributions

This fork introduces several key enhancements:

- **✅ Refactored** `worldbank_ingest.py` to include 15 countries and indicators from 1960–2023.  
- **✅ Integrated** IMF WEO data, covering up to 2028 forecasts.  
- **✅ Introduced** XGBoost and LSTM support in the pipeline.  
- **✅ Added** robustness testing with ESRB and Laeven & Valencia labels.  
- **✅ Improved** visualizations: ROC curves, ALE plots, AUC tables.

---

## 📘 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run World Bank ingestion
python code/data_pipeline/woldBankIngest.py

# Run IMF ingestion
python code/data_pipeline/imgIngest.py

# Launch experiment (e.g., XGBoost & LSTM)
python code/model_new_perform.py
