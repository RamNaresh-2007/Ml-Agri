# 🌾 AgriYield AI — Precision Agricultural ML Intelligence Suite

An end-to-end Machine Learning and Agricultural Intelligence platform structured according to a modular 6-stage engineering workflow.

---

## 📁 Repository Structure

```
AgriculturePredict/
│
├── 01_Datasets/                        # Raw, sampled, and processed datasets
│   ├── crop_yield.csv                  # Full raw agricultural dataset (19,689 records)
│   ├── crop_yield_sample.csv           # 100-record sample for quick prototyping
│   └── processed_crop_yield.csv        # Cleaned dataset (target leakage removed, nulls dropped)
│
├── 02_Data_Preprocessing/              # Preprocessing scripts and feature engineering
│   ├── clean_and_preprocess.py         # Leakage prevention & IQR outlier detection
│   ├── categorical.py                  # High-cardinality categorical inspection
│   ├── categorical_encoding.py         # OneHotEncoder serialization
│   └── numerical_scaling.py            # StandardScaler & MinMaxScaler benchmarks
│
├── 03_Exploratory_Data_Analysis/       # Exploratory analysis and visualizations
│   ├── eda.py                          # Generates distributions, boxplots, heatmaps
│   ├── plots/                          # Generated PNG statistical plots
│   └── EDA_Outputs/                    # Correlation matrix CSV and summary charts
│
├── 04_Models_and_Pipelines/            # Machine learning model architectures
│   ├── linear_regression.py            # OLS & Ridge CV regression benchmarks
│   ├── decision_trees.py               # DecisionTree hyperparameter depth pruning
│   ├── random_forest.py                # Champion Random Forest regressor
│   ├── clustering_models.py            # K-Means, Agglomerative, PCA & t-SNE
│   └── complete_pipeline.py            # Master end-to-end ML training pipeline
│
├── 05_Web_Application/                 # Production-grade web interface & APIs
│   ├── app.py                          # Flask application server & REST API
│   ├── dashboard.py                    # Interactive Streamlit intelligence dashboard
│   ├── config.py                       # Application constants & paths
│   ├── templates/                      # Jinja2 HTML templates
│   │   └── index.html                  # Responsive SaaS UI with multi-theme switcher
│   └── static/
│       ├── css/style.css               # Clean & modern theme styling
│       └── js/main.js                  # Dynamic Chart.js visualizations & AJAX
│
├── 06_Outputs_and_Utils/               # Serialized artifacts, metrics & utility functions
│   ├── Utils.py                        # Common helper utilities
│   ├── Model_Outputs/
│   │   ├── Regression/                 # Serialized model.joblib & comparison benchmarks
│   │   └── Clustering/                 # Serialized KMeans cluster models
│   ├── processed_data/                 # Preprocessor.joblib, train/test split CSVs
│   └── outputs/                        # Project metrics JSON & evaluation charts
│
├── app.py                              # Main application launcher (CLI & auto-browser)
├── organize.py                         # Structure verification script
├── test_dashboard.py                   # Automated verification test suite
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start

### 1. Launch Web Dashboard (Default)
Run from workspace root or inside `AgriculturePredict/`:
```bash
python app.py
```
> Opens `http://127.0.0.1:5000` automatically in your browser with interactive predictions, real-time analytics, and visual reports.

### 2. Launch Streamlit Intelligence Suite
```bash
python app.py --streamlit
```

### 3. Run Automated Tests
```bash
python test_dashboard.py
```
