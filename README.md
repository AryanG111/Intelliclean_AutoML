# 🤖 IntelliClean AutoML Pipeline

**Intelligent Automated Data Preprocessing for Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Overview

IntelliClean AutoML is a powerful, intelligent data preprocessing pipeline that automatically analyzes, cleans, and transforms your datasets for machine learning. It eliminates the tedious manual work of data preprocessing by applying ML-best practices intelligently based on your data's characteristics.

https://intelliclean-automl.streamlit.app/

## ✨ Key Features

### 🔍 Smart Data Characterization
- **Automatic Type Detection**: Identifies numerical, categorical, binary, datetime, and text columns
- **Data Quality Assessment**: Detects missing data patterns, outliers, duplicates, and data leakage
- **Statistical Analysis**: Analyzes distributions, skewness, and correlation patterns

### 🧹 Intelligent Preprocessing
- **Missing Data Handling**: KNN imputation, mean/median/mode strategies based on data patterns
- **Outlier Treatment**: Winsorization, clipping, or preservation based on severity
- **Smart Encoding**: One-hot, label, target, frequency encoding based on cardinality
- **Feature Engineering**: Automatic log transforms, datetime extraction, polynomial features

### 📊 ML-Ready Output
- **Clean Datasets**: Ready for immediate model training
- **Multiple Formats**: CSV, Excel, JSON, Parquet export options
- **Comprehensive Reports**: Detailed preprocessing documentation
- **Train/Test Splits**: Automatic stratified splitting for supervised learning

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intelliclean-automl.git
cd intelliclean-automl
```
### Basic Usage

    Run the Streamlit web app
```
bash

streamlit run app.py
```
    Upload your dataset through the web interface

    Configure processing options (target column, advanced settings)

    Click "Start Preprocessing" and download your cleaned data

### Programmatic Usage
```python

from intelliclean_automl import complete_preprocessing_pipeline

# Process a dataset
results = complete_preprocessing_pipeline('your_data.csv', target_column='price')

# Get processed data
cleaned_data = results['processed_data']
cleaned_data.to_csv('cleaned_data.csv', index=False)
```


## 📁 Project Structure
```text

intelliclean-automl/
├── app.py                 # Streamlit web application
├── intelliclean_automl.py # Core preprocessing engine
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── examples/             # Example datasets and notebooks
```


## 🔧 Core Components
**1. Data Characterization Engine**

    File Type Detection: Automatic recognition of CSV, Excel, JSON, Parquet, Feather

    Column Type Analysis: Intelligent detection of data types and patterns

    Quality Assessment: Missing data analysis, outlier detection, correlation analysis

**2. Adaptive Preprocessing Router**

    Strategy Selection: Chooses optimal preprocessing methods based on data characteristics

    ML-Best Practices: Implements industry-standard preprocessing techniques

    Customizable: Easy to extend with new preprocessing strategies

**3. Enhanced Processing Pipeline**

    Numerical Data: Scaling, outlier handling, distribution-based transformations

    Categorical Data: Smart encoding, cardinality-aware strategies

    Feature Engineering: Automatic feature creation and selection


## 🎨 Web Interface Features
### 📊 Interactive Dashboard

    Data Preview: Real-time dataset exploration

    Visual Analytics: Missing data heatmaps, distribution plots

    Quality Metrics: Data quality scoring and improvement tracking

### ⚙️ Configuration Panel

    Target Selection: Specify prediction target for supervised learning

    Advanced Options: Tune preprocessing strategies

    Real-time Preview: See changes before applying

### 📋 Results Management

    Comparison Tools: Before/after data comparison

    Export Options: Multiple format downloads

    Detailed Reports: Comprehensive preprocessing documentation


##📊 Supported Data Types

File Formats

    ✅ CSV/TSV/TXT

    ✅ Excel (.xlsx, .xls)

    ✅ JSON/JSONL

    ✅ Parquet

    ✅ Feather

Data Types

    Numerical: Integers, floats, continuous variables

    Categorical: Strings, limited unique values

    High Cardinality: Many unique categories

    Binary: Yes/No, True/False, 0/1

    Datetime: Dates, timestamps, time series

    Text: Long strings, descriptions, comments


## 🛠️ Technical Requirements

Python Version:
    Python 3.8 or higher

Core Dependencies
```txt

streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
plotly>=5.0.0
category-encoders>=2.6.0
scipy>=1.10.0
```
## 📈 Performance Benchmarks
```table
Dataset Size	Processing Time	Memory Usage
10,000 rows	~5-10 seconds	~500 MB
100,000 rows	~30-60 seconds	~2 GB
1M+ rows	~2-5 minutes	~8 GB
```
Note: Performance varies based on dataset complexity and hardware


## 🔄 Workflow Integration
**Standalone Application**
```bash

# Process single file
python -m intelliclean_automl process data.csv --target sales

# Batch processing
python -m intelliclean_automl batch-process data_folder/ --output cleaned_data/
```
**Python Library**
```python

# Integration with existing ML pipelines
from intelliclean_automl import preprocess_for_ml

# Get ML-ready data
ml_data = preprocess_for_ml('data.csv', target_column='target')
X_train, X_test = ml_data['X_train'], ml_data['X_test']
```
##🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.
Development Setup
```bash

# Fork and clone the repository
git clone https://github.com/yourusername/intelliclean-automl.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

**Areas for Contribution**

    New preprocessing strategies

    Additional file format support

    Performance optimization

    Enhanced visualization

    Documentation improvements
    
    
**IntelliClean AutoML** - Making data preprocessing intelligent, automatic, and accessible for everyone! 🚀

⭐ Star this repo if you find it useful!
