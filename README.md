# Real-Time Credit Card Fraud Detection System

A production-ready machine learning system that detects fraudulent credit card transactions in real time using XGBoost, Deep Learning, and AWS cloud infrastructure.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)](https://xgboost.readthedocs.io/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20Lambda%20S3-yellow)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

## Overview

This project implements an end-to-end fraud detection pipeline that identifies fraudulent credit card transactions with 98.7% ROC-AUC. The system processes 284,807 transactions, handles severe class imbalance using SMOTE, and benchmarks four machine learning models including a TensorFlow Neural Network. The best model is deployed on AWS SageMaker for real-time inference, with Power BI dashboards monitoring fraud trends and model performance.

## Results

| Model | ROC-AUC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.9727 | 0.0547 | 0.9184 | 0.1033 |
| Random Forest | 0.9848 | 0.4095 | 0.8776 | 0.5584 |
| XGBoost | 0.9712 | 0.8830 | 0.8469 | 0.8646 |
| TensorFlow Neural Network | 0.9872 | 0.6357 | 0.8367 | 0.7225 |

XGBoost was selected for production deployment due to the highest F1 score and lowest false positive rate.

## Architecture
```
Dataset (284,807 transactions)
↓
Preprocessing & Feature Engineering
↓
SMOTE Class Balancing
↓
Model Training (LR, RF, XGBoost, Neural Network)
↓
Evaluation & Comparison
↓
Best Model Deployment
↓
FastAPI → AWS SageMaker → Lambda → S3
↓
Power BI Monitoring Dashboard
```

## Tech Stack

- **Languages:** Python 3.11
- **ML Frameworks:** Scikit-learn, XGBoost, TensorFlow, Keras
- **Cloud:** AWS SageMaker, AWS Lambda, Amazon S3, IAM
- **API:** FastAPI, Uvicorn, Pydantic
- **Visualization:** Power BI, Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy, imbalanced-learn

## Features

- Real-time fraud detection with sub-second latency
- Four-model comparison framework
- Deep Learning (TensorFlow) and Gradient Boosting (XGBoost) implementations
- SMOTE-based class imbalance handling
- AWS serverless deployment for scalability
- Power BI dashboard with 13 DAX measures
- Model drift monitoring

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (V1-V28 PCA features + Amount + Time)
- **Target:** Binary (Fraud / Normal)
- **Class Distribution:** 492 fraud (0.17%) vs 284,315 normal

## Project Structure
```
fraud-detection/
│
├── 📁 notebooks/
│   └── Fraud Detection.ipynb        # Complete ML pipeline
│
├── 📁 src/
│   └── api.py                       # FastAPI service
│
├── 📁 output/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── fraud_by_hour.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision_recall_curve.png
│   ├── model_comparison.csv
│   ├── model_comparison_chart.png
│   └── confusion_matrix_comparison.png
│
├── 📁 dashboard/
│   └── fraud_dashboard.pbix         # Power BI dashboard
│
├── 📁 screenshots/
│  ├── IAM_roles.png
│   ├── Jupyter_predictions.png
│   ├── Lambda_function.png
│   ├── S3_bucket_files.png
│   ├── SageMaker_model.png
│   └── PowerBI_page1-4.png
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup and Installation

```bash
git clone https://github.com/navyayalavarthi/Real-Time-Fraud-Detection.git
cd Real-Time-Fraud-Detection

python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/ folder

# Run notebook
jupyter notebook "notebooks/Fraud Detection.ipynb"

# Start API
uvicorn src.api:app --reload --port 8001
```

## Skills Used

Python · XGBoost · TensorFlow · Keras · Deep Learning · Scikit-learn · Random Forest · Logistic Regression · SMOTE · Feature Engineering · Model Comparison · AWS SageMaker · AWS Lambda · Amazon S3 · IAM · FastAPI · Power BI · DAX · Pandas · NumPy · Matplotlib · Seaborn · Jupyter

## Key Learnings

- Class imbalance handling significantly impacts model performance
- Deep Learning (Neural Network) achieved highest ROC-AUC
- XGBoost provided best balance of precision and recall for production
- SMOTE balancing improved minority class detection
- AWS serverless deployment enables real-time fraud scoring at scale

## Author

Navya Yalavarthi
📧 navya.yalavarthi1@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/navya-yalavarthi-b21297289/
🐙 GitHub: https://github.com/navyayalavarthi

## License

MIT License — for educational and portfolio purposes.
Credit card dataset from Kaggle ULB Machine Learning Group.


