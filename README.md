readme = """# Real-Time Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Lambda%20%7C%20S3-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-teal?style=flat-square&logo=fastapi)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)

## Project Overview

Designed and deployed an end-to-end fraud detection pipeline to identify potentially fraudulent credit card transactions in real time. Built machine learning models using XGBoost and developed cloud-based workflows for scalable model inference and monitoring.

## Key Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.98+ |
| Recall (Fraud) | 85.71% |
| Precision (Fraud) | 77.06% |
| F1 Score | 81.16% |
| False Positives | 25 |
| False Negatives | 14 |
| Total Transactions | 284,807 |
| Fraud Cases Detected | 84 out of 98 |

## Project Architecture
```
creditcard.csv
↓
Data Preprocessing & Feature Engineering
↓
SMOTE Balancing → XGBoost Training
↓
Model Evaluation (ROC-AUC, Precision, Recall)
↓
FastAPI Inference Service
↓
AWS S3 (Model Storage)
↓
AWS SageMaker (Model Endpoint)
↓
AWS Lambda (Inference Trigger)
↓
Power BI Dashboard (Monitoring)
```
## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%)
- **Features:** V1-V28 (PCA transformed), Amount, Time

## Features Engineered

| Feature | Description |
|---------|-------------|
| Amount_scaled | StandardScaler applied to Amount |
| Time_scaled | StandardScaler applied to Time |
| hour | Hour of day extracted from Time (0-23) |
| log_amount | log1p transformation of Amount |

## Model Details

- **Algorithm:** XGBoost Classifier
- **Class Imbalance:** Handled using SMOTE
- **Trees:** 300 with early stopping
- **Key Parameters:** max_depth=6, learning_rate=0.1, scale_pos_weight=577

## AWS Infrastructure

| Service | Resource | Purpose |
|---------|----------|---------|
| Amazon S3 | fraud-detection-navya | Model and prediction storage |
| AWS SageMaker | fraud-detection-model-v2 | Model hosting and inference |
| AWS Lambda | fraud-detection-lambda | Automated inference trigger |
| IAM | FraudDetectionSageMakerRole | SageMaker permissions |
| IAM | FraudDetectionLambdaRole | Lambda permissions |

## Real-Time Prediction Example

```json
Input Transaction:
{
  "V1": -2.31, "V2": 1.95, ... "V28": -0.14,
  "Amount": 0.00,
  "Time": 406.0
}

Output:
{
  "fraud_probability": 0.9998,
  "is_fraud": true,
  "risk_level": "HIGH",
  "recommended_action": "Block transaction immediately",
  "timestamp": "2026-06-20T22:13:33"
}
```

## Power BI Dashboard

4-page monitoring dashboard built in Power BI:

| Page | Description |
|------|-------------|
| Fraud Trends | Fraud count by hour, risk level distribution, revenue at risk |
| Model Performance | Confusion matrix, Precision, Recall, F1, ROC-AUC gauge |
| False Positives | Transaction-level false positive analysis by hour |
| Model Drift | Weekly AUC trend, fraud volume over time, drift alert |

## Project Structure
```
fraud-detection/

├── notebooks/
│   └── Fraud Detection.ipynb    # Complete ML pipeline
├── src/
│   ├── inference.py             # SageMaker inference script
│   └── lambda_function.py       # AWS Lambda function
├── output/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── fraud_by_hour.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── precision_recall_curve.png  
├── screenshots/
│   ├── S3_bucket_files.png
│   ├── SageMaker_model.png
│   ├── Lambda_function.png
│   ├── IAM_roles.png
│   └── Jupyter_predictions.png
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/navyayalavarthi/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Download creditcard.csv and place in data/ folder

# Run the notebook
jupyter notebook notebooks/Fraud\\ Detection.ipynb
```

## AWS Configuration

```bash
# Configure AWS credentials
aws configure

# Required services
# - Amazon S3 (model storage)
# - AWS SageMaker (model endpoint)
# - AWS Lambda (inference trigger)
# - IAM roles (permissions)
```

## FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Real-time fraud scoring |
| /health | GET | API health check |
| /docs | GET | Interactive API documentation |

## Skills Used

Python · XGBoost · Scikit-learn · AWS SageMaker · AWS Lambda · Amazon S3 · FastAPI · Power BI · SQL · Jupyter

## Author

Navya 
📧 navya.yalavarthi1@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/navya-yalavarthi-b21297289/

