# Customer Churn Prediction API on AWS

A production-style machine learning API that predicts customer churn and is deployed on AWS using EC2, Docker, S3, IAM, and CloudWatch.

The project demonstrates an end-to-end ML deployment workflow — from model training and evaluation to containerized API deployment, secure model retrieval from Amazon S3, and cloud monitoring.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their service.

In this project, a machine learning model is trained on the Telco Customer Churn dataset and exposed through a REST API using FastAPI.

The trained model is:

- Stored privately in Amazon S3
- Retrieved securely by an EC2 instance using an IAM Role
- Served through a Dockerized FastAPI application
- Monitored using Amazon CloudWatch

### Key objective

Build a simple but complete cloud-based ML system that demonstrates:

**Machine Learning → API → Docker → AWS → Secure Model Storage → Monitoring**

---

## 🏗️ Architecture

```text
                    ┌─────────────────────────┐
                    │       Amazon S3         │
                    │                         │
                    │  churn_model.pkl        │
                    │  Private Bucket         │
                    └────────────┬────────────┘
                                 │
                         IAM Role Access
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        Amazon EC2       │
                    │                         │
                    │      Docker Container   │
                    │       ┌─────────────┐   │
                    │       │   FastAPI   │   │
                    │       │   /predict  │   │
                    │       └──────┬──────┘   │
                    │              │          │
                    └──────────────┼──────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │   CloudWatch     │
                         │                  │
                         │ Metrics + Logs   │
                         └──────────────────┘
````

### Request flow

```text
Client
  │
  ▼
FastAPI /predict
  │
  ▼
ML Pipeline
  │
  ▼
Churn Prediction
  │
  └──→ Probability of Churn
```

---

## 🧠 Machine Learning Pipeline

The model is trained using the Telco Customer Churn dataset.

### Data preprocessing

The pipeline includes:

* Removal of `customerID`
* Conversion of `TotalCharges` to numeric
* Missing-value handling
* Encoding of categorical features
* Standardization of numerical features
* Stratified train/test split

### Features

Numerical features:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Categorical features include:

* Gender
* Partner
* Dependents
* Phone Service
* Internet Service
* Contract
* Payment Method
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Paperless Billing
* Multiple Lines

### Models evaluated

The training pipeline compares:

* Logistic Regression
* Random Forest
* Gradient Boosting

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The best-performing model based on ROC-AUC is saved as:

```text
models/churn_model.pkl
```

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

### API

* FastAPI
* Pydantic
* Uvicorn

### Containerization

* Docker

### AWS

* Amazon EC2
* Amazon S3
* AWS IAM
* Amazon CloudWatch
* Boto3

---

## 📁 Project Structure

```text
aws-ml-churn-prediction/
│
├── app/
│   └── main.py
│
├── data/
│   └── churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   └── train.py
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

> The production deployment retrieves the model from Amazon S3 rather than relying on a model baked into the Docker image.

---

# ☁️ AWS Implementation

## 1. Amazon S3 — Model Storage

The trained model is stored in a private S3 bucket.

```text
s3://shreya-aws-churn-model-2026/
└── churn-model/
    └── churn_model.pkl
```

The bucket is not publicly accessible.

The FastAPI application downloads the model from S3 when the container starts if the model is not already present locally.

---

## 2. IAM Role — Secure AWS Access

An IAM Role named:

```text
EC2-Churn-S3-Role
```

is attached to the EC2 instance.

The role allows the application running on EC2 to access the S3 model without storing AWS access keys inside the application.

The application uses:

```python
boto3.client("s3")
```

to communicate with Amazon S3.

### Why IAM Roles?

Instead of doing this:

```text
AWS Access Key
AWS Secret Key
        ↓
Application
```

the architecture uses:

```text
EC2
 ↓
IAM Role
 ↓
Temporary AWS credentials
 ↓
S3
```

This avoids hardcoding long-lived credentials in application code.

---

## 3. Amazon EC2 — API Hosting

The FastAPI application is deployed on an EC2 instance running Amazon Linux.

The application is packaged as a Docker container.

```bash
docker build -t churn-api .
```

The container is then started with:

```bash
docker run -d -p 8000:8000 --name churn-api churn-api
```

The API is exposed on port `8000`.

---

## 4. Docker

Docker provides a consistent environment for running the application.

The Docker image contains:

* Python runtime
* Required dependencies
* FastAPI application
* Application configuration

The ML model itself is retrieved from S3 at runtime.

This keeps the model artifact separate from the application image.

---

## 5. Amazon CloudWatch

CloudWatch is used to monitor the EC2 deployment.

### Infrastructure metrics

The CloudWatch Agent collects:

* CPU utilization
* Memory utilization
* Disk utilization

### Application logs

Docker/FastAPI logs are also sent to CloudWatch.

Example log group:

```text
/aws/ec2/churn-api
```

This allows application behavior and infrastructure health to be monitored from AWS.

---

# 🔌 API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## API Home

```http
GET /
```

Response:

```json
{
  "message": "Customer Churn Prediction API",
  "status": "running"
}
```

---

## Churn Prediction

```http
POST /predict
```

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 846.0
}
```

Example response:

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.7342
}
```

---

# 🖥️ Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/Shreyabhat11/aws-ml-churn-prediction.git

cd aws-ml-churn-prediction
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the model

```bash
python src/train.py
```

The trained model will be generated at:

```text
models/churn_model.pkl
```

## 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

FastAPI's Swagger UI can be used to test the prediction endpoint.

---

# 🐳 Running with Docker

Build the image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -d \
  -p 8000:8000 \
  --name churn-api \
  churn-api
```

Check the container:

```bash
docker ps
```

View logs:

```bash
docker logs churn-api
```

Open:

```text
http://localhost:8000/docs
```

---

# 🔐 Security Considerations

This project follows several basic cloud security practices:

* S3 bucket remains private
* EC2 accesses S3 through an IAM Role
* AWS credentials are not hardcoded in the application
* `.env` and credential files are excluded through `.gitignore`
* EC2 SSH access is restricted to the configured source IP
* API access is controlled through the EC2 security group

### Production improvement

The current IAM role uses AWS-managed S3 read permissions for simplicity.

For a production deployment, permissions should be restricted further to only the required S3 bucket/object.

---

# 📊 Monitoring

CloudWatch provides visibility into both infrastructure and application behavior.

### Metrics

```text
CPU
Memory
Disk
```

### Logs

```text
FastAPI / Uvicorn logs
Docker container logs
Application errors
```

This enables basic troubleshooting without logging directly into the EC2 server.

---

# 🎯 Key Learning Outcomes

This project helped build practical understanding of:

### Machine Learning

* End-to-end preprocessing pipelines
* Model comparison
* Classification metrics
* Model serialization

### Backend

* REST API development
* Request validation
* Model inference through FastAPI

### Docker

* Containerizing ML applications
* Docker image creation
* Container execution
* Application logging

### AWS

* EC2 deployment
* S3 model storage
* IAM users and roles
* IAM-based authentication
* CloudWatch monitoring
* CloudWatch logs
* AWS CLI
* Boto3

### Cloud ML Architecture

* Separating model artifacts from application code
* Secure access between AWS services
* Monitoring deployed ML applications

---

# 🔮 Future Improvements

Possible extensions include:

* [ ] Deploy the model using Amazon SageMaker
* [ ] Add automated CI/CD with GitHub Actions
* [ ] Add model versioning in S3
* [ ] Add prediction request logging
* [ ] Add CloudWatch alarms
* [ ] Add an API authentication layer
* [ ] Add model performance monitoring
* [ ] Add automated model retraining
* [ ] Add a simple frontend for predictions

---

# 👩‍💻 Author

**Shreya Bhat**

AI/ML & Generative AI Developer

GitHub:
[https://github.com/Shreyabhat11](https://github.com/Shreyabhat11)

````