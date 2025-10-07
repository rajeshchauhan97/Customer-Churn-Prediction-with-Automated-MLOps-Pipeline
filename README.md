# Customer-Churn-Prediction-with-Automated-MLOps-Pipeline
This project demonstrates a complete, end-to-end MLOps pipeline for predicting customer churn. The system is designed to be production-ready and can be adapted for various industries, including Telecom, Banking, and E-commerce. The core functionality includes:

  * **Predictive Model:** An XGBoost model trained to identify customers at risk of churning.
  * **Automated Pipeline:** A CI/CD workflow (using GitHub Actions) that automatically trains, validates, and deploys the model.
  * **Scalable Infrastructure:** Deployment of the model as a real-time API endpoint on AWS SageMaker.
  * **Consistent Preprocessing:** A robust `scikit-learn` pipeline that ensures the same data transformations are applied during both training and inference, preventing model drift.

-----

### Key Features and MLOps Workflow

The project follows a modular and automated MLOps lifecycle:

1.  **Data Ingestion & Preprocessing:**

      * **Tool:** `pandas`, `scikit-learn`
      * **Process:** Raw data is loaded, cleaned, and a preprocessing pipeline is defined to handle feature engineering (e.g., one-hot encoding, scaling, imputation).

2.  **Model Training & Tracking:**

      * **Tool:** `XGBoost`, `scikit-learn`, `MLflow`
      * **Process:** The preprocessing pipeline and the XGBoost model are combined into a single `scikit-learn` pipeline. This entire pipeline is trained and the results (metrics, parameters, model artifact) are logged to an MLflow tracking server backed by AWS S3. The final model is registered in the MLflow Model Registry.

3.  **Containerization:**

      * **Tool:** `Docker`
      * **Process:** The FastAPI application and its dependencies are packaged into a lightweight Docker container. This ensures the environment is reproducible and portable.

4.  **Model Serving API:**

      * **Tool:** `FastAPI`, `MLflow`
      * **Process:** A FastAPI application is created to serve the model. It loads the latest version of the model pipeline directly from the MLflow Model Registry (via S3) and exposes a `/predict` endpoint for real-time inference.

5.  **CI/CD Automation:**

      * **Tool:** `GitHub Actions`, `AWS ECR`, `AWS SageMaker`
      * **Process:** Pushing code to the `main` branch triggers an automated workflow that:
          * Installs dependencies and runs the training script.
          * Authenticates with AWS.
          * Builds the Docker image and pushes it to Amazon Elastic Container Registry (ECR).
          * Deploys the containerized model to a real-time inference endpoint on AWS SageMaker.

-----

### Project Structure

```
customer-churn-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions workflow
├── data/
│   └── raw/
│       └── customer_churn.csv      # Telco customer churn dataset
├── deployment/
│   ├── Dockerfile                # Dockerfile for the FastAPI service
│   ├── app.py                    # FastAPI application
│   └── requirements.txt          # Dependencies for the deployment container
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Preprocessing logic
│   └── model_training.py         # Training and MLflow logging logic
├── README.md                     # Project documentation
├── requirements.txt              # Dev and training dependencies
└── settings.ini                  # AWS, MLflow configurations
```

-----

### Setup and Local Execution

#### Prerequisites

  * Python 3.10 installed on your system.
  * Git and a GitHub account.
  * Docker installed and running.
  * An AWS account with an IAM user configured with programmatic access and permissions for S3, ECR, and SageMaker.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-mlops.git
cd customer-churn-mlops
```

#### Step 2: Prepare the Data

Download the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the `Telco-Customer-Churn.csv` file inside the `data/raw/` directory, renaming it to `customer_churn.csv`.

#### Step 3: Local Environment Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**

      * **Windows:** `.\venv\Scripts\activate`
      * **macOS/Linux:** `source venv/bin/activate`

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

#### Step 4: Run Local Training with MLflow

To simulate the CI/CD pipeline's training step, you can run the training script locally. This will create a local `mlruns/` directory to store artifacts.

```bash
python src/model_training.py
```

To view the logged experiments, launch the MLflow UI:

```bash
mlflow ui
```

You can access the UI at `http://127.0.0.1:5000`.

-----

### API Endpoints

The deployed FastAPI application exposes a simple API for real-time inference.

  * **Base URL:** `http://<your-sagemaker-endpoint-url>`

  * **Endpoint:** `/predict`

  * **Method:** `POST`

  * **Request Body (JSON):**

    ```json
    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "Yes",
      "tenure": 24,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "Yes",
      "TechSupport": "Yes",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 85.05,
      "TotalCharges": 1949.4
    }
    ```

  * **Response:**

    ```json
    {
      "prediction": 0
    }
    ```

    (Where `0` indicates no churn and `1` indicates churn).

-----

### CI/CD Deployment with GitHub Actions

The `ci-cd.yml` workflow automates the deployment process. To run it, you must configure your GitHub repository with the necessary AWS secrets and environment variables.

1.  **Set up AWS IAM:** Create an IAM user with `AmazonS3FullAccess`, `AmazonEC2ContainerRegistryFullAccess`, and `AmazonSageMakerFullAccess`.
2.  **Set GitHub Secrets:** Go to your repository's **Settings \> Secrets and variables \> Actions** and add the following secrets:
      * `AWS_ACCESS_KEY_ID`
      * `AWS_SECRET_ACCESS_KEY`
3.  **Create AWS Resources:**
      * An S3 bucket for MLflow artifacts.
      * An ECR repository to store your Docker image.
      * **Update `ci-cd.yml` with your bucket and repository names.**
4.  **Trigger the Workflow:** Pushing code to the `main` branch will automatically start the CI/CD pipeline, building and deploying your model to AWS SageMaker.

-----

### Conclusion

This project provides a comprehensive and practical example of a production-ready machine learning system. By automating the entire process from training to deployment, it demonstrates key MLOps principles, including reproducibility, consistency, and scalability, making it an excellent showcase for any data science or MLOps portfolio.
