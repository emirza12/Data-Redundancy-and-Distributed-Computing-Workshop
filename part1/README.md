# Titanic Prediction API with Consensus Model

This repository contains a Flask-based API that allows you to predict the survival status of Titanic passengers using multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, and SVM). Additionally, a **consensus model** aggregates the predictions from all models to provide a final prediction.

## Prerequisites

Ensure you have the following software installed:

- Python 3.x
- Flask
- Ngrok
- Scikit-learn
- Pandas
- Numpy

You can install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include the following:

### How to Set Up the Project

 **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

 **Run Flask App**:

   Start the Flask app using the following command:

   ```bash
   python app.py
   ```

   This will start the Flask app on `http://127.0.0.1:5000`.

 **Expose the Flask App via Ngrok** (optional):

   If you want to access the API from other computers or share it with others, you can use [Ngrok](https://ngrok.com/).

   - First, download and install Ngrok from the [official site](https://ngrok.com/download).
   - Start Ngrok to expose your local Flask server:

     ```bash
     ngrok http 5000
     ```

     Ngrok will provide a publicly accessible URL (e.g., `http://<random_id>.ngrok.io`).

## API Endpoints

### 1. `/predict`

This endpoint allows you to get a prediction for survival using a specified model.

**Query Parameters**:

- `model`: The model you want to use for prediction. It can be one of the following:
  - `logistic`
  - `decision`
  - `random`
  - `svm`
- `pclass`: Passenger class (integer)
- `sex`: Gender (0 for male, 1 for female)
- `age`: Age of the passenger (float)
- `fare`: Fare paid by the passenger (float)

**Example Request**:
```bash
http://127.0.0.1:5000/predict?model=logistic&pclass=3&sex=1&age=22.0&fare=7.25
```

**Response**:

```json
{
  "model": "logistic_regression",
  "prediction": "Survived"
}
```

### 2. `/predict_consensus`

This endpoint allows you to get a prediction for survival using the **consensus model** (average of all model predictions).

**Query Parameters**:

- `pclass`: Passenger class (integer)
- `sex`: Gender (0 for male, 1 for female)
- `age`: Age of the passenger (float)
- `fare`: Fare paid by the passenger (float)

**Example Request**:
```bash
http://127.0.0.1:5000/predict_consensus?pclass=3&sex=1&age=22.0&fare=7.25
```

**Response**:

```json
{
  "avg_probability": 0.8698302023630932,
  "model": "Consensus Model",
  "prediction": "Survived"
}
```