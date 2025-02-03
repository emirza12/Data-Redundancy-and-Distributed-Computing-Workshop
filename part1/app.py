from flask import Flask, request, jsonify
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

app = Flask(__name__)

# Dictionary to store loaded models
models = {}

# Load saved models
model_files = ["logistic_regression_model.pkl", "decision_tree_model.pkl", "random_forest_model.pkl", "svm_model.pkl"]

model_path = "models"
if os.path.exists(model_path):
    for model_file in model_files:
        model_name = model_file.split('_')[0]  # Extract model name, e.g., 'logistic_regression'
        try:
            with open(f"{model_path}/{model_file}", 'rb') as file:
                models[model_name] = pickle.load(file)
            print(f"{model_name} model loaded successfully!")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
else:
    print("The 'models' directory does not exist.")

# Evaluate models and print results
def evaluate_models():
    # Load Titanic dataset for evaluation
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numerical values

    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']

    # Evaluate models
    for model_name, model in models.items():
        y_pred = model.predict(X)
        
        # Accuracy
        accuracy = accuracy_score(y, y_pred)
        
        # Additional performance metrics
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Print evaluation results for each model
        print(f"\n--- {model_name} Model Evaluation ---")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print()

# Run evaluation when starting the app
evaluate_models()

@app.route('/')
def home():
    return "Welcome to the Titanic Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
    model_name = request.args.get('model', default='logistic_regression', type=str)

    # Check if the model is valid
    if model_name not in models:
        return jsonify({'error': f"Invalid model. Choose from {list(models.keys())}."})

    # Get parameters from the request
    try:
        pclass = int(request.args.get('pclass'))
        sex = int(request.args.get('sex'))  # 0 for male, 1 for female
        age = float(request.args.get('age'))
        fare = float(request.args.get('fare'))
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid input parameters.'})

    # Prepare the input data for prediction
    input_data = [[pclass, sex, age, fare]]

    # Make the prediction
    model = models[model_name]
    prediction = model.predict(input_data)

    # Return the prediction
    return jsonify({
        'model': model_name,
        'prediction': 'Survived' if prediction[0] == 1 else 'Did not survive'
    })

# New route for consensus prediction (averaging model outputs)
@app.route('/predict_consensus', methods=['GET'])
def predict_consensus():
    try:
        pclass = int(request.args.get('pclass'))
        sex = int(request.args.get('sex'))  # 0 for male, 1 for female
        age = float(request.args.get('age'))
        fare = float(request.args.get('fare'))
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid input parameters.'})

    # Prepare the input data for prediction
    input_data = [[pclass, sex, age, fare]]

    # Store the predictions of all models
    predictions = []

    # Get the output probabilities from each model
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):  # Check if model has predict_proba method
            prob = model.predict_proba(input_data)[0, 1]  # Get the probability of class '1' (Survived)
            predictions.append(prob)

    # Average the probabilities of all models
    avg_prob = np.mean(predictions)

    # Determine the consensus prediction based on the averaged probability
    consensus_prediction = 'Survived' if avg_prob >= 0.5 else 'Did not survive'

    # Return the consensus prediction
    return jsonify({
        'model': 'Consensus Model',
        'avg_probability': avg_prob,
        'prediction': consensus_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
