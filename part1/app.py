from flask import Flask, request, jsonify
import pickle
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

app = Flask(__name__)

# Dictionary to store loaded models and their performance (accuracy)
models = {}
model_performance = {}

# Load saved models
model_files = ["logistic_regression_model.pkl", "decision_tree_model.pkl", "random_forest_model.pkl", "svm_model.pkl"]
model_path = "models"

if os.path.exists(model_path):
    for model_file in model_files:
        model_name = model_file.split('_')[0]  # Extract model name
        try:
            with open(f"{model_path}/{model_file}", 'rb') as file:
                models[model_name] = pickle.load(file)
            model_performance[model_name] = 0.25  # Initialize with equal weight
            print(f"{model_name} model loaded successfully!")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
else:
    print("The 'models' directory does not exist.")

# Evaluate models and update their performance
def evaluate_models():
    # Load Titanic dataset for evaluation
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numerical values

    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']

    # Evaluate models and store their accuracies
    total_accuracy = 0
    print("\nInitial weights and performance:")
    for model_name, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        model_performance[model_name] = accuracy  # Update model performance
        total_accuracy += accuracy
        print(f"{model_name} accuracy: {accuracy:.2f} | Initial weight: {0.25:.2f}")  # Print initial weights

    # Normalize the model performance (weights)
    print("\nNormalizing weights...")
    for model_name in model_performance:
        model_performance[model_name] /= total_accuracy  # Normalize to ensure weights sum to 1

    print("\nUpdated weights after normalization:")
    for model_name, weight in model_performance.items():
        print(f"{model_name}: {weight:.2f}")

# Run evaluation when starting the app
evaluate_models()

@app.route('/')
def home():
    return "Welcome to the Titanic Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
    model_name = request.args.get('model', default='logistic_regression', type=str)

    if model_name not in models:
        return jsonify({'error': f"Invalid model. Choose from {list(models.keys())}."})

    try:
        pclass = int(request.args.get('pclass'))
        sex = int(request.args.get('sex'))  # 0 for male, 1 for female
        age = float(request.args.get('age'))
        fare = float(request.args.get('fare'))
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid input parameters.'})

    input_data = [[pclass, sex, age, fare]]
    model = models[model_name]
    prediction = model.predict(input_data)
    return jsonify({
        'model': model_name,
        'prediction': 'Survived' if prediction[0] == 1 else 'Did not survive'
    })

@app.route('/predict_consensus', methods=['GET'])
def predict_consensus():
    try:
        pclass = int(request.args.get('pclass'))
        sex = int(request.args.get('sex'))  # 0 for male, 1 for female
        age = float(request.args.get('age'))
        fare = float(request.args.get('fare'))
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid input parameters.'})

    input_data = [[pclass, sex, age, fare]]

    # Store the predictions of all models
    predictions = []
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):  # Check if model has predict_proba method
            prob = model.predict_proba(input_data)[0, 1]  # Probability of class '1' (Survived)
            predictions.append(prob)

    # Apply weights to the predictions
    weighted_avg = np.dot(predictions, list(model_performance.values()))  # Weighted average
    consensus_prediction = 'Survived' if weighted_avg >= 0.5 else 'Did not survive'

    # Print the consensus performance in the console
    print("\nConsensus Prediction Performance:")
    print(f"Weighted Average Probability: {weighted_avg:.2f}")
    print(f"Consensus Prediction: {consensus_prediction}")
    
    # Calculate performance of the consensus model
    # In this case, we need to use the same dataset to calculate the accuracy of the consensus model
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']

    # Evaluate consensus model using weighted predictions
    consensus_predictions = []
    for i in range(len(X)):
        prob = np.dot([model.predict_proba([X.iloc[i]])[0, 1] for model in models.values()], list(model_performance.values()))
        consensus_predictions.append(1 if prob >= 0.5 else 0)

    consensus_accuracy = accuracy_score(y, consensus_predictions)
    print(f"Consensus Model Accuracy: {consensus_accuracy:.2f}")
    
    return jsonify({
        'model': 'Weighted Consensus Model',
        'avg_probability': weighted_avg,
        'prediction': consensus_prediction,
        'consensus_accuracy': consensus_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)

