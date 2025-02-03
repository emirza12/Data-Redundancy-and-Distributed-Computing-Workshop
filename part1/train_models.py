import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Prepare data
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numerical values

X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "logistic_regression": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC()
}

# Create 'models' folder if it doesn't exist
models_directory = 'models'
if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Train and save each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional performance metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} accuracy: {accuracy:.2f}")
    print(f"{model_name} precision: {precision:.2f}")
    print(f"{model_name} recall: {recall:.2f}")
    print(f"{model_name} F1-score: {f1:.2f}")
    print(f"{model_name} confusion matrix:\n{conf_matrix}")
    
    # Save the model to a .pkl file
    model_filename = os.path.join(models_directory, f"{model_name}_model.pkl")
    try:
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"{model_name} model saved successfully!\n")
    except Exception as e:
        print(f"Error saving {model_name}: {e}")
