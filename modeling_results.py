import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

def evaluate_models(X, y, problem_type="classification"):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate based on problem type
        if problem_type == "regression":
            score = mean_squared_error(y_test, predictions)
            results[name] = {"MSE": score}
        else:
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            results[name] = {"Accuracy": score, "Report": report}
        
        print(f"\n{name} Results:")
        print(results[name])
    
    return results

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    results = evaluate_models(X, y, problem_type="classification")