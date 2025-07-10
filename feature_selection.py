import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(df, target_column, k=5):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Correlation-based selection for numerical features
    corr_matrix = df[numerical_cols].corr()
    high_corr = corr_matrix[abs(corr_matrix) > 0.8].stack().index.tolist()
    print("Highly correlated features:", high_corr)
    
    # Feature importance using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))
    
    # Select top k features using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print("\nSelected Features:", selected_features)
    
    return selected_features, X[selected_features]

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    selected_features, X_selected = select_features(df, target_column="target")