import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, output_dir="eda_plots"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe(include='all'))
    
    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        # Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=30)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{output_dir}/{col}_histogram.png")
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{output_dir}/{col}_boxplot.png")
        plt.close()
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, data=df)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.savefig(f"{output_dir}/{col}_countplot.png")
        plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    print(f"EDA plots saved in {output_dir}")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    perform_eda(df)