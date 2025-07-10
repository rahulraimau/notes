import pandas as pd

def clean_data(df):
    # Check initial data info
    print("Initial Data Info:")
    print(df.info())
    
    # Handle missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    df = df.fillna(df.mean(numeric_only=True))  # Fill numeric columns with mean
    df = df.fillna("Unknown")  # Fill categorical columns with 'Unknown'
    
    # Remove duplicates
    df = df.drop_duplicates()
    print("\nDuplicates removed:", df.duplicated().sum())
    
    # Convert data types
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 10:  # Convert to category if few unique values
            df[col] = df[col].astype('category')
    
    # Final data info
    print("\nCleaned Data Info:")
    print(df.info())
    return df

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    cleaned_df = clean_data(df)
    cleaned_df.to_csv("cleaned_data.csv", index=False)