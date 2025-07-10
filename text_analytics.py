import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def text_analytics(df, text_column):
    # Initialize sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")
    
    # Preprocess text
    df[text_column] = df[text_column].str.lower().str.replace(r'[^\w\s]', '')
    
    # Sentiment analysis
    sentiments = [classifier(text)[0] for text in df[text_column]]
    df["sentiment"] = [result["label"] for result in sentiments]
    df["sentiment_score"] = [result["score"] for result in sentiments]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df[text_column])
    feature_names = vectorizer.get_feature_names_out()
    print("Top TF-IDF Features:", feature_names[:10])
    
    return df, X

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("text_data.csv")
    df, X = text_analytics(df, text_column="review")
    df.to_csv("text_analyzed.csv", index=False)