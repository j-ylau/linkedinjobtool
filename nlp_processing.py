from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text_data(text):
    print(f"Original Text: {text[:50]}")  # Print the first 50 characters of the original text
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [ps.stem(word) for word in words if word.lower() not in stop_words]
    
    processed_text = ' '.join(words)
    print(f"Processed Text: {processed_text[:50]}")  # Print the first 50 characters of the processed text
    
    return processed_text

def perform_tfidf_vectorization(text_data):
    vectorizer = TfidfVectorizer()
    transformed_data = vectorizer.fit_transform(text_data)
    print(f"Transformed Data Shape: {transformed_data.shape}")
    return transformed_data
