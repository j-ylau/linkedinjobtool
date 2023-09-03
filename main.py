from preprocessing import load_and_preprocess_data
from nlp_processing import preprocess_text_data, perform_tfidf_vectorization
from model_building import build_and_train_model
from evaluation import evaluate_model

def main():
    # Step 1: Data Collection and Preprocessing
    data = load_and_preprocess_data('./data/job_postings.csv')

    # Step 2: NLP Preprocessing
    data['processed_description'] = data['description'].apply(preprocess_text_data)
    X = perform_tfidf_vectorization(data['processed_description'])

    # Step 3 & 4: Model Building
    model, X_test, y_test = build_and_train_model(X, data['med_salary'])

    # Step 5: Model Evaluation
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
