import pandas as pd

def load_and_preprocess_data(filepath):
    job_data = pd.read_csv(filepath)
    relevant_columns = ['description', 'max_salary', 'med_salary', 'min_salary']
    job_data = job_data[relevant_columns]
    job_data.dropna(inplace=True)
    return job_data
