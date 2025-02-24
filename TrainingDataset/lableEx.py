import pandas as pd

def get_real_and_fake_articles(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Ensure the CSV contains 'text' and 'label' columns
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns")
    
    # Filter articles by 'REAL' and 'FAKE' labels
    real_articles = data[data['label'].str.upper() == 'REAL'].head(20)
    fake_articles = data[data['label'].str.upper() == 'FAKE'].head(20)
    
    # Combine real and fake articles
    selected_articles = pd.concat([real_articles[['text', 'label']], fake_articles[['text', 'label']]])
    
    return selected_articles

# Example usage
csv_file_path = '/organized_dataset.csv'  # Replace with your CSV file path
articles = get_real_and_fake_articles(csv_file_path)
print(articles)
