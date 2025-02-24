import pandas as pd

# Load the original dataset
data = pd.read_csv('D:/FakeNewsDetected/TrainingDataset/Uploaded_data.csv')

# Display first few rows to check structure
print(data.head())

# Filter FAKE and REAL news
fake_data = data[data['label'].str.lower() == 'fake']
real_data = data[data['label'].str.lower() == 'real']

# Check if there are enough records
if len(fake_data) >= 1000 and len(real_data) >= 1000:
    # Select 1000 fake and 1000 real news articles
    fake_data_sample = fake_data.sample(n=1000, random_state=42)
    real_data_sample = real_data.sample(n=1000, random_state=42)

    # Combine the two datasets
    new_dataset = pd.concat([fake_data_sample, real_data_sample])

    # Shuffle the dataset to mix fake and real news
    new_dataset = new_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the new dataset to a CSV file
    new_dataset.to_csv('D:/FakeNewsDetected/FakeNews/organized_dataset.csv', index=False)

    print("1000 real and 1000 fake data has been successfully saved to 'organized_dataset.csv'.")

else:
    print("Insufficient data: The dataset must have at least 1000 real and 1000 fake news articles.")

