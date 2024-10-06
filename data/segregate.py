import os
import pandas as pd

csv_path = os.getcwd() + '/features_3_sec.csv/features_3_sec.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(csv_path)

# Select the specified columns
selected_columns = [
    'filename', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
    'spectral_bandwidth_var', 'label'
]

# Filter the DataFrame to include only the selected columns
filtered_data = data[selected_columns]

# Display the first few rows of the filtered DataFrame
print(filtered_data.columns)

# Split the data into training and testing sets
# Select the last 250 samples of each label
train_data = filtered_data.groupby('label', group_keys=False).apply(lambda x: x.tail(250))
test_data = filtered_data.drop(train_data.index)

# Save the training data to a new CSV file
train_data.to_csv('train_data.csv', index=False)

# Save the testing data to a new CSV file
test_data.to_csv('test_data.csv', index=False)