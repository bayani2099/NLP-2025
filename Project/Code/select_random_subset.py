import pandas as pd

# Load the dataset from CSV
input_file = '../data/preprocessed/DF_train.csv' 
data = pd.read_csv(input_file)

# Ensure the dataset has the required column
if 'label_sexist' not in data.columns:
    raise ValueError("The dataset does not contain the required column 'label_sexist'.")

# Separate the data into two groups based on the label
sexist_data = data[data['label_sexist'] == 1]
not_sexist_data = data[data['label_sexist'] == 0]

# Randomly select 50 samples from each group
sexist_sample = sexist_data.sample(n=50, random_state=42)
not_sexist_sample = not_sexist_data.sample(n=50, random_state=42)

# Combine the samples into a single dataframe
balanced_sample = pd.concat([sexist_sample, not_sexist_sample])

# Select only the required columns
balanced_sample = balanced_sample[['rewire_id', 'text', 'label_sexist']]

# Save the samples to a new CSV file
output_file = '../data/preprocessed/DF_train_subset.csv'
balanced_sample.to_csv(output_file, index=False)

print(f"Balanced samples saved to '{output_file}'")
