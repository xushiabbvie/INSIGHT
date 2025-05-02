import pandas as pd

# Load the data from the file into a pandas DataFrame
drug_data = pd.read_csv(
    "demo_data.txt", 
    sep="\t", header=None
)

# Create a frequency table of the second column (V2)
drug_freq = drug_data[1].value_counts().reset_index()
drug_freq.columns = ['Var1', 'Frequency']

# Calculate the mean of the third column (V3) for each unique value in the second column (V2)
drug_mean = drug_data.groupby(1)[2].mean()

# Assign the means to their respective names
drug_mean_dict = drug_mean.to_dict()

# Create a copy of the original data
drug_data_mean0 = drug_data.copy()

# Subtract the mean values from the third column (V3) based on the second column (V2)
drug_data_mean0[2] = drug_data[2] - drug_data[1].map(drug_mean_dict)

# Save the modified DataFrame to a new file
drug_data_mean0.to_csv(
    "demo_data_mean0.txt",
    header=False, index=False, sep="\t"
)