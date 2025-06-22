import os
import pandas as pd

# Full path to your main CSV file
full_path = os.path.join(os.path.dirname(__file__), 'data', 'MachineLearningCSV.csv')

# Create data/ folder if it doesn't exist
sample_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(sample_dir, exist_ok=True)

# Load and sample the data
df = pd.read_csv(full_path, encoding='latin1')
df.sample(n=1000, random_state=42).to_csv(os.path.join(sample_dir, 'sample.csv'), index=False)

print("✅ sample.csv generated in the data folder!")
