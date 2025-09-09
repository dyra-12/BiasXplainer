import glob
import pandas as pd

# Get list of files sorted by block number
csv_files = sorted(glob.glob("block_*.csv"))

# Read and concatenate
dfs = [pd.read_csv(f) for f in csv_files]
combined = pd.concat(dfs, ignore_index=True)

# Write to file
combined.to_csv("combined.csv", index=False)