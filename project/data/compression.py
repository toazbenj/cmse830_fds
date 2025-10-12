import io
import pandas as pd
import time
import os

# Load your CSV once
csv_path = 'AURSAD.pkl'
feather_path = 'aursad.feather'
df = pd.read_pickle(csv_path)

# Test Feather
start = time.time()
df.to_feather(feather_path)
feather_write_time = time.time() - start

start = time.time()
df_feather = pd.read_feather(feather_path)
feather_read_time = time.time() - start

# Check file sizes
csv_size = os.path.getsize(csv_path) / (1024*1024)  # MB
feather_size = os.path.getsize(feather_path) / (1024*1024)  # MB

print(f"CSV size: {csv_size:.1f}MB")
print(f"Feather size: {feather_size:.1f}MB")
print(f"Feather read time: {feather_read_time:.2f}s")
print(f"Feather write time: {feather_write_time:.2f}s")

print(df_feather.head())
