import os
import zipfile

# Get the Kaggle API key 
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    raise FileNotFoundError("Kaggle API key not found. Place kaggle.json in ~/.kaggle/")

# Download dataset
os.system("kaggle datasets download -d philiphyde1/nfl-stats-1999-2022 -p data/")

# Unzip dataset
zip_path = "data/nfl-stats-1999-2022.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/")

print("Dataset ready in 'data/' folder.")
