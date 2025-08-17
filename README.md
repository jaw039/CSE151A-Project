# CSE151A-Project — NFL Stats (1999–2022)

## Dataset
- **Source**: [NFL Stats (1999-2022) on Kaggle](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)
- **Description**: Comprehensive NFL player and team statistics covering weekly and yearly data from 1999 to 2022

## Environment Setup Requirements

### Prerequisites
- Python 3.8+ 
- pip package manager
- Kaggle account with API access

### Dependencies
Install required packages:
```bash
pip install pandas numpy matplotlib seaborn jupyterlab kaggle
```

### Kaggle API Setup
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download the `kaggle.json` file
4. Place `kaggle.json` in the appropriate directory:
   - **Windows**: `%USERPROFILE%\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`
5. Set proper file permissions (read-only for user)

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/jaw039/CSE151A-Project.git
cd CSE151A-Project
```

### 2. Download Dataset
Run the data download script:
```bash
python get_data.py
```

This will:
- Download the NFL stats dataset from Kaggle
- Extract files to the `data/` directory
- Verify the setup is complete

### 3. Verify Installation
Check that data files are available:
```python
import pandas as pd
import os

# List available data files
print("Available data files:")
for file in os.listdir("data/"):
    if file.endswith('.csv'):
        print(f"- {file}")

# Load sample data
df = pd.read_csv("data/weekly_player_stats_offense.csv")
print(f"\nSample data shape: {df.shape}")
print(df.head())
```

## Project Structure
```
CSE151A-Project/
├── data/                    # Dataset files (created after running get_data.py)
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code and scripts
├── get_data.py            # Dataset download script
└── README.md             # This file
```

## Usage Notes
- The `data/` folder is created automatically when running `get_data.py`
- Large data files should be added to `.gitignore` to avoid committing to version control
- Dataset contains both weekly and yearly statistics for offensive and defensive players
- All CSV files include headers with detailed column descriptions

## Troubleshooting
- **Kaggle API errors**: Ensure `kaggle.json` is properly placed and has correct permissions
- **Permission errors**: Check that your Kaggle account has access to public datasets
- **Download failures**: Verify internet connection and try running `get_data.py` again

## License
This project is for educational purposes. Please respect Kaggle's terms of service and the original dataset license.
