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

### Question 1: How many observations does your dataset have?
<img width="342" height="728" alt="image" src="https://github.com/user-attachments/assets/348b2dee-d488-4ff2-93b1-b71bf5563c89" />


### Question 2: Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.
<img width="868" height="763" alt="image" src="https://github.com/user-attachments/assets/faeaaba7-ca3b-445e-8d05-0c73fa28ab20" />

Dataset Column Description
Our dataset is a comprehensive collection of NFL statistics from 1999-2022, comprising 215,243 total observations distributed across eight CSV files. The data have two primary axes: (weekly vs. yearly) and (player vs. team), which is ideal for our goal of building a fantasy football autodrafter.

Player-Level Data (Primary Focus):

Weekly Offensive Stats (58,629 observations): This is the most critical dataset for our project. Each observation represents a single player's offensive performance in one game. 
Yearly Offensive Stats (7,133 observations): Adding all weekly data (16 - 17 games) into a full-season summary for each player, useful for establishing a baseline performance. 
Defensive Stats (Weekly & Yearly): While our primary focus is offense, the 117,993 weekly and 16,148 yearly defensive observations will be used for drafting team defenses (D/ST)
Team-Level Data (Contextual): The team-level files provide broader context on team tendencies (e.g., pass-heavy vs. run-heavy offenses) that can influence a player's opportunity and projected output.

<img width="946" height="332" alt="image" src="https://github.com/user-attachments/assets/7dfedeea-2b04-4d59-9168-4ab59cc35ee4" />
For our fantasy football autodrafter, the dataset does not have a pre-made target column. 
Our target variable will be 'fantasy_points', which we will need to create. 
How it will be created: 
We will calculate it based on standard fantasy scoring rules applied to the continuous variables. 
For example, in a Points Per Reception (PPR) league: 
  - 1 point per reception ('receptions') 
  - 0.1 points per rushing/receiving yard ('rushing_yards', 'receiving_yards')
  - 6 points per rushing/receiving touchdown ('rushing_tds', 'receiving_tds')
  - 4 points per passing touchdown ('passing_tds')
  - -2 points per interception thrown ('interceptions')

This engineered 'fantasy_points' column will be a continuous variable on a ratio scale, 
and it will be the value our model aims to predict for future player performance.


### Question 3: Do you have missing and duplicate values in your dataset?
<img width="1265" height="808" alt="image" src="https://github.com/user-attachments/assets/8181a913-540b-4180-848e-2b2a10c1e98b" />


<img width="1032" height="802" alt="image" src="https://github.com/user-attachments/assets/1d511e7f-323f-4d72-a768-5b356a01f08a" />


<img width="1133" height="552" alt="image" src="https://github.com/user-attachments/assets/aecc071d-ea99-4385-a042-f0821d3430a5" />

