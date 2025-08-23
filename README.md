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

# Question 4: Data Exploration

### 4.1: How many Observations does your data have?
<img width="342" height="728" alt="image" src="https://github.com/user-attachments/assets/348b2dee-d488-4ff2-93b1-b71bf5563c89" />


### 4.2: Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.
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


### 4.3: Do you have missing and duplicate values in your dataset?
<img width="1265" height="808" alt="image" src="https://github.com/user-attachments/assets/8181a913-540b-4180-848e-2b2a10c1e98b" />

# Question 5: Data Plots

### Plot your data with various types of charts like bar charts, pie charts, scatter plots etc. and clearly explain the plots.

<img width="1032" height="802" alt="image" src="https://github.com/user-attachments/assets/1d511e7f-323f-4d72-a768-5b356a01f08a" />

**Description for the Pie Chart (Position Distribution)**
This pie chart illustrates the distribution of players across different positions within our dataset. From a fantasy drafting perspective, this visualization is key to understanding positional value. For example, positions with smaller slices, like Running Back or Tight End, demonstrates that theres less talents/players available at those positions. This increases the value of drafting those elite players at those positions, as the drop-off in production to the next available player is much steeper. Our autodrafter will use this information to prioritize drafting top-tier talent at these premium positions early on, securing a significant advantage over our opponents.

<img width="1133" height="552" alt="image" src="https://github.com/user-attachments/assets/aecc071d-ea99-4385-a042-f0821d3430a5" />
**Description for the Bar Chart (Fantasy Points Consistency)**
This bar chart displays the consistency of fantasy scoring by position, measured by the standard deviation of weekly points. A lower bar signifies more consistent, predictable scoring, while a higher bar indicates more "boom-or-bust" potential. For a successful fantasy draft, balancing high-upside players with a reliable, high-floor team is essential.

Lower Bars (More Consistent): Positions like Quarterback often show lower variance, providing a stable scoring floor each week. These are safer, more dependable assets.
Higher Bars (Less Consistent): Positions like Wide Receiver or Tight End can have higher variance, meaning their weekly scores fluctuate more. 

Our autodrafter will leverage this insight to manage risk. It will prioritize drafting players from more consistent positions to build a reliable core, while strategically targeting less consistent, high-upside players to gain a competitive edge in weekly matchups. This balance is fundamental to assessing true positional value.

### Question 6: How will you preprocess your data? Handle data imbalance if needed. You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo. (3 points)

Our preprocessing approach is designed to transform raw weekly player statistics into meaningful, position-specific performance metrics. The main goal is to calculate the Value over Replacement Player, for each player at every position. For example, a top tier Quarterback may have the highest number of raw points, but taking a quarterback in the first round is usually not the best strategy since there are many quarterbacks who perform well each week since the position is the most consistent in the sport usually. Therefore, choosing a top tier running back or Wide Receiver in the first round is the best, since their VORP, or value over a replacement player is ultimately the best factor of who you should draft first. 

The best fantasy team is one that is well balanced at each position, and while nobody can get all the best players on one team, the best approach is to build a balanced team that can consistently produce on a week to week basis.

**1. Aggregation of Weekly Data**



**2. VORP Calculation**



**3. Position-Specific Normalization**

Each position has unique performance metrics (e.g., rushing yards for RBs, receiving yards for WRs/TEs, passing yards for QBs).

Metrics are normalized per position to account for natural differences in production scales and ensure fair comparison across players.

**4. Additional Factors**

Player age, seasons played, and general position trends (e.g., typical weekly output) are incorporated.

This helps adjust rankings for players who may be improving, aging, or trending differently relative to their peers.

**5. Handling Data Imbalance**

Some positions (e.g., TEs) or metrics may have fewer observations than others.

While we do not perform explicit oversampling or undersampling at this stage, we note these imbalances for consideration in downstream modeling.

Aggregating metrics and computing normalized scores per position partially mitigates imbalance issues by comparing players within the same role.
