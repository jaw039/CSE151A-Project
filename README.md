# CSE151A-Project — NFL Fantasy Football Point Projections

# Introduction
Every year, millions of Americans get ready to participate in a tradition that pits them against their friends, family and loved ones. This is of course, Fantasy Football, a competition to see who can draft the best team and win a fictional league, based off week to week performance of real NFL players. Due to the short lived nature of NFL careers, year to year changes in coaching, drafted players, rosters, and player roles, it is oftentimes impossible to predict the state of football year after year. Yet, despite that, we tried to do it, using statistical analysis and machine learning to predict how to draft the best team possible. Having a good predictive mode for fantasy football is used in many autodrafters for leagues online, but predictive analysis and statistical modeling is used for college players as well, to see how they may or may not translate to the next level. This is important to teams, because they may be able to find undervalued players in the draft that they can get for a "steal" because other teams passed, not knowing their true value, or it can be used to evaluate the "bust" potential of a player who may be overvalued due to a skillset that won't translate well to the professional level.

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

# Data Exploration

# Major Preprocessing
There are many different positions and they all require a different set of values to calculate their fantasy value/projections, so we decided to just focus on one and try to get our supervised learning model to predict projected values for the next season based on previous season's values. 

We decided to go with the Running Back position, as it is one of the shortest lived positions in the NFL, but one of the most valuable in fantasy, due to the running back getting rushes, receptions, rushing TDs, receiving TDs, fumbles, and lots of snaps on offense. We figured that if we were able to generalize the regression model for this position, then being able to do the same for other positions wouldn't be as challenging.

For Our Preprocessing, we just filtered the yearly offense stats to the running back position, and removed all irrelevant columns, keeping only the player name, the season, total rushing yards per season, total rushing touchdowns per season, total snaps on offense per season, and Yards Per Carry per season. Additionally, we had to filter by season type, and remove all Post Season stats since they don't factor into fantasy football and only the regular season does. After this was done, we were ready to begin our model training based on our data.

<img width="1070" height="632" alt="image" src="https://github.com/user-attachments/assets/434ecccc-2a20-4b6e-8943-3b0d5e8b040a" />

The above is a graph depicting the relationship of previous year's rushing yards to the fantasy points for the following year. We can see a linear trend in the data with lots of variance. Some notable outliers in this include Christian McCaffrey, who suffered a patellar tendon injury and was unable to play for much of the season, despite having been the consensus best running back in the league the previous season due to his production as the engine of the 49ers offense. Another outlier is Chase Brown for the Cincinatti Bengals, who happened to lose their starting running back Joe Mixon in free agency in 2024 to the Houston Texans. Chase Brown saw his snap counts increase greatly as a result. Many such outliers exist, with the state of the NFL and its players in constant flux, especially at the Running Back position where talent is plenty, but careers are short lived.

<img width="1062" height="756" alt="image" src="https://github.com/user-attachments/assets/8c1118bf-2546-4cb6-854f-f2abe17305e8" />
The above is a grid of graphs depicting the relationship between snap count (how many times a player was on the field over the course of the season), Rushing Yards, Rushing Touchdowns, and Yards Per Carry (How many total Yards / How many rushing attempts). Many of these variables have a vaguely linear correlation with 2024 fantasy points, except for Yards Per Carry, which is to be expected as it can be biased towards small sample sizes. 

<img width="1191" height="800" alt="image" src="https://github.com/user-attachments/assets/7c678fbc-215d-4b44-85d8-d05497e3b7e9" />
# First Model
We used Support Vector Regression to try and find a prediction line to fit the data we had, in order to predict the stats of a player the next season depending on the various stats we had calculated from the previous season. The test set size was 20% of our total set of data of running backs. It also had notable outliers including Christian McCaffrey, and Chase Brown, as well as Saquon Barkley who saw one of the best historical seasons at Running Back after signing with a new team that catered much better to his strengths than his previous team.
The error reportings are as follows: 
Train RMSE: 69.9235350510531
Test RMSE: 97.70489967679828
Train R2: 0.47908500488083483
Test R2: -0.025677153402659858

# Tweaking our Model
The Test RMSE seemed very high, and this is a sign that the model isn't well trained, since it is significantly higher than our Training RMSE, so we decided to go and tweak a few things with the model. First, we decided to only score the RB1 and RB2 on every team, which meant that we would be getting rid of many of the lower performers that likely wouldn't be drafted in an 8-12 man fantasy league. This meant we'd only be keeping Running Backs who were RB1 or RB2 in both 2023 and 2024, meaning no rookies or risers would be considered. This significantly reduced the size of our dataset, but increased our model's accuracy by removing data that was skewing our model. 

<img width="1200" height="816" alt="image" src="https://github.com/user-attachments/assets/3945f172-d99e-4f55-addf-d1edd8347a69" />

Train RMSE: 71.8861509058448
Test RMSE: 74.92593988977025
Train R2: 0.45734933567917113
Test R2: 0.2159844530783509

Then, we used GridSearchCV to find the best hyperparameters for our SVR, and implemented those into our model, giving us the following:
<img width="1202" height="810" alt="image" src="https://github.com/user-attachments/assets/7c59693d-bc07-4ad2-88b5-8cf72050895a" />
Train RMSE: 85.32755327031286
Test RMSE: 71.48681447974681
Train R2: 0.2354454736847571
Test R2: 0.28630580305958364

While these values were lower than our previous hyperparameters, the samples were all very close to the line and saw very few outliers, so we decided to tweak the sample and found that the RMSE was increased drastically for this model when we did. We decided to settle on the parameters {'C': 10, 'epsilon': 1.0, 'gamma': 'scale', 'kernel': 'rbf'} as it was the one that best fit our data. We think that due to the relatively small sample size, fluid nature of the sport and rosters, injuries, scheme fits, coaching changes, offensive line play, and more, statistical modeling for one position in isolation is very difficult to find a reliable prediction for. The data will almost always be underfitted, since sports are a very small sample space and see statistical anomalies happen very often, especially in a 17 game season like the NFL. The position we chose may also be at fault due to it being a much more injury prone position, as well as being much more reliant on other factors such as offensive line blocking, defensive run stopping, and more. When predicting stats for other positions, we may not see this much variance in season to season production, and may find it much more reliable in terms of data. 

# Conclusions
Ways to improve our model could be through better feature engineering to handle outliers, and account for injuries, coaching changes, and offensive line efficiency, as well as the defense played against. While factoring in offensive line efficiency, coaching changes, and defensive efficiency is quite subjective, injuries and outliers due to injury can have a big effect on our model. 

Injury handling can be done through weighting snaps played more. If a player has less snaps than usual, it is indicative that either they aren't producing well, or they are suffering through an injury. Regardless, these are both valid reasons to weight snaps played more heavily, as they indicate the number of times that a player will be in use. 

Outliers are slightly more difficult to manage. Chase Brown for example, was an example of an outlier due to his promotion on the team from another player leaving. While Chase Brown isn't a top 5 running back in the league, he saw a great increase in production from having more touches alone. Having some sort of multiplier given to a player if their depth chart position moves upward from one season to the next could be a good way to implement a better prediction algorithm for this kind of outlier. 

In the next milestone, we hope to implement some kind of predictive modeling for all offensive positions in fantasy, and implementing some kind of Gradient Boosting Regressor with an emphasis on game theory to select the best player available in a fantasy draft, and to build the best team possible. We would do this by looking at previous season statistics and calculating who the best N players will be the next season, and how much better each player is than the Value over Replacement Player, which would be the (N+1)th player. Then we would use a Gradient Boosting Regressor to select the best player at each point in the draft, and evaluate how well it did by looking at the total VORP and comparing it to the real statistics of that season.


## License
This project is for educational purposes. Please respect Kaggle's terms of service and the original dataset license.

