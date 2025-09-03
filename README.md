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

TODO: ADD CORRELATION MATRIX USING SEABORN BETWEEN CATEGORICAL VARIABLES FOR RB, QB, WR

<img width="1032" height="802" alt="image" src="https://github.com/user-attachments/assets/1d511e7f-323f-4d72-a768-5b356a01f08a" />

**Description for the Pie Chart (Position Distribution)**
This pie chart illustrates the distribution of players across different positions within our dataset. From a fantasy drafting perspective, this visualization is key to understanding positional value. For example, positions with smaller slices, like Running Back or Tight End, demonstrates that theres less talents/players available at those positions. This increases the value of drafting those elite players at those positions, as the drop-off in production to the next available player is much steeper. Our autodrafter will use this information to prioritize drafting top-tier talent at these premium positions early on, securing a significant advantage over our opponents.

<img width="1133" height="552" alt="image" src="https://github.com/user-attachments/assets/aecc071d-ea99-4385-a042-f0821d3430a5" />
**Description for the Bar Chart (Fantasy Points Consistency)**
This bar chart displays the consistency of fantasy scoring by position, measured by the standard deviation of weekly points. A lower bar signifies more consistent, predictable scoring, while a higher bar indicates more "boom-or-bust" potential. For a successful fantasy draft, balancing high-upside players with a reliable, high-floor team is essential.

Lower Bars (More Consistent): Positions like Quarterback often show lower variance, providing a stable scoring floor each week. These are safer, more dependable assets.
Higher Bars (Less Consistent): Positions like Wide Receiver or Tight End can have higher variance, meaning their weekly scores fluctuate more. 

Our autodrafter will leverage this insight to manage risk. It will prioritize drafting players from more consistent positions to build a reliable core, while strategically targeting less consistent, high-upside players to gain a competitive edge in weekly matchups. This balance is fundamental to assessing true positional value.

# Major Preprocessing

Our first step in preprocessing was narrowing down our data to the 2023 and 2024 seasons. This was done by setting the dataframe to only include all values where the season was 2023 or 2024. We did this in order to train and evaluate our model on the most recent, and therefore most relevant data. Next, we decided to remove all playoff and postseason statistics. Since this is for Fantasy Football, the regular season is all that matters and playoffs have no bearing on your team performance, so 
we filtered the dataframe further, selecting only "season type" as "regular". 

After doing this, we checked our data for duplicates and null entries, and found some null entries in the "yards per carry" column, so we initialized them with 0s, meaning the player had 0 yards per carry, or didn't have any rushing attempts.

Since our model was predicting fantasy output from the previous season's performance, we decided for running backs, to keep the rushing yards, rushing touchdowns, yards per carry, and snaps played on offense as our indicator values for 2023, and use 2024's fantasy points output as our value to be predicted/ dependent variable. We plotted the charts below for those variables:
<img width="1062" height="756" alt="image" src="https://github.com/user-attachments/assets/f71a5eb2-8f19-42cc-b391-9d9f3a51a131" />



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

# Supervised Learning
## Step 1: Principal Component Analysis:

Since all fantasy players are scored on the same scale, regardless of position, we decided to conduct PCA on player's fantasy data to determine the type of player each one is. The first step we took was filtering the offense data from 2024, to find the offensive players that would be able to impact fantasy points. 

<img width="1376" height="308" alt="image" src="https://github.com/user-attachments/assets/64850f43-f118-4b47-a6d7-277b2f808e34" />

Then we scaled the data to make the mean 0, standard deviation 1, and standardize all players.

<img width="1366" height="392" alt="image" src="https://github.com/user-attachments/assets/d6e665ad-436c-4625-831e-8316c7d2078d" />

After that, we aggregated player stats across the season, and applied PCA on them, to find hidden archetypes that each player followed.

<img width="1367" height="416" alt="image" src="https://github.com/user-attachments/assets/15d20262-cd11-4d1b-9f78-357fe40c6c1b" />

The result was the following 2D graph based on the first two principal components.

<img width="945" height="657" alt="image" src="https://github.com/user-attachments/assets/d33ed7f1-d821-4b69-89ff-0d79ae3f4b5e" />

Looking at the graph, if you know football players, then you will see a clear trend in the three "branches" of our data. To the far bottom right is Quarterback data, which is pretty sparse and spread out due to the relatively smaller number of quarterbacks compared to other positions in the NFL. To the far left bottom is receivers who have a much more dense clustering, due to there being more receivers, and tight ends also taking receiver duty a lot of the time. Then, the center to top part of the graph sees mostly running backs, likely characterized by high rushing yardage, and a larger number of rushing touchdowns. 

Taking a look at the PC1 and PC2 weights, we see that PC1's highest positive weight is indeed Passing Yards, explaining the quarterbacks being far to the right on the X-axis, and it's most negative weight is receptions, followed by receiving yards, explaining receivers being on the low end of the X-axis. Having the highest variance in our data accounted by passing yards makes sense, because of the small number of Quarterbacks, as well as the variety in Quarterback play being from players who play it safe and throw short passes until an opening is found, to improvisers who make their own openings and can extend plays through mobility, and creativity. PC2's highest positive weight is rushing yards, and its most negative weight is passing yards. This explains the high passing yardage but low rushing yardage quarterbacks being towards the bottom of the graph, like "Jared Goff" and "Joe Burrow", while hybrid/dual-threat quarterbacks like Jayden Daniels, Jalen Hurts, and Lamar Jackson can be seen closer to the middle of the graph. 

<img width="757" height="751" alt="image" src="https://github.com/user-attachments/assets/69e7c687-8b72-4f79-899d-85c11eab9c50" />

After adding a 3rd dimension and plotting another Principal Component, our data didn't have much more useful information to tell us, so we decided to stick with using two principal components. An interesting observation we made was that the position of players formed clusters. We conducted K-means clustering to K=3 to see if machine learning could correctly identify the player positions by their position on the graph. 

<img width="952" height="658" alt="image" src="https://github.com/user-attachments/assets/16483bee-119f-4012-8af0-de513de261be" />

After doing K-means clustering, we evaluated the accuracy of this clustering by comparing each cluster with the actual position
. 
<img width="1357" height="252" alt="image" src="https://github.com/user-attachments/assets/7e4314e8-702f-4dd7-a770-ad14c98ab196" />

We saw that there was a minor issue in this method, where all players that didn't have significant passing or rushing yards were classified as WRs/ TEs, regardless of position due to the large number of WRs with very small statistics. To remedy this, I decided to add another cluster, and got the following graph:

<img width="950" height="657" alt="image" src="https://github.com/user-attachments/assets/98b2f158-a46d-4f62-9fb6-ab953f96bb91" />

When evaluating this new K-means algorithm, we got the following:

<img width="416" height="292" alt="image" src="https://github.com/user-attachments/assets/50f3459d-5837-4a50-8d31-cf577f8eecda" />

This new graph was easily able to define the WR/TE position cluster with 100% accuracy, Running Backs in Cluster 2 were not misclassified at all, and QBs were also not misclassified. After playing around with the number of clusters, and increasing/decreasing them, I was able to find this graph: 


<img width="952" height="836" alt="image" src="https://github.com/user-attachments/assets/5fbe69aa-d8b8-47db-a0dc-b6da7a104014" />
I decided to manually label this graph, to show the various different archetypes of players in the dataset. 

## License
This project is for educational purposes. Please respect Kaggle's terms of service and the original dataset license.

