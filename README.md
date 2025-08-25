# CSE151A-Project — NFL Stats (1999–2022)

## Dataset
- **Source**: [NFL Stats (1999-2022) on Kaggle](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)
- **Description**: Comprehensive NFL player and team statistics covering weekly and yearly data from 1999 to 2022


# Major Preprocessing
There are many different positions and they all require a different set of values to calculate their fantasy value/projections, so we decided to just focus on one and try to get our supervised learning model to predict projected values for the next season based on previous season's values. 

We decided to go with the Running Back position, as it is one of the shortest lived positions in the NFL, but one of the most valuable in fantasy, due to the running back getting rushes, receptions, rushing TDs, receiving TDs, fumbles, and lots of snaps on offense. We figured that if we were able to generalize the regression model for this position, then being able to do the same for other positions wouldn't be as challenging.

For Our Preprocessing, we just filtered the yearly offense stats to the running back position, and removed all irrelevant columns, keeping only the player name, the season, total rushing yards per season, total rushing touchdowns per season, total snaps on offense per season, and Yards Per Carry per season. Additionally, we had to filter by season type, and remove all Post Season stats since they don't factor into fantasy football and only the regular season does. After this was done, we were ready to begin our model training based on our data.

<img width="1070" height="632" alt="image" src="https://github.com/user-attachments/assets/434ecccc-2a20-4b6e-8943-3b0d5e8b040a" />

The above is a graph depicting the relationship of previous year's rushing yards to the fantasy points for the following year. We can see a linear trend in the data with lots of variance. Some notable outliers in this include Christian McCaffrey, who suffered a patellar tendon injury and was unable to play for much of the season, despite having been the consensus best running back in the league the previous season due to his production as the engine of the 49ers offense. Another outlier is Chase Brown for the Cincinatti Bengals, who happened to lose their starting running back Joe Mixon in free agency in 2024 to the Houston Texans. Chase Brown saw his snap counts increase greatly as a result. Many such outliers exist, with the state of the NFL and its players in constant flux, especially at the Running Back position where talent is plenty, but careers are short lived.

<img width="1062" height="756" alt="image" src="https://github.com/user-attachments/assets/8c1118bf-2546-4cb6-854f-f2abe17305e8" />
The above is a grid of graphs depicting the relationship between snap count (how many times a player was on the field over the course of the season), Rushing Yards, Rushing Touchdowns, and Yards Per Carry (How many total Yards / How many rushing attempts). Many of these variables have a vaguely linear correlation with 2024 fantasy points, except for Yards Per Carry, which is to be expected as it can be biased towards small sample sizes. 

<img width="1191" height="800" alt="image" src="https://github.com/user-attachments/assets/7c678fbc-215d-4b44-85d8-d05497e3b7e9" />
We used Support Vector Regression to try and find a prediction line to fit the data we had, in order to predict the stats of a player the next season depending on the various stats we had calculated from the previous season. The test set size was 20% of our total set of data of running backs. It also had notable outliers including Christian McCaffrey, and Chase Brown, as well as Saquon Barkley who saw one of the best historical seasons at Running Back after signing with a new team that catered much better to his strengths than his previous team.
The error reportings are as follows: 
Train RMSE: 69.9235350510531
Test RMSE: 97.70489967679828
Train R2: 0.47908500488083483
Test R2: -0.025677153402659858

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

While these values were lower than our previous hyperparameters, the samples were all very close to the line and saw very few outliers, so we decided to tweak the sample and found that the RMSE was increased drastically for this model when we did. We decided to settle on the parameters {'C': 10, 'epsilon': 1.0, 'gamma': 'scale', 'kernel': 'rbf'} as it was the one that best fit our data.
