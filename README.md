# CSE151A-Project — NFL Stats (1999–2022)

## Dataset
- **Source**: [NFL Stats (1999-2022) on Kaggle](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)
- **Description**: Comprehensive NFL player and team statistics covering weekly and yearly data from 1999 to 2022


# Major Preprocessing
There are many different positions and they all require a different set of values to calculate their fantasy value/projections, so we decided to just focus on one and try to get our supervised learning model to predict projected values for the next season based on previous season's values. 

We decided to go with the Running Back position, as it is one of the shortest lived positions in the NFL, but one of the most valuable in fantasy, due to the running back getting rushes, receptions, rushing TDs, receiving TDs, fumbles, and lots of snaps on offense. We figured that if we were able to generalize the regression model for this position, then being able to do the same for other positions wouldn't be as challenging.

<img width="1070" height="632" alt="image" src="https://github.com/user-attachments/assets/434ecccc-2a20-4b6e-8943-3b0d5e8b040a" />

The above is a graph depicting the relationship of previous year's rushing yards to the fantasy points for the following year. We can see a linear trend in the data with lots of variance. Some notable outliers in this include Christian McCaffrey, who suffered a patellar tendon injury and was unable to play for much of the season, despite having been the consensus best running back in the league the previous season due to his production as the engine of the 49ers offense. Another outlier is Chase Brown for the Cincinatti Bengals, who happened to lose their starting running back Joe Mixon in free agency in 2024 to the Houston Texans. Chase Brown saw his snap counts increase greatly as a result. Many such outliers exist, with the state of the NFL and its players in constant flux, especially at the Running Back position where talent is plenty, but careers are short lived.

<img width="1062" height="756" alt="image" src="https://github.com/user-attachments/assets/8c1118bf-2546-4cb6-854f-f2abe17305e8" />
The above is a grid of graphs depicting the relationship between snap count (how many times a player was on the field over the course of the season), Rushing Yards, Rushing Touchdowns, and Yards Per Carry (How many total Yards / How many rushing attempts). Many of these variables have a vaguely linear correlation with 2024 fantasy points, except for Yards Per Carry, which is to be expected as it can be biased towards small sample sizes. 
