import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


player_stats_df = pd.read_csv('C:/Users/haide/Documents/GitHub/Python-Data-Analysis-Project/vctDataSet/Datasets/vct_2024/players_stats/players_stats.csv')



# 1. Player Effectiveness and Consistency
# Finding the players with the highest Ratings, Kills:Deaths ratio, and Average Combat Score
top_rated_players = player_stats_df[['Player', 'Teams', 'Rating', 'Kills:Deaths', 'Average Combat Score']].sort_values(by=['Rating'], ascending=False).drop_duplicates(subset=['Player']).head(10)

print(top_rated_players)

# plt.hist(top_rated_players['Rating'])

#top_rated_players.to_csv('top_rated_players.csv')
#top_rated_players.to_excel('top_rated_players.xlsx', sheet_name='Sheet1', index=False)

# 2. First Engagements and Clutch Factor
# Finding players with high First Kill and Clutch Success ratios
first_engagements = player_stats_df[['Player', 'Teams', 'First Kills', 'First Deaths', 'Clutch Success %']].copy()
first_engagements['First Engagement Ratio'] = first_engagements['First Kills'] / (first_engagements['First Deaths'] + 1e-9)
top_first_engagers = first_engagements.sort_values(by=['First Engagement Ratio'], ascending=False).drop_duplicates(subset=['Player']).head(10)

print(top_first_engagers)