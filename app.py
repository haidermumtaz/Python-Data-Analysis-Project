import numpy as np
import pandas as pd


player_stats_df = pd.read_csv('C:/Users/haide/Documents/GitHub/Python-Data-Analysis-Project/vctDataSet/Datasets/vct_2024/players_stats/players_stats.csv')



# 1. Player Effectiveness and Consistency
# Extracting the players with the highest Ratings, Kills:Deaths ratio, and Average Combat Score
top_rated_players = player_stats_df[['Player', 'Teams', 'Rating', 'Kills:Deaths', 'Average Combat Score']].sort_values(by=['Rating'], ascending=False).drop_duplicates(subset=['Player']).head(10)

print(top_rated_players)