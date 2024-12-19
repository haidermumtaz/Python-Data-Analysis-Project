import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



player_stats_df = pd.read_csv('C:/Users/haide/Documents/GitHub/Python-Data-Analysis-Project/vctDataSet/Datasets/vct_2024/players_stats/players_stats.csv')



# Player Effectiveness and Consistency
# Finding the players with the highest Ratings, Kills:Deaths ratio, and Average Combat Score
# Get top 10 players by rating
top_rated_players = (
    player_stats_df[['Player', 
                     'Teams', 
                     'Rating',
                     'Kills:Deaths', 
                     'Average Combat Score']]
    .sort_values(by=['Rating'], ascending=False)
    .drop_duplicates(subset=['Player'])
    .head(10)
)

print(top_rated_players)

# plt.hist(top_rated_players['Rating'])

#top_rated_players.to_csv('top_rated_players.csv')
#top_rated_players.to_excel('top_rated_players.xlsx', sheet_name='Sheet1', index=False)

#First Engagements and Clutch Factor
# Finding players with high First Kill and Clutch Success ratios
first_engagements = player_stats_df[['Player', 'Teams', 'First Kills', 'First Deaths', 'Clutch Success %']].copy()
first_engagements['First Engagement Ratio'] = first_engagements['First Kills'] / (first_engagements['First Deaths'] + 1e-9)
top_first_engagers = first_engagements.sort_values(by=['First Engagement Ratio'], ascending=False).drop_duplicates(subset=['Player']).head(10)

print(top_first_engagers)



# Data for the Top-Rated Players: Visualizing Rating and Kills:Deaths
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Top Players by Rating and First Engagement Performance")

# Plot 1: Top 10 Players by Rating with their K/D Ratios
sns.barplot(
    data = top_rated_players.sort_values("Rating", ascending=False),
    x = "Rating",
    y = "Player",
    hue = "Kills:Deaths",
    palette = "viridis",
    ax = axes[0]
)
axes[0].set_title("Top-Rated Players with K/D Ratios")
axes[0].set_xlabel("Player Rating")
axes[0].set_ylabel("Player")
axes[0].legend(title="Kills:Deaths Ratio")



plt.show()