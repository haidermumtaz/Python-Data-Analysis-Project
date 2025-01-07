import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter


player_stats_df = pd.read_csv('C:/Users/haide/Documents/GitHub/Python-Data-Analysis-Project/vctDataSet/Datasets/vct_2024/players_stats/players_stats.csv')

player_stats = player_stats_df.groupby('Player').agg({
    'Teams': 'last',  
    'Rating': 'mean',
    'Average Combat Score': 'mean',
    'Kills:Deaths': 'mean',
    'Headshot %': lambda x: x.str.rstrip('%').astype(float).mean(),
    'Average Damage Per Round': 'mean',
    'Kills Per Round': 'mean'
}).round(2)


def create_table(data, target_column, title):
    # Reorder columns to have the target variable first after Player and Team
    reordered_columns = ['Player', 'Teams', target_column] + [col for col in data.columns if col not in ['Player', 'Teams', target_column]]
    data_reordered = data.reset_index()[reordered_columns]

    # Create a mapping for column name replacements
    column_mapping = {
        'Average Damage Per Round': 'ADR',
        'Kills Per Round': 'KPR',
        'Teams': 'Team',
        'Average Combat Score': 'ACS',
        'Kills:Deaths': 'K/D'
    }

    # Replace column labels using the mapping
    col_labels = [column_mapping.get(col, col) for col in reordered_columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()
    table = ax.table(
        cellText=data_reordered.values,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    num_columns = len(data_reordered.columns)
    for i in range(num_columns):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')
    for j in range(1, len(data_reordered) + 1):
        for i in range(num_columns):
            if j % 2 == 0:
                table[(j, i)].set_facecolor('#f2f2f2')
            else:
                table[(j, i)].set_facecolor('#ffffff')
    plt.title(title, pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()



top_players_rating = player_stats.sort_values('Rating', ascending=False).head(10)

# Sort by ACS
top_players_acs = player_stats.sort_values('Average Combat Score', ascending=False).head(10)

# Sort by K/D ratio
top_players_kd = player_stats.sort_values('Kills:Deaths', ascending=False).head(10)

# Create table for top players by Rating
create_table(top_players_rating, 'Rating', 'Top 10 Players by Average Rating')

# Create table for top players by ACS
create_table(top_players_acs, 'Average Combat Score', 'Top 10 Players by Average Combat Score')

# Create table for top players by K/D ratio
create_table(top_players_kd, 'Kills:Deaths', 'Top 10 Players by Average K/D Ratio')

# Extract player names from each table
players_rating = set(top_players_rating.index)
players_acs = set(top_players_acs.index)
players_kd = set(top_players_kd.index)

# Combine all players into a single list
all_players = list(players_rating) + list(players_acs) + list(players_kd)

# Count occurrences of each player
player_counts = Counter(all_players)

# Find players that appear more than once
common_players = [player for player, count in player_counts.items() if count > 1]

# Display the common players
print("Players appearing in more than one category:")
print(common_players)

# Optionally, create a DataFrame to display their stats
common_players_stats = player_stats.loc[common_players]
print(common_players_stats)

# Create a table for common players
create_table(common_players_stats, 'Rating', 'Highest-Performing and Most Consistent Players')

# Analyze clutch performance and entry impact
clutch_stats = player_stats_df.groupby('Player').agg({
    'Clutch Success %': lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce').mean()

}).round(3)



# Sort by clutch success
top_clutch_players = clutch_stats.sort_values('Clutch Success %', ascending=False).head(10)

def create_clutch_table(data, title):
    # Merge team information with clutch stats
    data_with_team = data.merge(
        player_stats[['Teams']], 
        left_index=True, 
        right_index=True
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()
    
    # Update column labels to include Team
    col_labels = ['Player', 'Team', 'Clutch Success %']
    
    # Reorder columns to match the labels
    data_display = data_with_team.reset_index()[['Player', 'Teams', 'Clutch Success %']]
    
    # Create the table
    table = ax.table(
        cellText=data_display.values,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    num_columns = len(data_display.columns)
    
    # Color header row
    for i in range(num_columns):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')
    
    # Alternate row colors for better readability
    for j in range(1, len(data_display) + 1):
        for i in range(num_columns):
            if j % 2 == 0:
                table[(j, i)].set_facecolor('#f2f2f2')  # Light gray for even rows
            else:
                table[(j, i)].set_facecolor('#ffffff')  # White for odd rows
    
    # Add title
    plt.title(title, pad=20, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Create table for top clutch players
create_clutch_table(top_clutch_players, 'Top 10 Players by Average Clutch Success %')

