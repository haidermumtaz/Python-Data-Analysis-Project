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


def create_generic_table(data, title, target):
    """
    Creates and displays a styled table from a given DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be displayed as a table.
    - title (str): The title of the table.
    - target (str): The column name to prioritize and place next to 'Player' and 'Teams'.
    """
    # Reset index to ensure 'Player' or other index columns are included if present
    data_reset = data.reset_index()

    # Identify key columns to prioritize
    key_columns = ['Player', 'Teams']
    existing_key_columns = [col for col in key_columns if col in data_reset.columns]

    # Include the target column if it exists
    if target in data_reset.columns:
        existing_key_columns.append(target)
    else:
        print(f"Warning: Target column '{target}' not found in DataFrame.")

    # Determine the remaining columns
    remaining_columns = [col for col in data_reset.columns if col not in existing_key_columns]

    # Reorder columns: key columns first (Player, Teams, Target), then the rest
    reordered_columns = existing_key_columns + remaining_columns
    data_reordered = data_reset[reordered_columns]

    # Create a mapping for column name replacements (extend as needed)
    column_mapping = {
        'Average Damage Per Round': 'ADR',
        'Kills Per Round': 'KPR',
        'Teams': 'Team',
        'Average Combat Score': 'ACS',
        'Kills:Deaths': 'K/D',
        'Clutch Success %': 'Clutch %'
        # Add more mappings here if needed
    }

    # Replace column labels using the mapping
    col_labels = [column_mapping.get(col, col) for col in reordered_columns]

    # Initialize the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()

    # Create the table
    table = ax.table(
        cellText=data_reordered.values,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    # Set table styling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    num_columns = len(data_reordered.columns)

    # Style the header row
    for i in range(num_columns):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')

    # Alternate row colors for better readability
    for j in range(1, len(data_reordered) + 1):
        for i in range(num_columns):
            if j % 2 == 0:
                table[(j, i)].set_facecolor('#f2f2f2')  # Light gray for even rows
            else:
                table[(j, i)].set_facecolor('#ffffff')  # White for odd rows

    # Add title to the table
    plt.title(title, pad=20, fontsize=14)

    # Adjust layout and display the table
    plt.tight_layout()
    plt.show()



top_players_rating = player_stats.sort_values('Rating', ascending=False).head(10)

# Sort by ACS
top_players_acs = player_stats.sort_values('Average Combat Score', ascending=False).head(10)

# Sort by K/D ratio
top_players_kd = player_stats.sort_values('Kills:Deaths', ascending=False).head(10)

# # Create table for top players by Rating
# create_generic_table(top_players_rating, 'Top 10 Players by Average Rating', 'Rating')

# # Create table for top players by ACS
# create_generic_table(top_players_acs, 'Top 10 Players by Average Combat Score', 'Average Combat Score')

# # Create table for top players by K/D ratio
# create_generic_table(top_players_kd, 'Top 10 Players by Average K/D Ratio', 'Kills:Deaths')

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
# print("Players appearing in more than one category:")
# print(common_players)

# Optionally, create a DataFrame to display their stats
common_players_stats = player_stats.loc[common_players]
print(common_players_stats)

# Create a table for common players
create_generic_table(common_players_stats, 'Highest-Performing and Most Consistent Players', 'Rating')

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
create_generic_table(top_clutch_players, 'Top 10 Players by Average Clutch Success %', 'Clutch Success %')

def create_generic_bar(data, title, target):
    """
    Creates and displays a styled bar graph from a given DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be displayed as a bar graph.
    - title (str): The title of the graph.
    - target (str): The column name to use for the bar values.
    """
    # Reset index if it contains the player names
    data_reset = data.reset_index()

    # Create a mapping for column name replacements
    column_mapping = {
        'Average Damage Per Round': 'ADR',
        'Kills Per Round': 'KPR',
        'Teams': 'Team',
        'Average Combat Score': 'ACS',
        'Kills:Deaths': 'K/D',
        'Clutch Success %': 'Clutch %'
        # Add more mappings here if needed
    }

    # Get the target column label
    target_label = column_mapping.get(target, target)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    bars = ax.bar(
        data_reset['Player'],
        data_reset[target],
        color='#4472C4'
    )

    # Customize the plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xlabel('Player', fontsize=10)
    ax.set_ylabel(target_label, fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add team labels above each bar
    if 'Teams' in data_reset.columns:
        for idx, bar in enumerate(bars):
            team = data_reset['Teams'].iloc[idx]
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                team,
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=0
            )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height/2,  # Position in middle of bar
            f'{height:.2f}',
            ha='center',
            va='center',
            color='white',
            fontsize=9
        )

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Place grid lines behind bars

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

create_generic_bar(top_clutch_players, 'Top 10 Players by Clutch Success %', 'Clutch Success %')


# create_generic_bar(top_players_rating, 'Top Players by Rating', 'Rating')

# # Create table for top players by Rating


# # Create table for top players by ACS
# create_generic_bar(top_players_acs, 'Top 10 Players by Average Combat Score', 'Average Combat Score')

# # Create table for top players by K/D ratio
# create_generic_bar(top_players_kd, 'Top 10 Players by Average K/D Ratio', 'Kills:Deaths')

def create_radar_chart(data, title):
    """
    Creates a radar chart to visualize player performance across multiple statistics.
    Values are normalized to a 0-1 scale for fair comparison.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing player statistics
    - title (str): Title for the visualization
    """
    # Select the metrics we want to compare (excluding Teams)
    metrics = ['Rating', 'Average Combat Score', 'Kills:Deaths', 
               'Headshot %', 'Average Damage Per Round', 'Kills Per Round']
    
    # Create a copy of the data for normalization
    plot_data = data[metrics].copy()
    
    # Normalize each metric to 0-1 scale
    for metric in metrics:
        min_val = plot_data[metric].min()
        max_val = plot_data[metric].max()
        plot_data[metric] = (plot_data[metric] - min_val) / (max_val - min_val)
    
    # Number of metrics
    num_metrics = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
    
    # Plot for each player
    for idx, player in enumerate(data.index):
        # Get player's normalized values
        values = plot_data.loc[player].values.flatten().tolist()
        values += values[:1]  # Complete the circle
        
        # Plot the player's line
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{player} ({data.loc[player, 'Teams']})")
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each metric and label them
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric if metric not in ['Average Combat Score', 'Average Damage Per Round', 'Kills Per Round'] 
                        else {'Average Combat Score': 'ACS', 
                             'Average Damage Per Round': 'ADR',
                             'Kills Per Round': 'KPR'}[metric] 
                        for metric in metrics])
    
    # Set the ylim to [0, 1] since we normalized the data
    ax.set_ylim(0, 1)
    
    # Add gridlines at 0.2, 0.4, 0.6, and 0.8
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0)
    
    # Add legend
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    
    # Add title
    plt.title(title, y=1.05, fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()
# Create the radar chart for common players
create_radar_chart(common_players_stats, 'Performance Comparison of Top Players Across Categories')

def create_performance_heatmap(data, title):
    """
    Creates a heatmap to visualize player performance across multiple statistics.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing player statistics
    - title (str): Title for the visualization
    """
    # Select metrics for comparison (excluding Teams)
    metrics = ['Rating', 'Average Combat Score', 'Kills:Deaths', 
               'Headshot %', 'Average Damage Per Round', 'Kills Per Round']
    
    # Create a copy of the data with selected metrics
    plot_data = data[metrics].copy()
    
    # Normalize the data for each metric (0-1 scale)
    for column in plot_data.columns:
        plot_data[column] = (plot_data[column] - plot_data[column].min()) / \
                           (plot_data[column].max() - plot_data[column].min())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(plot_data, 
                annot=True, 
                cmap='RdYlBu_r',
                fmt='.2f',
                cbar_kws={'label': 'Normalized Score'},
                yticklabels=[f"{player}\n({data.loc[player, 'Teams']})" 
                            for player in plot_data.index])
    
    # Customize the plot
    plt.title(title, pad=20, fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Create the heatmap
create_performance_heatmap(common_players_stats, 'Performance Heatmap of Top Players')
