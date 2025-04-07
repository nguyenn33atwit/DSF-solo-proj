import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def load_and_validate_data(file_path):
    # load the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    # define the list of columns that are expected in the DataFrame
    required_columns = ["Team A", "Team B", "Team A Score", "Team B Score", "Map"]
    # identify any columns that are in the required list but missing from the DataFrame
    missing_columns = [col for col in required_columns if col not in data.columns]
    # if there are missing columns, raise a ValueError with a message indicating which columns are missing
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return data


def calculate_win_rates_per_map(data):
    # initialize an empty dictionary to store the win rates for each team
    win_rates = {}
    # get a sorted list of all unique team names from both 'Team A' and 'Team B' columns
    all_teams = sorted(set(data["Team A"].tolist() + data["Team B"].tolist()))
    # iterate through each unique team
    for team in all_teams:
        # initialize an empty dictionary to store the win rates for the current team
        win_rates[team] = {}
        # filter the DataFrame to get matches where the current team is in 'Team A'
        team_a_matches = data[data["Team A"] == team]
        # filter the DataFrame to get matches where the current team is in 'Team B'
        team_b_matches = data[data["Team B"] == team]
        # iterate through each unique map name in the data
        for map_name in data["Map"].unique():
            # filter the 'Team A' matches for the current map
            map_team_a = team_a_matches[team_a_matches["Map"] == map_name]
            # filter the 'Team B' matches for the current map
            map_team_b = team_b_matches[team_b_matches["Map"] == map_name]
            # calculate the total number of matches played by the current team on the current map
            total_matches = len(map_team_a) + len(map_team_b)
            # calculate the number of wins for the current team on the current map
            # a win occurs when a team's 'Score' is greater than the other team
            wins = len(map_team_a[map_team_a["Team A Score"] > map_team_a["Team B Score"]]) + \
                   len(map_team_b[map_team_b["Team B Score"] > map_team_b["Team A Score"]])
            # calculate the win rate. If there are no matches, the win rate is 0
            win_rates[team][map_name] = (wins / total_matches) if total_matches > 0 else 0
    # return the dictionary containing win rates for each team on each map
    return win_rates


def select_team(data, prompt):
    # get a sorted list of all unique teams from both 'Team A' and 'Team B' columns
    all_teams = sorted(set(data["Team A"].tolist() + data["Team B"].tolist()))
    # print the available teams with their corresponding numbers for user selection
    print(f"\nAvailable teams for {prompt}:")
    for i, team in enumerate(all_teams, 1):
        print(f"{i}. {team}")
    # start a loop to continuously prompt the user until a valid selection is made
    while True:
        try:
            # get the user's input as a number and adjust it to be 0-based index
            team_idx = int(input(f"Enter team number for {prompt}: ")) - 1
            return all_teams[team_idx]
        except ValueError:
            # if the user enters a wrong value, print an error message
            print("Invalid team selection.")

def calculate_match_results(data, selected_team):
    # filter the DataFrame to get matches where the selected team is in 'Team A' and create a copy
    team_a_matches = data[data["Team A"] == selected_team].copy()
    # filter the DataFrame to get matches where the selected team is in 'Team B' and create a copy
    team_b_matches = data[data["Team B"] == selected_team].copy()

    # apply the combined determine_result function to 'team_a_matches'
    team_a_matches['Result'] = team_a_matches.apply(lambda row: determine_result(row, selected_team), axis=1)

    # apply the combined determine_result function to 'team_b_matches'
    team_b_matches['Result'] = team_b_matches.apply(lambda row: determine_result(row, selected_team), axis=1)

    return team_a_matches, team_b_matches


def determine_result(row, selected_team):
    if row['Team A'] == selected_team:
        if row['Team A Score'] > row['Team B Score']:
            return 'Win'
        elif row['Team A Score'] == row['Team B Score']:
            return 'Draw'
        else:
            return 'Loss'
    elif row['Team B'] == selected_team:
        if row['Team B Score'] > row['Team A Score']:
            return 'Win'
        elif row['Team B Score'] == row['Team A Score']:
            return 'Draw'
        else:
            return 'Loss'


def check_special_cases(team_name, team_matches, opponent_position, data, win_rates_per_map):
    # initialize an empty list to store the descriptions of special cases
    special_cases = []
    # iterate through each match in the provided team's matches
    for index, row in team_matches.iterrows():
        # check if the result of the match was a 'Win' for the team
        if row['Result'] == 'Win':
            # get the name of the opponent team from the specified column
            opponent_team = row[opponent_position]
            # get the name of the map on which the match was played
            current_map = row['Map']
            # get the win rate of the current team on the current map from the provided dictionary
            team_win_rate = win_rates_per_map.get(team_name, {}).get(current_map, 0)
            # get the win rate of the opponent team on the current map
            opponent_win_rate = win_rates_per_map.get(opponent_team, {}).get(current_map, 0)
            # calculate the total number of matches played by the current team on the current map
            team_matches_count = len(data[(data['Map'] == current_map) &
                                          ((data['Team A'] == team_name) | (data['Team B'] == team_name))])
            # calculate the total number of matches played by the opponent team on the current map
            opponent_matches_count = len(data[(data['Map'] == current_map) &
                                              ((data['Team A'] == opponent_team) | (data['Team B'] == opponent_team))])
            # check if both teams have played at least one match on the current map
            # and if the opponent's win rate is strictly greater than the current team's win rate
            if (team_matches_count > 0 and opponent_matches_count > 0 and
                    opponent_win_rate > team_win_rate):
                # if the conditions are met, add a description of the special case to the list
                special_cases.append(
                    f"{team_name} beat {opponent_team} on {current_map} "
                    f"(Opponent's win rate: {opponent_win_rate:.2f} > {team_name}'s win rate: {team_win_rate:.2f})"
                )

    # return the list of special case descriptions
    return special_cases


def has_special_case(team_name, map_name, data, win_rates_per_map):
    # filter the data to only include matches played on the specified map
    filtered_data = data[data['Map'] == map_name]
    # calculate the match results for the given team on the filtered data
    team_a_matches, team_b_matches = calculate_match_results(filtered_data, team_name)

    # initialize an empty list to store special cases
    special_cases = []
    # check for special cases where the team was 'Team A'
    special_cases.extend(check_special_cases(team_name, team_a_matches, 'Team B', data, win_rates_per_map))
    # check for special cases where the team was 'Team B'
    special_cases.extend(check_special_cases(team_name, team_b_matches, 'Team A', data, win_rates_per_map))

    # return True if the list of special cases is not empty, indicating at least one special case exists
    return len(special_cases) > 0


def plot_team_comparison(team1, team2, all_maps, win_rates_per_map, special_cases_team1, special_cases_team2):
    # create a new figure and a set of subplots
    plt.figure(figsize=(14, 10))

    # get the total number of maps
    n_maps = len(all_maps)
    # create an array of indices for the x-axis positions of the bars
    index = np.arange(n_maps)
    # define the width of the bars
    bar_width = 0.35

    # create the bar plot for the win rates of the first team
    team1_bars = plt.bar(index,
                         [win_rates_per_map.get(team1, {}).get(map_name, 0) for map_name in all_maps],
                         bar_width,
                         label=f'{team1} Win Rate',
                         color='skyblue')

    # create the bar plot for the win rates of the second team, offset by the bar width
    team2_bars = plt.bar(index + bar_width,
                         [win_rates_per_map.get(team2, {}).get(map_name, 0) for map_name in all_maps],
                         bar_width,
                         label=f'{team2} Win Rate',
                         color='lightcoral')

    # add visual indicators for special cases by adding a small bar on top of the existing win rate bar
    for i, map_name in enumerate(all_maps):
        # if team1 has a special case on the current map, add a blue bar on top
        if special_cases_team1.get(map_name, False):
            plt.bar(i, 0.1, bar_width, bottom=win_rates_per_map.get(team1, {}).get(map_name, 0),
                    color='blue', alpha=0.7, label='_nolegend_')

        # if team2 has a special case on the current map, add a red bar on top
        if special_cases_team2.get(map_name, False):
            plt.bar(i + bar_width, 0.1, bar_width, bottom=win_rates_per_map.get(team2, {}).get(map_name, 0),
                    color='red', alpha=0.7, label='_nolegend_')

    # set the labels for the x and y axes
    plt.xlabel('Maps')
    plt.ylabel('Win Rate')
    # set the title of the plot
    plt.title(f'Win Rates Comparison: {team1} vs {team2}')
    # set the x-axis tick positions and labels, rotating the labels for better readability
    plt.xticks(index + bar_width / 2, all_maps, rotation=45, ha='right')
    # set the y-axis limits to ensure the special case indicators are visible
    plt.ylim(0, 1.2)

    # create custom legend elements to explain the different parts of the plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label=f'{team1} Base Win Rate'),
        Patch(facecolor='blue', alpha=0.7, edgecolor='black', label=f'{team1} Special Case Boost'),
        Patch(facecolor='lightcoral', edgecolor='black', label=f'{team2} Base Win Rate'),
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label=f'{team2} Special Case Boost'),
    ]
    # add the legend to the plot
    plt.legend(handles=legend_elements, loc='upper right')

    # adjust the plot layout to prevent labels from being cut off
    plt.tight_layout()
    # add a grid to the y-axis for easier comparison of win rates
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # save the generated plot as a PNG file
    plt.savefig('team_comparison.png')
    print("\nComparison plot saved as 'team_comparison.png'")
    # close the plot to free up memory
    plt.close()


def compare_team_winrates_per_map(file_path):
    # load and validate the data from the specified file path
    data = load_and_validate_data(file_path)
    # calculate the win rates for each team on each map
    win_rates_per_map = calculate_win_rates_per_map(data)

    # select the first team for comparison
    team1 = select_team(data, "Team 1")
    # select the second team for comparison
    team2 = select_team(data, "Team 2")

    # get a sorted list of all unique map names from the data
    all_maps = sorted(data["Map"].unique())
    # print a message indicating which teams are being compared
    print(f"\nComparing {team1} and {team2} across all maps:")

    # initialize dictionaries to store predictions and special case information for each team and map
    predictions = {}
    special_cases_team1 = {}
    special_cases_team2 = {}

    # iterate through each map to compare the teams' win rates and check for special cases
    for map_name in all_maps:
        # get the win rate of the first team on the current map
        win_rate_team1 = win_rates_per_map.get(team1, {}).get(map_name, 0)
        # get the win rate of the second team on the current map
        win_rate_team2 = win_rates_per_map.get(team2, {}).get(map_name, 0)

        # check if the first team has a special case (beat a higher-ranked opponent) on the current map
        has_special_case_team1 = has_special_case(team1, map_name, data, win_rates_per_map)
        # check if the second team has a special case on the current map
        has_special_case_team2 = has_special_case(team2, map_name, data, win_rates_per_map)

        # store whether each team has a special case on the current map
        special_cases_team1[map_name] = has_special_case_team1
        special_cases_team2[map_name] = has_special_case_team2

        # adjust the win rates by adding a 10% boost if the team has a special case on the map
        adjusted_win_rate_team1 = win_rate_team1 + (0.1 if has_special_case_team1 else 0)
        adjusted_win_rate_team2 = win_rate_team2 + (0.1 if has_special_case_team2 else 0)

        # print the win rates for both teams on the current map, indicating if a special case applies
        print(f"\n--- Map: {map_name} ---")
        print(
            f"{team1} Win Rate: {win_rate_team1:.2f}{' (+10% due to special case)' if has_special_case_team1 else ''}")
        print(
            f"{team2} Win Rate: {win_rate_team2:.2f}{' (+10% due to special case)' if has_special_case_team2 else ''}")

        # predict the winner on the current map based on the adjusted win rates
        if adjusted_win_rate_team1 > adjusted_win_rate_team2:
            predictions[map_name] = team1
            print(f"Prediction: {team1} is likely to win on {map_name}.")
        elif adjusted_win_rate_team2 > adjusted_win_rate_team1:
            predictions[map_name] = team2
            print(f"Prediction: {team2} is likely to win on {map_name}.")
        else:
            predictions[map_name] = "Draw (or very close)"
            print(f"Prediction: It's a close call between {team1} and {team2} on {map_name}.")

    # print an overall summary of the map predictions
    print("\n--- Overall Map Predictions ---")
    for map_name, predicted_winner in predictions.items():
        print(f"On {map_name}: {predicted_winner}")

    # generate a visual comparison of the two teams' win rates across all maps
    plot_team_comparison(team1, team2, all_maps, win_rates_per_map, special_cases_team1, special_cases_team2)


compare_team_winrates_per_map('maps_scores.csv')