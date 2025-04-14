import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


def prepare_data_for_regression(data, win_rates_per_map):
    regression_data = []
    for index, row in data.iterrows():
        map_name = row['Map']
        team_a = row['Team A']
        team_b = row['Team B']
        score_a = row['Team A Score']
        score_b = row['Team B Score']

        # get win rates (handle cases where a team might not have played on a map)
        win_rate_a = win_rates_per_map.get(team_a, {}).get(map_name, 0)
        win_rate_b = win_rates_per_map.get(team_b, {}).get(map_name, 0)

        # create features: win rate difference, average score difference
        win_rate_diff = win_rate_a - win_rate_b
        score_diff = score_a - score_b

        # target variable: 1 if Team A wins, -1 if Team B wins, 0 if draw
        if score_a > score_b:
            winner = 1
        elif score_b > score_a:
            winner = -1
        else:
            winner = 0

        regression_data.append([win_rate_diff, score_diff, winner, map_name, team_a, team_b])

    regression_df = pd.DataFrame(regression_data, columns=['win_rate_difference', 'score_difference', 'winner', 'map', 'team_a', 'team_b'])
    return regression_df

def train_linear_regression_model(df):
    # select features and target
    features = ['win_rate_difference', 'score_difference']
    target = 'winner'

    # handle potential NaN values (e.g., if a team has no prior games)
    df_cleaned = df.dropna(subset=features + [target])

    if df_cleaned.empty:
        print("Warning: No valid data for linear regression after handling missing values.")
        return None

    X = df_cleaned[features]
    y = df_cleaned[target]

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = model.predict(X_test)

    # evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n--- Linear Regression Model ---")
    print(f"Mean Squared Error on Test Set: {mse:.2f}")

    # return the trained model and the test data
    return model, X_test, y_test

def predict_with_linear_regression(model, win_rate_diff, score_diff):
    if model is None:
        print("Error: Linear regression model not trained.")
        return None

    # prepare the input data as a DataFrame
    input_data = pd.DataFrame({'win_rate_difference': [win_rate_diff], 'score_difference': [score_diff]})

    # make the prediction
    prediction = model.predict(input_data)[0]
    return prediction

def visualize_regression_comparison(team_a, team_b, map_names, win_rate_diffs, predictions):
    n_maps = len(map_names)
    index = np.arange(n_maps)
    bar_width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Map')
    ax1.set_ylabel('Win Rate Difference (Team A - Team B)', color=color)
    rects1 = ax1.bar(index, win_rate_diffs, bar_width, label='Win Rate Difference', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(index)
    ax1.set_xticklabels(map_names, rotation=45, ha='right')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Linear Regression Prediction Value', color=color)  # we already handled the x-label with ax1
    line2 = ax2.plot(index + bar_width / 2, predictions, marker='o', linestyle='-', color=color, label='LR Prediction')
    ax2.tick_params(axis='y', labelcolor=color)

    # add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()  # otherwise the right y-label might be slightly clipped
    plt.title(f'Win Rate Difference and Linear Regression Prediction: {team_a} vs {team_b}')
    plt.savefig(f'regression_comparison_{team_a}_vs_{team_b}.png')
    print(f"\nRegression comparison chart saved as 'regression_comparison_{team_a}_vs_{team_b}.png'")
    plt.close()


def interpret_prediction_value(prediction_value):
    if prediction_value > 0.2:
        return "Team A is likely to win"
    elif prediction_value < -0.2:
        return "Team B is likely to win"
    else:
        return "It's a close match"


if __name__ == "__main__":
    file_path = 'maps_scores.csv'  # specify your CSV file path

    # run the team win rate comparison and plotting
    load_and_validate_data(file_path)

    # prepare data for linear regression
    data = load_and_validate_data(file_path)
    win_rates_per_map = calculate_win_rates_per_map(data)
    all_teams = sorted(set(data["Team A"].tolist() + data["Team B"].tolist()))

    if len(all_teams) >= 2:
        print("\n--- Linear Regression Analysis ---")

        # let the user select two teams for linear regression-based prediction
        team_a_predict = select_team(data, "Team A for linear regression prediction")
        team_b_predict = select_team(data, "Team B for linear regression prediction")

        # get win rates for the selected teams across all maps
        unique_maps = sorted(data['Map'].unique())
        win_rate_team_a_predict = {map_name: win_rates_per_map.get(team_a_predict, {}).get(map_name, 0) for map_name in unique_maps}
        win_rate_team_b_predict = {map_name: win_rates_per_map.get(team_b_predict, {}).get(map_name, 0) for map_name in unique_maps}

        # prepare regression data (using all matches for training)
        regression_df = prepare_data_for_regression(data, win_rates_per_map)

        # train the linear regression model
        linear_regression_model, _, _ = train_linear_regression_model(regression_df)

        if linear_regression_model is not None:
            print(f"\n--- Linear Regression Prediction: {team_a_predict} vs {team_b_predict} ---")
            map_names_for_plot = []
            win_rate_diffs_for_plot = []
            predictions_for_plot = []

            for map_name in unique_maps:
                wr_a = win_rate_team_a_predict.get(map_name, 0)
                wr_b = win_rate_team_b_predict.get(map_name, 0)
                wr_diff = wr_a - wr_b
                # for prediction, we don't have the future score difference, so we use 0
                prediction_value = predict_with_linear_regression(linear_regression_model, wr_diff, 0)
                prediction_label = interpret_prediction_value(prediction_value)

                map_names_for_plot.append(map_name)
                win_rate_diffs_for_plot.append(wr_diff)
                predictions_for_plot.append(prediction_value)

                print(f"On map '{map_name}': Win Rate Diff ({team_a_predict} - {team_b_predict}) = {wr_diff:.2f}, Linear Regression Prediction Value = {prediction_value:.2f} ({prediction_label})")

            # generate comparison chart
            visualize_regression_comparison(team_a_predict, team_b_predict, map_names_for_plot, win_rate_diffs_for_plot, predictions_for_plot)

        else:
            print("Linear regression model could not be trained.")

    else:
        print("\nNot enough teams in the data to perform linear regression comparison.")