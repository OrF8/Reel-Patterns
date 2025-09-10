# ===============================================================
# Reel Patterns: Curtain Call, Please - Visualizations
#
# Requirements:
#   pip install -r requirements.txt
#
# Usage:
#   python curtain_call_visualizations.py
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from scipy.stats import norm
from constants import (
    ROI,
    AUDIENCE_RATING,
    CRITIC_RATING,
    TITLE_MAP,
    PROBABILITY_OF_SUCCESS_FIGURE_OUT_PATH_MAP,
    MOVIES_PER_SEQUEL_FIGURE_OUT_PATH,
    DPI,
    MAIN_COLOR,
    CI_COLOR,
    GROUP_ID,
    RELEASE_DATE,
    N_FILMS_IN_GROUP,
    INDEX_IN_GROUP,
    BEST_METRIC_VALUE_SO_FAR,
    PERCENT_OF_BEST_SO_FAR,
    AS_SUCCESSFUL_AS_BEST_SO_FAR,
    N_SUCCESSES,
    N_MOVIES,
    PROB_OF_SUCCESS_NAIVE,
    BAYESIAN_SHRUNKEN_PROB_OF_SUCCESS,
    CI_LOWER,
    CI_UPPER,
    Metric,
    ROTTEN_TOMATOES_PATH,
    WIKIDATA_PATH,
    IMDB_PATH
)

def prepare_dataframe_for_probability_of_success_calculation(data: pd.DataFrame, metric: Metric = ROI,
                                                             max_length_of_group: int = 6,
                                                             min_length_of_group: int = 0) -> pd.DataFrame:
    """
    Prepares a DataFrame for calculating the probability of success by filtering
    and cleaning the data based on specified criteria.

    - The function filters out groups that do not meet the specified length constraints.
    - Missing or infinite values in the `metric` column are not allowed.
    - Rows with missing values in the `GROUP_ID` column are dropped,
      and the DataFrame is sorted by `GROUP_ID` and `RELEASE_DATE`.

    :param data: The input DataFrame containing the data to be processed.
    :param metric: This column represents the metric to be analyzed.
    :param max_length_of_group: The maximum number of movies allowed in a group.
    :param min_length_of_group: The minimum number of movies required in a group.
    :return: A cleaned and filtered DataFrame ready for probability of success calculation.
    :raises: AssertionError if the `metric` column contains missing or infinite values.
    """
    df: pd.DataFrame = data.copy()
    # Keep only groups with at least min_length_of_group and at most max_length_of_group movies
    df = df[(df[N_FILMS_IN_GROUP] >= min_length_of_group) & (df[N_FILMS_IN_GROUP] <= max_length_of_group)]

    # Assert no missing or inf values in metric column
    assert (df[metric].notna().all() & (np.isfinite(df[metric])).all()),\
            "Metric column contains missing or infinite values"

    # Drop missing GROUPING_COL values and resort
    df = df[df[GROUP_ID].notna()].sort_values(by=[GROUP_ID, RELEASE_DATE])
    return df


def calculate_dataframe_for_probability_of_success(df: pd.DataFrame, metric: Metric = ROI,
                                                   tolerance_for_successful: int = 0,
                                                   m: int = 5, alpha: float = 0.15) -> pd.DataFrame:
    """
    Calculate a DataFrame summarizing the probability of success for movies in a franchise
    based on their performance metrics.
    This function computes various statistics, including naive and Bayesian estimates of
    the probability of success, as well as confidence intervals for these probabilities.

    - The function assumes that the input DataFrame is pre-sorted by group and index.
    - Rows without a previous best metric value
      (e.g., the first movie in each franchise) are excluded from the calculations.
    - The Wilson confidence interval is used for better accuracy in proportion estimates.

    :param df: Input DataFrame containing movie data.
               Must include columns for group IDs, metric values, and index within the group.
    :param metric: The performance metric to evaluate (e.g., ROI).
    :param tolerance_for_successful: The percentage gap tolerance
                                     for a movie to be considered as successful as the best-so-far.
    :param m: The prior weight for the Bayesian estimate.
    :param alpha: The significance level for the Wilson confidence interval.
    :return: A DataFrame with the following columns:
             - INDEX_IN_GROUP: The index within the group.
             - n_successes: The number of successful movies at each index.
             - n_movies: The total number of movies at each index.
             - _PROB_OF_SUCCESS_NAIVE: The naive estimate of the probability of success.
             - _BAYESIAN_SHRUNKEN_PROB_OF_SUCCESS: The Bayesian estimate of the probability of success.
             - _CI_LOWER: The lower bound of the Wilson confidence interval.
             - _CI_UPPER: The upper bound of the Wilson confidence interval.
    """
    # For each group at each index calculate the best metric value so far (before current movie)
    df[BEST_METRIC_VALUE_SO_FAR] = df.groupby(GROUP_ID, sort=False)[metric].cummax()
    df[BEST_METRIC_VALUE_SO_FAR] = df.groupby(GROUP_ID, sort=False)[BEST_METRIC_VALUE_SO_FAR].shift(1)

    # Calculate % gap vs best-so-far (positive means >= best-so-far)
    df[PERCENT_OF_BEST_SO_FAR] = 100.0 * (df[metric] / df[BEST_METRIC_VALUE_SO_FAR] - 1.0)

    # Keep only rows that have a previous best to compare to (i.e., not the first movie in each franchise)
    df = df[df[BEST_METRIC_VALUE_SO_FAR].notna()].copy()

    # Calculate indicator of whether the current movie is as successful as best-so-far (1 if yes, 0 if no)
    df.loc[:, AS_SUCCESSFUL_AS_BEST_SO_FAR] = (df[PERCENT_OF_BEST_SO_FAR] >= tolerance_for_successful).astype(int)

    # Aggregate per index: calculate number of successes and total number of movies
    agg_df: pd.DataFrame = df.groupby(INDEX_IN_GROUP, as_index=False).agg(
        n_successes=(AS_SUCCESSFUL_AS_BEST_SO_FAR, "sum"),
        n_movies=(AS_SUCCESSFUL_AS_BEST_SO_FAR, "count"),
    )

    K_i: pd.Series = agg_df[N_SUCCESSES]
    N_i: pd.Series = agg_df[N_MOVIES]

    # Calculate naive estimate of probability of success at each index
    agg_df[PROB_OF_SUCCESS_NAIVE] = K_i / N_i

    # Calculate mean success rate across all indices to use as prior mean
    P_mean: float = K_i.sum() / N_i.sum()

    # Calculate shrunken estimate of probability of success at each index
    agg_df[BAYESIAN_SHRUNKEN_PROB_OF_SUCCESS] = (K_i + m * P_mean) / (N_i + m)

    # Calculate Wilson confidence interval 85% (better for proportions than normal approx)
    z = norm.ppf(1 - alpha / 2)

    P_i = agg_df[PROB_OF_SUCCESS_NAIVE]
    density: float = 1.0 + (z**2) / N_i
    center: float = (P_i + (z**2) / (2.0 * N_i)) / density
    margin: float = (z * np.sqrt((P_i * (1.0 - P_i) + (z**2) / (4.0 * N_i)) / N_i)) / density

    agg_df[CI_LOWER] = np.clip(center - margin, 0.0, 1.0)
    agg_df[CI_UPPER] = np.clip(center + margin, 0.0, 1.0)

    return agg_df


def plot_probability_of_success(data: pd.DataFrame, metric: Metric = ROI,
                                max_length_of_group: int = 6, min_length_of_group: int = 0,
                                tolerance_for_successful: int = 0, m: int = 5,
                                alpha: float = 0.15, title: str = TITLE_MAP[ROI]) -> None:
    """
    Plots the probability of success for a given dataset and metric, with confidence intervals.
    This function visualizes the Bayesian shrunken probability of success for groups within
    the dataset, along with confidence intervals.
    The x-axis represents the index in the franchise, and the y-axis represents the probability of success.

    - The function uses a Bayesian approach to calculate the probability of success.
        - Confidence intervals are displayed as a shaded region in the plot.
        - The x-axis ticks are adjusted to show integer values corresponding to group indices.

    :param data: The input data containing the relevant metrics and group information.
    :param metric: The metric to evaluate the probability of success.
    :param max_length_of_group: The maximum length of groups to consider.
    :param min_length_of_group: The minimum length of groups to consider
    :param tolerance_for_successful: The tolerance level for defining success.
    :param m: The smoothing parameter for Bayesian shrinkage.
    :param alpha: The significance level for confidence intervals.
    :param title: The title of the plot.
    """

    df: pd.DataFrame = prepare_dataframe_for_probability_of_success_calculation(
        data, metric, max_length_of_group, min_length_of_group
    )
    success_prob_summary: pd.DataFrame = calculate_dataframe_for_probability_of_success(
        df, metric, tolerance_for_successful, m, alpha
    )

    X: pd.Series = success_prob_summary[INDEX_IN_GROUP]
    Y_SHRUNK: pd.Series = success_prob_summary[BAYESIAN_SHRUNKEN_PROB_OF_SUCCESS]
    CI_LO: pd.Series = success_prob_summary[CI_LOWER]
    CI_HI: pd.Series = success_prob_summary[CI_UPPER]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot CI as shaded region
    ax.fill_between(X + 1, CI_LO, CI_HI, alpha=0.2, color=CI_COLOR, label="Confidence interval (85%)")

    # Plot Bayesian shrunken estimate as line with markers
    ax.plot(X + 1, Y_SHRUNK, marker="o", color=MAIN_COLOR, linewidth=2, label="Probability")

    # Labels and grid
    ax.set_xlabel("Index in Franchise")
    ax.set_ylabel("Probability of Success")
    ax.set_ylim(0, 0.6)
    ax.set_xticks(np.arange(int(X.min()) + 1, int(X.max()) + 2, 1))  # integer ticks
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="upper right")
    ax.set_title(title, fontdict={"fontweight": "bold", "fontsize": 14})

    plt.tight_layout()
    plt.savefig(PROBABILITY_OF_SUCCESS_FIGURE_OUT_PATH_MAP[metric], dpi=DPI)
    plt.show()


def plot_movies_per_sequel_index(data: pd.DataFrame) -> None:
    """
    Plots a bar chart showing the number of movies at each sequel index across franchises.

    This function takes a DataFrame, counts the occurrences of each sequel index, and
    visualizes the distribution as a bar chart.
    :param data: A pandas DataFrame containing the data to be plotted.
    :raises KeyError: If the required column `INDEX_IN_GROUP` is not present in the DataFrame.
    :raises FileNotFoundError: If the directory for saving the plot does not exist.
    """
    data[INDEX_IN_GROUP].value_counts().sort_index().plot(
        kind="bar", figsize=(8, 5), color=MAIN_COLOR, edgecolor="black", zorder=3
    )
    plt.xlabel("Index in Group")
    plt.xticks()
    plt.xticks(rotation=0)
    plt.xlim(left=0.5)
    plt.ylabel("Number of Movies")
    plt.title(
        "Number of Movies in Each Sequel Index Across Franchises",
        fontdict={"fontweight": "bold", "fontsize": 14},
    )
    plt.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(MOVIES_PER_SEQUEL_FIGURE_OUT_PATH, dpi=DPI)
    plt.show()


if __name__ == "__main__":
    # Load all dataframes
    rt_movies_df: pd.DataFrame = pd.read_csv(ROTTEN_TOMATOES_PATH)
    wikidata_movies_df: pd.DataFrame = pd.read_csv(WIKIDATA_PATH)
    imdb_movies_df: pd.DataFrame = pd.read_csv(IMDB_PATH)

    data: pd.DataFrame = preprocess_data(rt_movies_df, wikidata_movies_df, imdb_movies_df)

    plot_movies_per_sequel_index(data)
    plot_probability_of_success(data, metric=ROI, title=TITLE_MAP[ROI])
    plot_probability_of_success(data, metric=AUDIENCE_RATING, title=TITLE_MAP[AUDIENCE_RATING])
    plot_probability_of_success(data, metric=CRITIC_RATING, title=TITLE_MAP[CRITIC_RATING])
