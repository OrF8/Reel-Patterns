# ========================================================================
# Reel Patterns: Reel Hits Meet Real Hits - Data Visualization
#
# Requirements:
#   pip install -r requirements.txt
#
# Usage:
#   python reel_hits_viz.py
# ========================================================================

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.stats import rankdata

FIGURE_OUT_PATH_DIRECTORY: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
FIG_SIZE: Tuple[int, int] = (13, 13)
DPI: int = 300
PEARSON_TITLE: str = "Pearson Correlation Heatmap For Movie's\nSuccess and Their Soundtrack Popularity\n"
PEARSON_PATH: str = os.path.join(FIGURE_OUT_PATH_DIRECTORY, "reel_hits_pearson_heatmap.png")
SPEARMAN_TITLE: str = "Spearman Correlation Heatmap For Movie's\nSuccess and Their Soundtrack Popularity\n"
SPEARMAN_PATH: str = os.path.join(FIGURE_OUT_PATH_DIRECTORY, "reel_hits_spearman_heatmap.png")
CORRELATION_LABEL_FMT: str = f"{{corr}} Correlation"
SCATTER_TITLE: str = "Correlation Between Soundtrack Popularity and Average Rating"
SCATTER_PATH: str = os.path.join(FIGURE_OUT_PATH_DIRECTORY, "corr_pop_rating.png")
DATA_DIR_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SOUND_PATH: str = os.path.join(DATA_DIR_PATH, "reel_hits.csv")
MOVIE_PATH: str = os.path.join(DATA_DIR_PATH, "clean_tmdb.csv")

# Load dataset
sound_df: pd.DataFrame = pd.read_csv(SOUND_PATH).rename(columns={"tconst": "imdb_id"})
# Remove entries with zero album popularity
sound_df = sound_df[sound_df['album_popularity'] > 0]

movie_df: pd.DataFrame = pd.read_csv(MOVIE_PATH)
# Remove entries from the bottom 5% revenue
movie_df = movie_df[movie_df['revenue'].quantile(0.05) < movie_df['revenue']]

df: pd.DataFrame = pd.merge(sound_df, movie_df, on=["imdb_id", "title", "revenue"])

# Select relevant columns
cols: List[str] = ['revenue', 'vote_average', 'album_popularity', 'n_tracks', 'album_length_ms']

# Rename columns for better readability in plots
data: pd.DataFrame = df[cols].rename(columns=lambda x: x.replace('_', '\n').title())
data.rename(columns={"N\nTracks": "Number of\nTracks", "Album\nLength\nMs": "Album\nLength"}, inplace=True)

# Define weights based on vote_count
weights: pd.Series = df['vote_count']


def weighted_corr(x, y, w) -> float:
    """
    Calculate weighted correlation.
    :param x: The first array of values.
    :param y: The second array of values.
    :param w: The weights for each value.
    :return: The weighted correlation coefficient.
    """
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    cov_xy = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)
    std_x = np.sqrt(np.sum(w * (x - w_mean_x) ** 2) / np.sum(w))
    std_y = np.sqrt(np.sum(w * (y - w_mean_y) ** 2) / np.sum(w))
    return cov_xy / (std_x * std_y)


def calc_corr(data) -> pd.DataFrame:
    """
    Calculate the weighted Pearson correlation matrix for the given DataFrame.
    :param data: The DataFrame that contains the data.
    :return: The weighted Pearson correlation matrix.
    """
    return data.corr(method=lambda x, y: weighted_corr(x, y, weights))


def create_heatmap(df: pd.DataFrame, is_pearson: bool = True) -> None:
    """
    Create a heatmap for the given DataFrame.
    :param df: The DataFrame that contains the data.
    :param is_pearson: Whether to create a Pearson or Spearman heatmap.
    """
    label: str = "Pearson" if is_pearson else "Spearman"
    title: str = PEARSON_TITLE if is_pearson else SPEARMAN_TITLE
    path: str = PEARSON_PATH if is_pearson else SPEARMAN_PATH

    # Compute weighted correlation matrix
    corr = calc_corr(df)

    # Plot heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap="viridis", fmt=".2f", square=True,
                     cbar_kws={"shrink": .8}, annot_kws={"size": 20})
    plt.title(title,
              fontsize=22, fontweight="bold")
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(CORRELATION_LABEL_FMT.format(corr=label), fontsize=18)

    plt.subplots_adjust(left=0.1)
    plt.tight_layout()

    plt.savefig(path, dpi=DPI)
    plt.show()


def create_pearson_heatmap() -> None:
    """
    Create a Pearson correlation heatmap for movie success and soundtrack popularity.
    Calculates a weighted Pearson correlation matrix using vote counts as weights.
    """
    create_heatmap(data)


def create_spearman_heatmap() -> None:
    """
    Create a Spearman correlation heatmap for movie success and soundtrack popularity.
    Ranks the data and then calculates a weighted Pearson correlation matrix using vote counts as weights.
    """
    # Rank the data
    ranked_data = data.apply(rankdata)
    # Then create the heatmap
    create_heatmap(ranked_data, is_pearson=False)


def create_revenue_hist() -> None:
    """
    Create a histogram for movie revenues.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['revenue'], bins=100, kde=True)
    plt.title("Distribution of Movie Revenues", fontsize=16)
    plt.xlabel("Revenue", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_album_pop_hist() -> None:
    """
    Create a histogram for album popularity.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['album_popularity'], bins=100, kde=True)
    plt.title("Distribution of Album Popularity", fontsize=16)
    plt.xlabel("Album Popularity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_vote_count_hist() -> None:
    """
    Create a histogram for vote counts.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vote_count'], bins=100, kde=True)
    plt.title("Distribution of Vote Count", fontsize=16)
    plt.xlabel("Vote Count", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_scatterplot_pop_vote() -> None:
    """
    Create a scatter plot showing the correlation between album popularity and average rating, with a regression line.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='album_popularity', y='vote_average', data=df, color='#A40000')
    sns.regplot(x='album_popularity', y='vote_average', data=df, scatter=False, color='green',
                line_kws={"linewidth": 2}, ci=None)
    plt.title(SCATTER_TITLE, fontsize=18, fontweight="bold")
    plt.text(0.01, 0.98, "Corr=0.44", transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top', color='#515151')
    plt.xlabel("Album Popularity", fontsize=16)
    plt.ylabel("Average Rating", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=DPI)
    plt.show()


# TODO: Uncomment and use if needed
# def create_scatterplot_pop_revenue():
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='album_popularity', y='revenue', data=df)
#     sns.regplot(x='album_popularity', y='revenue', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
#     plt.title("Album Popularity vs Revenue", fontsize=16)
#     plt.xlabel("Album Popularity", fontsize=14)
#     plt.ylabel("Revenue", fontsize=14)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    create_pearson_heatmap()
    create_spearman_heatmap()
    create_scatterplot_pop_vote()
