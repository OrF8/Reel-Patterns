import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import rankdata

FIG_SIZE: Tuple[int, int] = (13, 13)

# Load dataset
sound_df = pd.read_csv("..\\data\\reel_hits.csv").rename(columns={"tconst": "imdb_id"})
# Remove entries with zero album popularity
sound_df = sound_df[sound_df['album_popularity'] > 0]

movie_df = pd.read_csv("..\\data\\clean_tmdb.csv")
# Remove entries from top and bottom 5% revenue
movie_df = movie_df[movie_df['revenue'].between(movie_df['revenue'].quantile(0.05), movie_df['revenue'].quantile(0.95))]

df = pd.merge(sound_df, movie_df, on=["imdb_id", "title", "revenue"])

# Select relevant columns
cols = ['revenue', 'vote_average', 'album_popularity', 'n_tracks', 'album_length_ms']

# Rename columns for better readability in plots
data = df[cols].rename(columns=lambda x: x.replace('_', '\n').title())
data.rename(columns={"N\nTracks": "Number of\nTracks", "Album\nLength\nMs": "Album\nLength"}, inplace=True)

# Define weights based on vote_count
weights = df['vote_count']


def weighted_corr(x, y, w):
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
    :param data: The DataFrame containing the data.
    :return: The weighted Pearson correlation matrix.
    """
    return data.corr(method=lambda x, y: weighted_corr(x, y, weights))


def create_pearson_heatmap() -> None:
    """
    Create a Pearson correlation heatmap for movie success and soundtrack popularity.
    Calculates a weighted Pearson correlation matrix using vote counts as weights.
    """
    # Compute weighted correlation matrix
    corr = calc_corr(data)

    # Plot heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap="viridis", fmt=".2f", square=True,
                cbar_kws={"shrink": .8}, annot_kws={"size": 20})
    plt.title("Pearson Correlation Heatmap For Movie's\nSuccess And Their Soundtrack Popularity\n",
              fontsize=22, fontweight="bold")
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Pearson Correlation", fontsize=18)

    plt.subplots_adjust(left=0.1)
    plt.tight_layout()

    plt.savefig("..\\figures\\reel_hits_pearson_heatmap.png")
    plt.show()


def create_spearman_heatmap() -> None:
    # Rank the data
    ranked_data = data.apply(rankdata)

    # Compute weighted Spearman correlation matrix
    corr = calc_corr(ranked_data)

    # Plot heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap="viridis", fmt=".2f", square=True,
                     cbar_kws={"shrink": .8}, annot_kws={"size": 20})
    plt.title("Spearman Correlation Heatmap For Movie's\nSuccess And Their Soundtrack Popularity\n",
              fontsize=22, fontweight="bold")
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Spearman Correlation", fontsize=18)

    plt.subplots_adjust(left=0.1)
    plt.tight_layout()

    plt.savefig("..\\figures\\reel_hits_spearman_heatmap.png")
    plt.show()


def create_revenue_hist():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['revenue'], bins=100, kde=True)
    plt.title("Distribution of Movie Revenues", fontsize=16)
    plt.xlabel("Revenue", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_album_pop_hist():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['album_popularity'], bins=100, kde=True)
    plt.title("Distribution of Album Popularity", fontsize=16)
    plt.xlabel("Album Popularity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_vote_count_hist():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vote_count'], bins=100, kde=True)
    plt.title("Distribution of Vote Count", fontsize=16)
    plt.xlabel("Vote Count", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_scatterplot_pop_vote():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='album_popularity', y='vote_average', data=df)
    sns.regplot(x='album_popularity', y='vote_average', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
    plt.title("Album Popularity vs Vote Average", fontsize=16)
    plt.xlabel("Album Popularity", fontsize=14)
    plt.ylabel("Vote Average", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_scatterplot_pop_revenue():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='album_popularity', y='revenue', data=df)
    sns.regplot(x='album_popularity', y='revenue', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
    plt.title("Album Popularity vs Revenue", fontsize=16)
    plt.xlabel("Album Popularity", fontsize=14)
    plt.ylabel("Revenue", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_scatterplot_pop_revenue()


# from scipy.stats import pearsonr, spearmanr
#
# def calc_corr_with_pvalues(data, method):
#     """
#     Calculate correlation matrix with p-values.
#     :param data: The DataFrame containing the data.
#     :param method: The correlation method ('pearson' or 'spearman').
#     :return: A tuple of DataFrames (correlation matrix, p-value matrix).
#     """
#     cols = data.columns
#     corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
#     pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
#
#     for i in cols:
#         for j in cols:
#             if method == 'pearson':
#                 corr, pval = pearsonr(data[i], data[j])
#             elif method == 'spearman':
#                 corr, pval = spearmanr(data[i], data[j])
#             corr_matrix.loc[i, j] = corr
#             pval_matrix.loc[i, j] = pval
#
#     return corr_matrix, pval_matrix
#
#
# def create_heatmap_with_pvalues(method):
#     """
#     Create a heatmap with correlation coefficients and p-values.
#     :param method: The correlation method ('pearson' or 'spearman').
#     """
#     if method == 'spearman':
#         ranked_data = data.apply(rankdata)
#         corr, pvals = calc_corr_with_pvalues(ranked_data, method)
#     else:
#         corr, pvals = calc_corr_with_pvalues(data, method)
#
#     # Combine correlation and p-values for annotations
#     annotations = corr.round(2).astype(str) + "\n(p=" + pvals.round(3).astype(str) + ")"
#
#     # Plot heatmap
#     plt.figure(figsize=FIG_SIZE)
#     ax = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=annotations, fmt="", cmap="viridis", square=True,
#                      cbar_kws={"shrink": .8}, annot_kws={"size": 10})
#     plt.title(f"{method.capitalize()} Correlation Heatmap with P-Values\n", fontsize=22, fontweight="bold")
#     plt.xticks(rotation=0, fontsize=16)
#     plt.yticks(rotation=0, fontsize=16)
#
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=16)
#     cbar.set_label(f"{method.capitalize()} Correlation", fontsize=18)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Example usage
# create_heatmap_with_pvalues('pearson')
# create_heatmap_with_pvalues('spearman')
