import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

FIG_SIZE: Tuple[int, int] = (13, 13)

# Load dataset
sound_df = pd.read_csv("data\\reel_hits.csv").rename(columns={"tconst": "imdb_id"})
movie_df = pd.read_csv("data\\clean_tmdb.csv")

df = pd.merge(sound_df, movie_df, on=["imdb_id", "title", "revenue"])


def create_heatmap() -> None:
    # Select relevant columns
    cols = ['revenue', 'vote_average', 'album_popularity', 'n_tracks', 'album_length_ms']
    data = df[cols].rename(columns=lambda x: x.replace('_', '\n').title())
    data.rename(columns={"N\nTracks": "Number of\nTracks", "Album\nLength\nMs": "Album\nLength"}, inplace=True)

    # Compute correlation matrix
    corr = data.corr()

    # Plot heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(data=corr, annot=True, cmap="viridis", fmt=".2f", square=True,
                cbar_kws={"shrink": .8}, annot_kws={"size": 20})
    plt.title("Correlation Heatmap For Movie's Success\n And Their Soundtrack Popularity\n",
              fontsize=22, fontweight="bold")
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Correlation", fontsize=18)

    plt.subplots_adjust(left=0.1)
    plt.tight_layout()

    plt.savefig("figures\\reel_hits_heatmap.png")
    plt.show()


if __name__ == "__main__":
    create_heatmap()
