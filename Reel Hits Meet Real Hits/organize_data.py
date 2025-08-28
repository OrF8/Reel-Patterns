import datetime
import spotipy
import os
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from rapidfuzz import fuzz
from dotenv import load_dotenv
from typing import Union, List, Tuple, Any, Dict, Set, Optional

from organize_data import df_reel_hits

# Load environment variables from .env file (API keys)
load_dotenv()

# Spotify API setup
cred_manager: SpotifyClientCredentials = SpotifyClientCredentials(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                                                  client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"))
sp: spotipy.Spotify = spotipy.Spotify(client_credentials_manager=cred_manager,
                                      requests_timeout=15, retries=3, status_forcelist=(429, 500, 502, 503, 504))

# Constants
SOUNDTRACK_HINTS: Tuple[str, str, str, str, str] = (
    "soundtrack", "original motion picture", "music from the motion picture", "original score", "ost"
)
CLEAN_TMDB: pd.DataFrame = pd.read_csv("..\\data\\clean_tmdb.csv")
BIG_REEL_HITS_PATH: str = "..\\data\\reel_hits_big.csv"
SMALL_REEL_HITS_PATH: str = "..\\data\\reel_hits_small.csv"
REEL_HITS_PATH: str = "..\\data\\reel_hits.csv"
IMDB_ID_COL: str = "tconst"
MOVIE_TITLE_COL: str = "title"
MOVIE_YEAR_COL: str = "year"
MOVIE_REVENUE_COL: str = "revenue"
RELEASE_YEAR_COL: str = "release_year"
SPOTIFY_ALBUM_ID_COL: str = "spotify_album_id"
SPOTIFY_ALBUM_NAME_COL: str = "spotify_album_name"
SPOTIFY_ALBUM_YEAR_COL: str = "spotify_album_year"
MATCH_SCORE_COL: str = "match_score"
ALBUM_POPULARITY_COL: str = "album_popularity"
AVG_TRACK_POPULARITY_COL: str = "avg_track_popularity"
SUM_TRACK_POPULARITY_COL: str = "sum_track_popularity"
NUM_TRACKS_COL: str = "n_tracks"
ALBUM_LEN_MS_COL: str = "album_length_ms"
ALBUM_LEN_MIN_COL: str = "album_length_min"
ALBUM_ARTISTS_COL: str = "album_artists"
NAME_METRIC: str = "name"
ID_METRIC: str = "id"
POPULARITY_METRIC: str = "popularity"
ITEMS_METRIC: str = "items"
DURATION_MS_METRIC: str = "duration_ms"
BATCH_SIZE: int = 50  # Max number of tracks to fetch in one API call


def candidate_queries(title: str, year: Optional[int]) -> list[str]:
    """
    Generate candidate search queries for a movie title and optional year.
    These queries are designed to find soundtrack albums related to the movie.
    :param title: The title of the movie.
    :param year: The release year of the movie (optional).
    :return:
    """
    # Base queries
    base = [f'album:"{title}"', f'{title} soundtrack', f'album:"{title} Original Motion Picture Soundtrack"',
            f'album:"{title} Original Score"', f'album:"{title} Music From the Motion Picture"']
    if year:
        # If year is provided, add year-specific queries
        base += [f'album:"{title}" year:{year}', f'{title} year:{year} soundtrack']
    return base


def album_release_year(album) -> Optional[int]:
    """
    Extract the release year from an album's 'release_date' field.
    :param album: The album dictionary from Spotify API.
    :return: The release year as an integer, or None if not available.
    """
    # 'release_date' can be "YYYY" or "YYYY-MM-DD"
    rd = album.get(RELEASE_YEAR_COL)
    if not rd:
        return None
    try:
        return datetime.date.fromisoformat(rd).year
    except ValueError:
        # Fallback: try to parse the first 4 characters as year
        try:
            return int(rd[:4])
        except ValueError:
            # Unable to parse year
            return None


def heuristic_score(album, movie_title, movie_year) -> float:
    """
    Compute a heuristic score for how well an album matches a movie title and year.
    The score is based on:
    1. Title similarity (using fuzzy matching)
    2. Presence of soundtrack hints in the album name
    3. Proximity of the album release year to the movie release year
    4. Bonuses for exact or near year match
    :param album: The album dictionary from Spotify API.
    :param movie_title: The title of the movie.
    :param movie_year: The release year of the movie.
    :return: A float score indicating the quality of the match (0-127).
    """
    name = album.get(NAME_METRIC, '')
    # Title similarity (robust to parentheses)
    title_sim: float = fuzz.token_set_ratio(movie_title, name)
    # Hints
    hint_bonus: int = 15 if any(h in name.lower() for h in SOUNDTRACK_HINTS) else 0
    # Year proximity
    album_year: Optional[int] = album_release_year(album)
    year_bonus: int = 0
    if movie_year and album_year:
        diff: int = abs(album_year - movie_year)
        year_bonus = 12 if diff <= 1 else 6 if diff == 2 else 0
    return title_sim + hint_bonus + year_bonus


def find_best_soundtrack_album(title, year) -> Tuple[Union[Dict[str, Any], None], float]:
    """
    Find the best matching soundtrack album for a given movie title and year.
    :param title: The title of the movie.
    :param year: The release year of the movie.
    :return: A tuple of the best matching album (or None) and its heuristic score.
    """
    tried: Set[str] = set()
    best, best_score = None, -1

    for q_title in [title]:
        for q in candidate_queries(q_title, year):
            if q in tried: continue
            tried.add(q)
            res = sp.search(q=q, type='album', limit=10)
            for alb in res.get('albums', {}).get(ITEMS_METRIC, []):
                score: float = heuristic_score(alb, title, year)
                if score > best_score:
                    best, best_score = alb, score
    return best, best_score


def get_album_popularity_metrics(album_id: str) -> Dict[str, Any]:
    """
    Get popularity metrics for an album and its tracks.
    :param album_id: The Spotify ID of the album.
    :return: A dictionary with album and track popularity metrics:
             1. Album popularity
             2. Average track popularity
             3. Sum of track popularity
             4. Number of tracks
             5. Album length in milliseconds and minutes
             6. Set of unique artists in the album
    """
    album = sp.album(album_id)
    pop_album: int = album.get(POPULARITY_METRIC)
    # Gather all tracks’ popularity
    results = sp.album_tracks(album_id, limit=BATCH_SIZE)
    items = results.get(ITEMS_METRIC, [])
    album_artists: Set[str] = set(artist[NAME_METRIC] for track in items for artist in track.get("artists", []))
    while results.get("next"):
        results = sp.next(results)
        items += results.get(ITEMS_METRIC, [])

    # Album length in milliseconds and minutes
    album_length_ms: int = sum(t.get(DURATION_MS_METRIC) for t in items if t.get(DURATION_MS_METRIC))
    album_length_min: float = album_length_ms / 60000

    # Fetch track details in batches to get popularity
    ids = [t[ID_METRIC] for t in items if t.get(ID_METRIC)]
    pop_list = []
    for i in range(0, len(ids), BATCH_SIZE):
        batch = sp.tracks(ids[i:i+BATCH_SIZE]).get("tracks", [])
        pop_list.extend([t.get(POPULARITY_METRIC) for t in batch if t and t.get(POPULARITY_METRIC) is not None])
    if pop_list:
        avg_track_pop = sum(pop_list) / len(pop_list)
        sum_track_pop = sum(pop_list)
    else:
        avg_track_pop = None
        sum_track_pop = None

    return {
        ALBUM_POPULARITY_COL: pop_album,
        AVG_TRACK_POPULARITY_COL: avg_track_pop,
        SUM_TRACK_POPULARITY_COL: sum_track_pop,
        NUM_TRACKS_COL: len(ids),
        ALBUM_LEN_MS_COL: album_length_ms,
        ALBUM_LEN_MIN_COL: album_length_min,
        ALBUM_ARTISTS_COL: album_artists
    }


def process_movie(tconst: str, title: str, year: int, revenue: float) -> Dict[str, Any]:
    """
    Process a single movie to find its best matching soundtrack album and gather metrics.
    :param tconst: The IMDb ID of the movie.
    :param title: The title of the movie.
    :param year: The release year of the movie.
    :param revenue: The revenue of the movie.
    :return: A dictionary with movie and soundtrack album data.
    """
    album, score = find_best_soundtrack_album(title, year)
    if not album:
        return {IMDB_ID_COL: tconst, MOVIE_TITLE_COL: title, MOVIE_YEAR_COL: year,
                MOVIE_REVENUE_COL: revenue, SPOTIFY_ALBUM_ID_COL: None, MATCH_SCORE_COL: None}

    metrics = get_album_popularity_metrics(album[ID_METRIC])
    return {
        IMDB_ID_COL: tconst,
        MOVIE_TITLE_COL: title,
        MOVIE_YEAR_COL: year,
        MOVIE_REVENUE_COL: revenue,
        SPOTIFY_ALBUM_ID_COL: album[ID_METRIC],
        SPOTIFY_ALBUM_NAME_COL: album[NAME_METRIC],
        SPOTIFY_ALBUM_YEAR_COL: album_release_year(album),
        MATCH_SCORE_COL: score,
        **metrics
    }


def filter_dataset(df: pd.DataFrame, vote_avg_predicate) -> pd.DataFrame:
    """
    Filter the dataset based on several criteria.
    1. Revenue > 0
    2. Release date is not null and year >= 1990
    3. The original language is English
    4. Vote average meets the given predicate and vote count >= 2000
    5. Select only relevant columns for further processing
    :param df: The DataFrame that contains movie data.
    :param vote_avg_predicate: A function that takes a vote average and returns a boolean.
    :return: The filtered DataFrame.
    """
    # Filter out movies with 0 revenue, null release date, null imdb_id, or null title
    df = df[(df[MOVIE_REVENUE_COL] > 0) &
            (df[RELEASE_YEAR_COL].notna()) &
            (df["imdb_id"].notna()) &
            (df[MOVIE_TITLE_COL].notna())]
    # Filter for movies released in or after 1990
    df = df[df[RELEASE_YEAR_COL].apply(lambda x: datetime.date.fromisoformat(x).year if pd.notna(x) else None) >= 1990]
    # Filter for English language movies
    df = df[df["original_language"] == "en"]
    # Filter based on vote average predicate and vote count
    df = df[vote_avg_predicate(df["vote_average"]) & (df["vote_count"] >= 2000)]
    # Select only relevant columns
    df = df[["imdb_id", MOVIE_TITLE_COL, RELEASE_YEAR_COL, MOVIE_REVENUE_COL]]
    print("Filtered DataFrame shape:", df.shape)
    return df


def gather_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gather results by processing each movie in the DataFrame.
    :param df: The DataFrame that contains movie data.
    :return: A DataFrame with processed movie and soundtrack data.
    """
    results: List[Dict[str, Any]] = []
    tuples: List[Tuple[Any, ...]] = list(df.itertuples(index=False))
    from tqdm import tqdm
    for row in tqdm(tuples[:500]):
        tconst: str = row.imdb_id
        title: str = row.title
        year: int = datetime.date.fromisoformat(row.release_date).year
        revenue: float = float(row.revenue)

        result: Dict[str, Any] = process_movie(tconst, title, year, revenue)
        results.append(result)

    return pd.DataFrame(results)


def create_big_reel_hits() -> pd.DataFrame:
    """
    Create the big reel hits dataset by filtering movies with higher ratings.
    :return: The DataFrame that contains big reel hits.
    """
    df: pd.DataFrame = CLEAN_TMDB.copy()
    print("Loaded clean_tmdb.csv with shape:", df.shape)

    # Filter for movies with vote_average >= 7.0
    df = filter_dataset(df, lambda x: x >= 7.0)

    results_df: pd.DataFrame = gather_results(df)

    print(f"Big Reel Hits shape:\n{results_df.head()}")
    results_df.to_csv(BIG_REEL_HITS_PATH, index=False)
    return results_df


def create_small_reel_hits() -> pd.DataFrame:
    """
    Create the small reel hits dataset by filtering movies with lower ratings.
    :return: The DataFrame that contains small reel hits.
    """
    df: pd.DataFrame = CLEAN_TMDB.copy()
    print("Loaded clean_tmdb.csv with shape:", df.shape)

    # Filter for movies with vote_average < 7.0
    df = filter_dataset(df, lambda x: x < 7.0)

    results_df: pd.DataFrame = gather_results(df)

    print(f"Small Reel Hits shape:\n{results_df.head()}")

    results_df.to_csv(SMALL_REEL_HITS_PATH, index=False)
    return results_df


def merge_hits(df_small: pd.DataFrame, df_big: pd.DataFrame) -> pd.DataFrame:
    """
    Merge small and big reel hits DataFrames.
    :param df_small: The DataFrame that contains small reel hits.
    :param df_big: The DataFrame that contains big reel hits.
    :return: The combined DataFrame.
    """
    merged: pd.DataFrame = pd.concat([df_small, df_big], ignore_index=True)
    print("Combined DataFrame shape:", merged.shape)

    return merged


if __name__ == "__main__":
    df_big_hits = create_big_reel_hits()
    df_small_hits = create_small_reel_hits()
    df_reel_hits = merge_hits(df_small_hits, df_big_hits)
    df_reel_hits.to_csv(REEL_HITS_PATH, index=False)
