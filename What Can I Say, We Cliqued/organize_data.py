import os
import pandas as pd

DATA_DIR_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
NAME_BASICS_PATH: str = os.path.join(DATA_DIR_PATH, "name.basics.tsv")
TITLE_BASICS_PATH: str = os.path.join(DATA_DIR_PATH, "title.basics.tsv")
AKAS_PATH: str = os.path.join(DATA_DIR_PATH, "title.akas.tsv")
PRINCIPALS_PATH: str = os.path.join(DATA_DIR_PATH, "title.principals.tsv")
RATINGS_PATH: str = os.path.join(DATA_DIR_PATH, "title.ratings.tsv")
COLLABS_PATH: str = os.path.join(DATA_DIR_PATH, "collabs.csv")
TSV_SEP: str = "\t"
ACTOR_NAME_ID_COL: str = "nconst"
ACTOR_NAME_COL: str = "primaryName"
PROFESSION_COL: str = "primaryProfession"
MOVIE_ID_COL: str  = "tconst"
MOVIE_TITLE_COL: str = "primaryTitle"
MOVIE_YEAR_COL: str = "startYear"
MOVIE_TYPE_COL: str = "titleType"
TITLE_ID_COL: str = "titleId"
TITLE_REGION_COL: str = "region"
TITLE_CATEGORY_COL: str= "category"
ACTOR_PROFESSION: str = "actor"
ACTRESS_PROFESSION: str = "actress"
DIRECTOR_PROFESSION: str = "director"
NUM_VOTES_COL: str = "numVotes"

def create_best_collabs() -> None:
    """
    Create a CSV file with the best collaborations between actors and directors from the IMDB dataset.
    """
    # Load all the actors and directors from the names.basics file
    names_basics = pd.read_csv(NAME_BASICS_PATH, sep=TSV_SEP,
                               usecols=[ACTOR_NAME_ID_COL, ACTOR_NAME_COL, PROFESSION_COL])
    names_basics = names_basics[names_basics[PROFESSION_COL].str.contains(
        f"{ACTOR_PROFESSION}|{ACTRESS_PROFESSION}|{DIRECTOR_PROFESSION}", na=False
    )]
    names_basics = names_basics[[ACTOR_NAME_ID_COL, ACTOR_NAME_COL]]

    print("Successfully loaded names_basics with shape:", names_basics.shape)

    # Load all the titles
    titles_basics = pd.read_csv(TITLE_BASICS_PATH, sep=TSV_SEP,
                                usecols=[MOVIE_ID_COL, MOVIE_TITLE_COL, MOVIE_YEAR_COL, MOVIE_TYPE_COL], dtype=str)
    # Filter for movies only
    titles_basics = titles_basics[titles_basics[MOVIE_TYPE_COL] == "movie"]
    # Filter for titles that have a start year and are from 1990 onwards
    titles_basics = titles_basics[titles_basics[MOVIE_YEAR_COL].str.isnumeric()]
    titles_basics = titles_basics[titles_basics[MOVIE_YEAR_COL].astype(int) >= 1990]
    titles_basics = titles_basics[titles_basics[MOVIE_TITLE_COL].notna()]

    # Save primaryTitle for convenience
    titles_basics = titles_basics[[MOVIE_ID_COL, MOVIE_TITLE_COL]]

    print("Successfully loaded titles_basics with shape:", titles_basics.shape)

    # Load the alternative titles
    alternatives = pd.read_csv(AKAS_PATH, sep=TSV_SEP,
                               usecols=[TITLE_ID_COL, TITLE_REGION_COL], dtype=str)
    # Filter only US movies
    alternatives = alternatives[alternatives[TITLE_REGION_COL] == "US"]
    alternatives = alternatives[[TITLE_ID_COL]].rename(columns={TITLE_ID_COL: MOVIE_ID_COL})

    print("Successfully loaded alternatives with shape:", alternatives.shape)


    # Load the title.principals file to get the relationships between titles and names
    titles_principals = pd.read_csv(PRINCIPALS_PATH, sep=TSV_SEP,
                                    usecols=[MOVIE_ID_COL, ACTOR_NAME_ID_COL, TITLE_CATEGORY_COL], dtype=str)
    # Filter for actors, actresses, and directors again, for consistency
    titles_principals = titles_principals[titles_principals[TITLE_CATEGORY_COL].isin(
        [ACTOR_PROFESSION, ACTRESS_PROFESSION, DIRECTOR_PROFESSION]
    )]
    titles_principals = titles_principals[[MOVIE_ID_COL, ACTOR_NAME_ID_COL]]

    print("Successfully loaded titles_principals with shape:", titles_principals.shape)

    # Load the ratings file to get the ratings for each title
    ratings = pd.read_csv(RATINGS_PATH, sep=TSV_SEP, dtype=str)
    # Filter out titles with a small amount of ratings
    ratings = ratings[ratings[NUM_VOTES_COL].str.isnumeric()]
    ratings = ratings[ratings[NUM_VOTES_COL].astype(int) > 1000]
    # Filter out titles with a rating below 7.0
    ratings = ratings[ratings['averageRating'].astype(float) >= 7.0]
    ratings = ratings[[MOVIE_ID_COL]]

    print("Successfully loaded ratings with shape:", ratings.shape)

    # Merge the dataframes to get the names associated with each title and the actor or director for each title
    merged_df = (titles_principals
                 .merge(titles_basics, on=MOVIE_ID_COL)
                 .merge(ratings, on=MOVIE_ID_COL)
                 .merge(alternatives, on=MOVIE_ID_COL)
                 .merge(names_basics, on="nconst", how="inner")
                 .drop_duplicates()
                 .rename(columns={MOVIE_TITLE_COL: 'title', ACTOR_NAME_COL: 'name'}))

    print("Merged with shape:", merged_df.shape)
    print(merged_df.head())
    merged_df.to_csv(COLLABS_PATH, index=False)



