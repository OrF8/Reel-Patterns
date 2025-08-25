import pandas as pd

def create_best_collabs() -> None:
    """
    Create a CSV file with the best collaborations between actors and directors from the IMDB dataset.
    """
    # Load all the actors and directors from the names.basics file
    names_basics = pd.read_csv("../data/name.basics.tsv", sep="\t",
                               usecols=['nconst', 'primaryName', 'primaryProfession'])
    names_basics = names_basics[names_basics['primaryProfession'].str.contains('actor|actress|director', na=False)]
    names_basics = names_basics[['nconst', 'primaryName']]

    print("Successfully loaded names_basics with shape:", names_basics.shape)

    # Load all the titles
    titles_basics = pd.read_csv("../data/title.basics.tsv", sep="\t",
                                usecols=['tconst', 'primaryTitle', 'startYear', 'titleType'], dtype=str)
    # Filter for movies only
    titles_basics = titles_basics[titles_basics['titleType'] == 'movie']
    # Filter for titles that have a start year and are from 1990 onwards
    titles_basics = titles_basics[titles_basics['startYear'].str.isnumeric()]
    titles_basics = titles_basics[titles_basics['startYear'].astype(int) >= 1990]
    titles_basics = titles_basics[titles_basics['primaryTitle'].notna()]
    titles_basics = titles_basics[['tconst', 'primaryTitle']]

    print("Successfully loaded titles_basics with shape:", titles_basics.shape)

    # Load the alternative titles
    alternatives = pd.read_csv("../data/title.akas.tsv", sep="\t",
                               usecols=['titleId', 'region'], dtype=str)
    # Filter only US movies
    alternatives = alternatives[alternatives['region'] == 'US']
    alternatives = alternatives[['titleId']].rename(columns={'titleId': 'tconst'})

    print("Successfully loaded alternatives with shape:", alternatives.shape)


    # Load the title.principals file to get the relationships between titles and names
    titles_principals = pd.read_csv("../data/title.principals.tsv", sep="\t",
                                    usecols=['tconst', 'nconst', 'category'], dtype=str)
    # Filter for actors, actresses, and directors again, for consistency
    titles_principals = titles_principals[titles_principals['category'].isin(['actor', 'actress', 'director'])]
    titles_principals = titles_principals[['tconst', 'nconst']]

    print("Successfully loaded titles_principals with shape:", titles_principals.shape)

    # Load the ratings file to get the ratings for each title
    ratings = pd.read_csv("../data/title.ratings.tsv", sep="\t", dtype=str)
    # Filter out titles with a small amount of ratings
    ratings = ratings[ratings['numVotes'].str.isnumeric()]
    ratings = ratings[ratings['numVotes'].astype(int) > 1000]
    # Filter out titles with a rating below 7.0
    ratings = ratings[ratings['averageRating'].astype(float) >= 7.0]
    ratings = ratings[['tconst']]

    print("Successfully loaded ratings with shape:", ratings.shape)

    # Merge the dataframes to get the names associated with each title and the actor or director for each title
    merged_df = (titles_principals
                 .merge(titles_basics, on="tconst")
                 .merge(ratings, on="tconst")
                 .merge(alternatives, on="tconst")
                 .merge(names_basics, on="nconst", how="inner")
                 .drop_duplicates()
                 .rename(columns={'primaryTitle': 'title', 'primaryName': 'name'}))
    print("Merged with shape:", merged_df.shape)
    print(merged_df.head())
    merged_df.to_csv("data\\collabs.csv", index=False)



