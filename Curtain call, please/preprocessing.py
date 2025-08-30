import pandas as pd

# Wikidata columns
FRANCHISE_ID = "franchise_qid"
FRANCHISE_LABEL = "franchise_label"
SERIES_ID = "series_qid"
SERIES_LABEL = "series_label"
WIKIDATA_FILM_ID = "wd_film_qid"
WIKIDATA_BUDGET = "budget_value"
WIKIDATA_REVENUE = "box_office_value"
RELEASE_DATE = "original_release_date"
CRITIC_RATING = "tomatometer_rating"
AUDIENCE_RATING = "audience_rating"

# IMDB columns
BUDGET = "budget"
REVENUE = "revenue"
IMDB_RELEASE_DATE = "release_date"

# Mutual identifier columns
IMDB_ID = "imdb_id"
ROTTEN_TOMATOES_ID = "rotten_tomatoes_link"

# New constructed columns
GROUP_ID = "group_id"
GROUP_LABEL = "group_label"
BUDGET_FROM_IMDB = "imdb_budget"
REVENUE_FROM_IMDB = "imdb_box_office_value"
ROI = "roi"
N_FILMS_IN_GROUP = "n_films_in_group"
INDEX_IN_GROUP = "index_in_group"

BAD_SERIES_LABELS = [
    "Walt Disney Animation Studios film",
    "BBC's 100 Greatest Films of the 21st Century",
    "DreamWorks Animation feature films",
    "list of Sony Pictures Animation productions",
    "Studio Ghibli Feature Films",
    "list of Pixar films",
    "list of Illumination films",
    "MonsterVerse",
]

REMOVE_FRANCHISE = 1
REMOVE_SERIES = 2


def preprocess_data():
    pass

def unify_series_and_franchise_columns(
    wikidata_movies_df: pd.DataFrame,
) -> pd.DataFrame:
    wikidata_movies_df = remove_invalid_movies(wikidata_movies_df)
    wikidata_movies_df = remove_bad_series_labels(wikidata_movies_df)
    series_to_remove, franchises_to_remove = calculate_franchises_and_series_to_remove(wikidata_movies_df)
    wikidata_movies_df = remove_series_and_franchises(
        wikidata_movies_df, series_to_remove, franchises_to_remove
    )
    wikidata_movies_df = create_single_grouping_column(wikidata_movies_df)
    return wikidata_movies_df

def remove_invalid_movies(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only movies that are part of a series or franchise and have a Wikidata QID
    part_of_series_or_franchise = (wikidata_movies_df[SERIES_ID].notna()) | (
        wikidata_movies_df[FRANCHISE_ID].notna()
    )
    have_valid_wikidata_qid = wikidata_movies_df[WIKIDATA_FILM_ID].notna()
    wikidata_movies_df = wikidata_movies_df[
        part_of_series_or_franchise & have_valid_wikidata_qid
    ]
    return wikidata_movies_df

def remove_bad_series_labels(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    # Remove rows with bad series labels
    bad_series_rows = wikidata_movies_df[SERIES_LABEL].isin(BAD_SERIES_LABELS)
    wikidata_movies_df.loc[bad_series_rows, SERIES_LABEL] = pd.NA
    wikidata_movies_df.loc[bad_series_rows, SERIES_ID] = pd.NA
    return wikidata_movies_df

def find_all_franchises_and_series_in_connected_component(
    wikidata_movies_df: pd.DataFrame,
    franchise_label: str = None,
    series_label: str = None,
):
    franchises_visited = set()
    franchises_to_check = [] if pd.isna(franchise_label) else [franchise_label]
    series_visited = set()
    series_to_check = [] if pd.isna(series_label) else [series_label]
    while franchises_to_check or series_to_check:
        rows = pd.Series(False, index=wikidata_movies_df.index)

        if franchises_to_check:
            curr_franchise = franchises_to_check.pop()
            rows |= wikidata_movies_df[FRANCHISE_LABEL] == curr_franchise
            franchises_visited.add(curr_franchise)

        if series_to_check:
            curr_series = series_to_check.pop()
            rows |= wikidata_movies_df[SERIES_LABEL] == curr_series
            series_visited.add(curr_series)

        related_movies = wikidata_movies_df[rows]
        for new_franchise in related_movies[FRANCHISE_LABEL].dropna().unique():
            if new_franchise not in franchises_visited:
                franchises_to_check.append(new_franchise)
        for new_series in related_movies[SERIES_LABEL].dropna().unique():
            if new_series not in series_visited:
                series_to_check.append(new_series)

    return franchises_visited, series_visited

def calculate_franchises_and_series_to_remove(wikidata_movies_df: pd.DataFrame):
    already_found_franchises = set()
    already_found_series = set()

    series_to_remove = set()
    franchises_to_remove = set()

    for series, franchise in (
        wikidata_movies_df[[SERIES_LABEL, FRANCHISE_LABEL]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        connected_franchises, connected_series = (
            find_all_franchises_and_series_in_connected_component(
                wikidata_movies_df, franchise, series
            )
        )
        if connected_franchises.issubset(
            already_found_franchises
        ) and connected_series.issubset(already_found_series):
            # Already processed this connected component
            continue
        sum_of_lengths = len(connected_franchises) + len(connected_series)
        if sum_of_lengths == 1:
            # Disconnected component with only one series or franchise, nothing to resolve
            continue
        if sum_of_lengths == 2:
            containment_relation_result = decide_who_to_remove_based_on_containment(
                wikidata_movies_df, series, franchise
            )
        if sum_of_lengths > 2:
            containment_relation_result = (
                REMOVE_FRANCHISE
                if len(connected_franchises) <= len(connected_series)
                else REMOVE_SERIES
            )

        if containment_relation_result == REMOVE_FRANCHISE:
            franchises_to_remove.update(connected_franchises)
        else:
            series_to_remove.update(connected_series)

        already_found_franchises.update(connected_franchises)
        already_found_series.update(connected_series)

    return series_to_remove, franchises_to_remove

def decide_who_to_remove_based_on_containment(
    df: pd.DataFrame, series_label, franchise_label
):
    """
    Returns one of: 'X_contains_Y', 'Y_contains_X', 'equal', or 'neither'.
    key_col: a unique item identifier column; if None, uses the row index.
    """
    series_items = set(df.index[df[SERIES_LABEL] == series_label])
    franchise_items = set(df.index[df[FRANCHISE_LABEL] == franchise_label])

    if series_items == franchise_items:
        return REMOVE_FRANCHISE
    if franchise_items <= series_items:
        return REMOVE_FRANCHISE
    if series_items <= franchise_items:
        return REMOVE_SERIES
    return (
        REMOVE_FRANCHISE if len(series_items) >= len(franchise_items) else REMOVE_SERIES
    )

def remove_series_and_franchises(
    wikidata_movies_df: pd.DataFrame,
    series_to_remove: set,
    franchises_to_remove: set,
) -> pd.DataFrame:
    for series_label in series_to_remove:
        rows = wikidata_movies_df[SERIES_LABEL] == series_label
        wikidata_movies_df.loc[rows, SERIES_LABEL] = pd.NA
        wikidata_movies_df.loc[rows, SERIES_ID] = pd.NA
        
    for franchise_label in franchises_to_remove:
        rows = wikidata_movies_df[FRANCHISE_LABEL] == franchise_label
        wikidata_movies_df.loc[rows, FRANCHISE_LABEL] = pd.NA
        wikidata_movies_df.loc[rows, FRANCHISE_ID] = pd.NA

    # Some movies may now be without a series or franchise. Remove them.
    part_of_series_or_franchise = (wikidata_movies_df[SERIES_ID].notna()) | (wikidata_movies_df[FRANCHISE_ID].notna())
    wikidata_movies_df = wikidata_movies_df[part_of_series_or_franchise]
    assert wikidata_movies_df[[SERIES_ID, FRANCHISE_ID]].isna().sum(axis=1).eq(1).all()
    assert wikidata_movies_df[[SERIES_LABEL, FRANCHISE_LABEL]].isna().sum(axis=1).eq(1).all()
    return wikidata_movies_df

def create_single_grouping_column(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    wikidata_movies_df[GROUP_LABEL] = wikidata_movies_df[FRANCHISE_LABEL].fillna(wikidata_movies_df[SERIES_LABEL])
    wikidata_movies_df[GROUP_ID] = wikidata_movies_df[FRANCHISE_ID].fillna(wikidata_movies_df[SERIES_ID])
    return wikidata_movies_df