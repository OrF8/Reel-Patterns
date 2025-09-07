import pandas as pd
from consts import (
    BAD_SERIES_LABELS,
    BUDGET,
    BUDGET_FROM_IMDB,
    GROUP_ID,
    GROUP_LABEL,
    SERIES_ID,
    SERIES_LABEL,
    FRANCHISE_ID,
    FRANCHISE_LABEL,
    RELEASE_DATE,
    IMDB_ID,
    IMDB_RELEASE_DATE,
    REVENUE,
    REVENUE_FROM_IMDB,
    WIKIDATA_REVENUE,
    WIKIDATA_BUDGET,
    ROI,
    N_FILMS_IN_GROUP,
    INDEX_IN_GROUP,
    REMOVE_FRANCHISE,
    REMOVE_SERIES,
    ROTTEN_TOMATOES_ID,
    WIKIDATA_FILM_ID,
)


def preprocess_data(
    rt_movies_df: pd.DataFrame,
    wikidata_movies_df: pd.DataFrame,
    imdb_movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocess movie data by integrating and enriching information from multiple sources.
    This function performs the following steps:
    1. Unifies the series and franchise columns in the Wikidata dataset.
    2. Merges Rotten Tomatoes reviews and audience metrics into the Wikidata dataset.
    3. Fills in missing budget and box office values using the IMDB dataset.
    4. Calculates group sizes for movies and filters out groups of size 1.
    5. Fills in missing release dates using the IMDB dataset.
    6. Calculates the index of each movie within its group based on release date.
    Args:
        rt_movies_df (pd.DataFrame): DataFrame containing Rotten Tomatoes movie data.
        wikidata_movies_df (pd.DataFrame): DataFrame containing Wikidata movie data.
        imdb_movies_df (pd.DataFrame): DataFrame containing IMDB movie data.
    Returns:
        pd.DataFrame: A preprocessed DataFrame with integrated and enriched movie data.
    """

    # Unify series and franchise columns
    wikidata_movies_df = unify_series_and_franchise_columns(wikidata_movies_df)

    # Add reviews and audience metrics from Rotten Tomatoes
    wikidata_movies_df = wikidata_movies_df.merge(
        rt_movies_df, on=ROTTEN_TOMATOES_ID, how="inner"
    )

    # Fill in missing budget and box office values from IMDB
    wikidata_movies_df = handle_missing_revenue_and_budget(
        wikidata_movies_df, imdb_movies_df
    )

    # Calculate group sizes per movie and filter out groups of size 1
    wikidata_movies_df = handle_groups_sizes(wikidata_movies_df)

    # Fill in missing release dates from IMDB
    wikidata_movies_df = handle_release_dates(wikidata_movies_df, imdb_movies_df)

    # Calculate index of movie within its group based on release date
    wikidata_movies_df = calculate_index_in_group(wikidata_movies_df)

    return wikidata_movies_df


def calculate_index_in_group(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the index of each row within its respective group in a DataFrame.
    This function sorts the input DataFrame by the `RELEASE_DATE` column and then
    assigns an index to each row within its group, based on the `GROUP_ID` column.
    The index is calculated as the cumulative count of rows within each group.
    Args:
        wikidata_movies_df (pd.DataFrame): A pandas DataFrame containing movie data.
            It must include the columns `RELEASE_DATE` and `GROUP_ID`.
    Returns:
        pd.DataFrame: The input DataFrame with an additional column `INDEX_IN_GROUP`,
        which contains the calculated index for each row within its group.
    Raises:
        KeyError: If the required columns `RELEASE_DATE` or `GROUP_ID` are missing
        from the input DataFrame.
    """

    wikidata_movies_df = wikidata_movies_df.sort_values(by=RELEASE_DATE)
    wikidata_movies_df[INDEX_IN_GROUP] = wikidata_movies_df.groupby(GROUP_ID).cumcount()
    return wikidata_movies_df


def handle_release_dates(
    wikidata_movies_df: pd.DataFrame, imdb_movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Handles missing release dates in a DataFrame by attempting to fill them using data from another DataFrame.

    This function merges the `wikidata_movies_df` DataFrame with the `imdb_movies_df` DataFrame on the IMDB ID column.
    It then fills in missing values in the `RELEASE_DATE` column of `wikidata_movies_df` using the corresponding values
    from the `IMDB_RELEASE_DATE` column of `imdb_movies_df`. Finally, it ensures that the `RELEASE_DATE` column is unified
    and contains the most complete data available.

    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data from Wikidata, including a `RELEASE_DATE` column.
        imdb_movies_df (pd.DataFrame): A DataFrame containing movie data from IMDB, including `IMDB_ID` and `IMDB_RELEASE_DATE` columns.

    Returns:
        pd.DataFrame: The updated `wikidata_movies_df` DataFrame with missing release dates filled where possible.
    """

    # Try to fill in missing release_date from IMDB
    wikidata_movies_df = wikidata_movies_df.merge(
        imdb_movies_df[[IMDB_ID, IMDB_RELEASE_DATE]], on=IMDB_ID, how="left"
    )

    # Unify the release_date columns so we have only one
    wikidata_movies_df[RELEASE_DATE] = wikidata_movies_df[RELEASE_DATE].fillna(
        wikidata_movies_df[IMDB_RELEASE_DATE]
    )
    return wikidata_movies_df


def handle_groups_sizes(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and processes a DataFrame of movies to handle group sizes.
    This function calculates the number of movies in each group and filters out
    groups that contain only one movie. It adds a new column to the DataFrame
    indicating the size of the group each movie belongs to.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data.
            It must include columns for group identification (GROUP_ID).
    Returns:
        pd.DataFrame: A filtered DataFrame containing only movies that are part
        of groups with more than one movie. The DataFrame includes an additional
        column (N_FILMS_IN_GROUP) indicating the size of each group.
    """

    # Calculating how many movies are in group for each line
    wikidata_movies_df[N_FILMS_IN_GROUP] = wikidata_movies_df.groupby(
        [GROUP_ID]
    ).transform("size")
    # Keep only movies that are part of a group with more than one movie
    valid_number_of_films_in_group = wikidata_movies_df[N_FILMS_IN_GROUP] > 1
    wikidata_movies_df = wikidata_movies_df[valid_number_of_films_in_group]
    return wikidata_movies_df


def handle_missing_revenue_and_budget(
    wikidata_movies_df: pd.DataFrame, imdb_movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Handles missing revenue and budget data in a DataFrame by merging data from another DataFrame
    and unifying the budget and revenue columns.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data from Wikidata,
                                           including columns for budget and revenue.
        imdb_movies_df (pd.DataFrame): A DataFrame containing movie data from IMDB,
                                       including columns for budget, revenue, and IMDB IDs.
    Returns:
        pd.DataFrame: A DataFrame with unified budget and revenue columns, an additional ROI column,
                      and rows with missing or invalid budget/revenue values removed.
    Process:
        1. Merges `wikidata_movies_df` with `imdb_movies_df` on the IMDB ID column to fill missing
           budget and revenue values.
        2. Renames the IMDB budget and revenue columns to avoid confusion.
        3. Fills missing budget and revenue values in the Wikidata DataFrame with values from IMDB.
        4. Calculates the ROI (Return on Investment) as the ratio of revenue to budget.
        5. Removes rows with missing, zero, or negative budget or revenue values.
    """

    # Try to fill in missing budget and box office value from IMDB
    wikidata_movies_df = wikidata_movies_df.merge(
        imdb_movies_df[[BUDGET, REVENUE, IMDB_ID]], on=IMDB_ID, how="left"
    )
    # Rename columns to avoid confusion
    wikidata_movies_df = wikidata_movies_df.rename(
        columns={BUDGET: BUDGET_FROM_IMDB, REVENUE: REVENUE_FROM_IMDB}
    )

    # Now dataframe has columns:
    # - BUDGET (from Wikidata)
    # - REVENUE (from Wikidata)
    # - BUDGET_FROM_IMDB (from IMDB)
    # - REVENUE_FROM_IMDB (from IMDB)

    # Unify the budget and box office value columns so we have only one of each
    wikidata_movies_df[BUDGET] = wikidata_movies_df[WIKIDATA_BUDGET].fillna(
        wikidata_movies_df[BUDGET_FROM_IMDB]
    )
    wikidata_movies_df[REVENUE] = wikidata_movies_df[WIKIDATA_REVENUE].fillna(
        wikidata_movies_df[REVENUE_FROM_IMDB]
    )
    # Calculate ROI column
    wikidata_movies_df[ROI] = wikidata_movies_df[REVENUE] / wikidata_movies_df[BUDGET]

    # Removing rows with missing or zero budget or box office value
    rows_to_remove = (
        wikidata_movies_df[BUDGET].isna()
        | wikidata_movies_df[REVENUE].isna()
        | (wikidata_movies_df[BUDGET] <= 0)
        | (wikidata_movies_df[REVENUE] <= 0)
    )
    wikidata_movies_df = wikidata_movies_df[~rows_to_remove]

    return wikidata_movies_df


def unify_series_and_franchise_columns(
    wikidata_movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cleans and processes a DataFrame of movie data by unifying series and franchise columns.
    This function performs the following steps:
    1. Removes invalid movies from the DataFrame.
    2. Removes bad or irrelevant series labels.
    3. Identifies and removes specific series and franchises that should be excluded.
    4. Creates a single grouping column to unify series and franchise information.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data, including series and franchise information.
    Returns:
        pd.DataFrame: A cleaned and processed DataFrame with unified series and franchise columns.
    """
    wikidata_movies_df = remove_invalid_movies(wikidata_movies_df)
    wikidata_movies_df = remove_bad_series_labels(wikidata_movies_df)
    series_to_remove, franchises_to_remove = calculate_franchises_and_series_to_remove(
        wikidata_movies_df
    )
    wikidata_movies_df = remove_series_and_franchises(
        wikidata_movies_df, series_to_remove, franchises_to_remove
    )
    wikidata_movies_df = create_single_grouping_column(wikidata_movies_df)
    return wikidata_movies_df


def remove_invalid_movies(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame of Wikidata movies to remove invalid entries.
    This function retains only the movies that are part of a series or franchise
    and have a valid Wikidata QID.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data from Wikidata.
            It is expected to have the following columns:
            - SERIES_ID: Column indicating if the movie is part of a series.
            - FRANCHISE_ID: Column indicating if the movie is part of a franchise.
            - WIKIDATA_FILM_ID: Column containing the Wikidata QID for the movie.
    Returns:
        pd.DataFrame: A filtered DataFrame containing only valid movies.
    """

    part_of_series_or_franchise = (wikidata_movies_df[SERIES_ID].notna()) | (
        wikidata_movies_df[FRANCHISE_ID].notna()
    )
    have_valid_wikidata_qid = wikidata_movies_df[WIKIDATA_FILM_ID].notna()
    wikidata_movies_df = wikidata_movies_df[
        part_of_series_or_franchise & have_valid_wikidata_qid
    ]
    return wikidata_movies_df


def remove_bad_series_labels(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with invalid series labels from the given DataFrame.
    This function identifies rows in the `wikidata_movies_df` DataFrame where the
    `SERIES_LABEL` column contains values that are considered invalid (as defined
    by the `BAD_SERIES_LABELS` list). For these rows, it sets the `SERIES_LABEL`
    and `SERIES_ID` columns to `pd.NA`.
    Args:
        wikidata_movies_df (pd.DataFrame): A pandas DataFrame containing movie data,
                                           including `SERIES_LABEL` and `SERIES_ID` columns.
    Returns:
        pd.DataFrame: The modified DataFrame with invalid series labels and IDs removed.
    """

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
    """
    Identifies all franchises and series connected to a given franchise or series
    within a dataset of movies. The function performs a breadth-first search to
    traverse the connected components of franchises and series.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data with
            columns for franchise and series labels.
        franchise_label (str, optional): The starting franchise label to search
            for connected components. Defaults to None.
        series_label (str, optional): The starting series label to search for
            connected components. Defaults to None.
    Returns:
        tuple: A tuple containing two sets:
            - franchises_visited (set): A set of all franchise labels connected
              to the starting franchise or series.
            - series_visited (set): A set of all series labels connected to the
              starting franchise or series.
    """
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
    """
    Identifies and determines which franchises and series should be removed
    from a given DataFrame of Wikidata movies based on their connected components
    and containment relationships.

    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data
            with columns for series and franchise labels.

    Returns:
        tuple[set, set]: A tuple containing two sets:
            - series_to_remove (set): A set of series labels to be removed.
            - franchises_to_remove (set): A set of franchise labels to be removed.

    The function processes connected components of series and franchises,
    determines their relationships, and decides which entities to remove
    based on containment rules and the size of the connected components.
    """
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
    """
    Removes specified series and franchises from a DataFrame of movies and filters out
    movies that no longer belong to any series or franchise.
    Args:
        wikidata_movies_df (pd.DataFrame): A DataFrame containing movie data with columns
            for series and franchise labels and IDs.
        series_to_remove (set): A set of series labels to be removed from the DataFrame.
        franchises_to_remove (set): A set of franchise labels to be removed from the DataFrame.
    Returns:
        pd.DataFrame: A filtered DataFrame where the specified series and franchises have been
        removed, and movies without any series or franchise are excluded.
    Raises:
        AssertionError: If the resulting DataFrame contains rows where both series and franchise
        information are missing or both are present.
    """
    for series_label in series_to_remove:
        rows = wikidata_movies_df[SERIES_LABEL] == series_label
        wikidata_movies_df.loc[rows, SERIES_LABEL] = pd.NA
        wikidata_movies_df.loc[rows, SERIES_ID] = pd.NA

    for franchise_label in franchises_to_remove:
        rows = wikidata_movies_df[FRANCHISE_LABEL] == franchise_label
        wikidata_movies_df.loc[rows, FRANCHISE_LABEL] = pd.NA
        wikidata_movies_df.loc[rows, FRANCHISE_ID] = pd.NA

    # Some movies may now be without a series or franchise. Remove them.
    part_of_series_or_franchise = (wikidata_movies_df[SERIES_ID].notna()) | (
        wikidata_movies_df[FRANCHISE_ID].notna()
    )
    wikidata_movies_df = wikidata_movies_df[part_of_series_or_franchise]
    assert wikidata_movies_df[[SERIES_ID, FRANCHISE_ID]].isna().sum(axis=1).eq(1).all()
    assert (
        wikidata_movies_df[[SERIES_LABEL, FRANCHISE_LABEL]]
        .isna()
        .sum(axis=1)
        .eq(1)
        .all()
    )
    return wikidata_movies_df


def create_single_grouping_column(wikidata_movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a single grouping column in the given DataFrame by combining franchise and series information.

    This function adds two new columns to the input DataFrame:
    - GROUP_LABEL: Contains the franchise label if available; otherwise, it falls back to the series label.
    - GROUP_ID: Contains the franchise ID if available; otherwise, it falls back to the series ID.

    Args:
        wikidata_movies_df (pd.DataFrame): A pandas DataFrame containing movie data with columns
                                           FRANCHISE_LABEL, SERIES_LABEL, FRANCHISE_ID, and SERIES_ID.

    Returns:
        pd.DataFrame: The modified DataFrame with the new GROUP_LABEL and GROUP_ID columns added.
    """
    wikidata_movies_df[GROUP_LABEL] = wikidata_movies_df[FRANCHISE_LABEL].fillna(
        wikidata_movies_df[SERIES_LABEL]
    )
    wikidata_movies_df[GROUP_ID] = wikidata_movies_df[FRANCHISE_ID].fillna(
        wikidata_movies_df[SERIES_ID]
    )
    return wikidata_movies_df