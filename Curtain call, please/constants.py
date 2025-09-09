import os
from typing import Dict, List, Literal

# Data path
DATA_DIR_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ROTTEN_TOMATOES_PATH: str = os.path.join(DATA_DIR_PATH, "rotten_tomatoes_movies.csv")
WIKIDATA_PATH: str = os.path.join(DATA_DIR_PATH, "wikidata_movies.csv")
IMDB_PATH: str = os.path.join(DATA_DIR_PATH, "clean_tmdb.csv")

# Wikidata columns
FRANCHISE_ID: str = "franchise_qid"
FRANCHISE_LABEL: str = "franchise_label"
SERIES_ID: str = "series_qid"
SERIES_LABEL: str = "series_label"
WIKIDATA_FILM_ID: str = "wd_film_qid"
WIKIDATA_BUDGET: str = "budget_value"
WIKIDATA_REVENUE: str = "box_office_value"
RELEASE_DATE: str = "original_release_date"

# IMDB columns
BUDGET: str = "budget"
REVENUE: str = "revenue"
IMDB_RELEASE_DATE: str = "release_date"

# Mutual identifier columns
IMDB_ID: str = "imdb_id"
ROTTEN_TOMATOES_ID: str = "rotten_tomatoes_link"

# New constructed columns
GROUP_ID: str = "group_id"
GROUP_LABEL: str = "group_label"
BUDGET_FROM_IMDB: str = "imdb_budget"
REVENUE_FROM_IMDB: str = "imdb_box_office_value"
ROI: str = "roi"
N_FILMS_IN_GROUP: str = "n_films_in_group"
INDEX_IN_GROUP: str = "index_in_group"

BAD_SERIES_LABELS: List[str] = [
    "Walt Disney Animation Studios film",
    "BBC's 100 Greatest Films of the 21st Century",
    "DreamWorks Animation feature films",
    "list of Sony Pictures Animation productions",
    "Studio Ghibli Feature Films",
    "list of Pixar films",
    "list of Illumination films",
    "MonsterVerse"
]

REMOVE_FRANCHISE: int = 1
REMOVE_SERIES: int = 2

CRITIC_RATING: str = "tomatometer_rating"
AUDIENCE_RATING: str = "audience_rating"

BEST_METRIC_VALUE_SO_FAR: str = "best_metric_value_so_far"
PERCENT_OF_BEST_SO_FAR: str = "percent_of_best_so_far"
AS_SUCCESSFUL_AS_BEST_SO_FAR: str = "as_successful_as_best_so_far"
PROB_OF_SUCCESS_NAIVE: str = "prob_of_success_naive"
N_SUCCESSES: str = "n_successes"
N_MOVIES: str = "n_movies"
BAYESIAN_SHRUNKEN_PROB_OF_SUCCESS: str = "bayesian_shrunken_prob_of_success"
CI_LOWER: str = "ci_lower"
CI_UPPER: str = "ci_upper"
Metric = Literal[AUDIENCE_RATING, CRITIC_RATING, ROI]
BASE_TITLE: str = "Probability of a Sequel to Be as Successful as the Best So Far\nIn Terms of {}"
TITLE_MAP: Dict[str, str] = {
    AUDIENCE_RATING: BASE_TITLE.format("Audience Rating"),
    CRITIC_RATING: BASE_TITLE.format("Critic Rating"),
    ROI: BASE_TITLE.format("Return on Investment (ROI) - Revenue / Budget")
}
FIGURE_OUT_PATH_DIRECTORY: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
MOVIES_PER_SEQUEL_FIGURE_OUT_PATH: str = os.path.join(FIGURE_OUT_PATH_DIRECTORY, "movies_per_sequel_index.png")
PROBABILITY_OF_SUCCESS_FIGURE_OUT_PATH_MAP: Dict[str, str] = {
    AUDIENCE_RATING: os.path.join(FIGURE_OUT_PATH_DIRECTORY, "prob_of_success_audience_rating.png"    ),
    CRITIC_RATING: os.path.join(FIGURE_OUT_PATH_DIRECTORY, "prob_of_success_critic_rating.png"    ),
    ROI: os.path.join(FIGURE_OUT_PATH_DIRECTORY, "prob_of_success_roi.png")
}
DPI: int = 300
MAIN_COLOR: str = "#A40000"
CI_COLOR: str = "#FF5B5B"
