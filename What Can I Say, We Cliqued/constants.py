# ===============================================================
# Reel Patterns: What Can I Say, We Cliqued - Constants
# ===============================================================

import os

DATA_DIR_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
COLLABS_PATH: str  = os.path.join(DATA_DIR_PATH, "collabs.csv")
MOVIE_ID_COL: str = "tconst"
ACTOR_NAME_ID_COL: str = "nconst"