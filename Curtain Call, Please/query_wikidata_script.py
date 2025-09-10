# ===============================================================
# Reel Patterns: Curtain Call, Please - Collecting Franchise Data
#
# Requirements:
#   pip install -r requirements.txt
#
# Usage:
#   python query_wikidata_script.py
# ===============================================================

import time
import requests
import os
import pandas as pd
from typing import Dict, List, Tuple, Iterable, Optional, Any
from constants import DATA_DIR_PATH, ROTTEN_TOMATOES_ID

WD_ENDPOINT: str = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT: str = "FranchiseMapper/1.0 (adirtuval@gmail.com)"
RT_MOVIES_DF: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR_PATH, "rotten_tomatoes_movies.csv"))
OUT_PATH: str = os.path.join(DATA_DIR_PATH, "wikidata_movies.csv")
ROTTEN_ID_COL: str = "rt_id"
WD_FILM_QID_COL: str = "wd_film_qid"  # 'WD' stands for Wikidata
WD_FILM_LABEL_COL: str = "wd_film_label"
SERIES_QID_COL: str = "series_qid"
SERIES_LABEL_COL: str = "series_label"
FRANCHISE_QID_COL: str = "franchise_qid"
FRANCHISE_LABEL_COL: str = "franchise_label"
BOM_FILM_ID_COL: str = "bom_film_id"  # 'BOM' stands for Box Office Mojo
IMDB_ID_COL: str ="imdb_id"
TMDB_ID_COL: str = "tmdb_id"
BUDGET_VALUE_COL: str  = "budget_value"
BUDGET_CUR_ID_COL: str = "budget_currency_qid"
BUDGET_CUR_COL: str = "budget_currency"
BUDGET_ASOF_COL: str = "budget_asof"
BUDGET_SOURCE_COL: str = "budget_source"
BO_VAL_COL: str = "box_office_value"  # 'BO' stands for Box Office, not Body Odor :)
BO_CUR_ID_COL: str = "box_office_currency_qid"
BO_CUR_COL: str = "box_office_currency"
BO_ASOF_COL: str = "box_office_asof"
BO_REGION_ID_COL: str = "box_office_region_qid"
BO_REGION_COL: str = "box_office_region"
FORWARD_SLASH: str = "/"
UNDERSCORE: str = "_"
HYPHEN: str = "-"

def escape_literal(string: str) -> str:
    """
    Escape a string for SPARQL string literal.
    :param string: input string.
    :return: escaped string.
    """
    return string.replace("\\", "\\\\").replace('"', '\\"')


def build_values_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    Build a VALUES block mapping variant -> original:
        VALUES (?rtid ?orig) { ("variant1" "original1") ("variant2" "original1") ... }
    :param pairs: list of (variant, original) tuples.
    :return: SPARQL VALUES block content.
    """
    rows = []
    for rtid, orig in pairs:
        rows.append(f'("{escape_literal(rtid)}" "{escape_literal(orig)}")')
    return " ".join(rows)


def sparql_for_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    SPARQL: match films by Rotten Tomatoes ID (P1258), return film, series (P179), franchise (P8345),
    money fields (budget P2130/P2769, box office P2142), and external IDs (BOM/IMDb/TMDb).
    For quantities, return amount & unit IRI, and qualifiers when present (P585 as-of, P1001 region).
    ?orig carries the original RT id to map results back to your rows.
    :param pairs: list of (variant, original) RT id tuples.
    :return: SPARQL query string.
    """
    values_block = build_values_pairs(pairs)
    return f"""
SELECT ?orig ?film ?filmLabel ?series ?seriesLabel ?franchise ?franchiseLabel

       ?budget_amount_pref ?budget_unit_pref ?budget_asof_pref
       ?budget_amount_any  ?budget_unit_any  ?budget_asof_any
       ?budget2769_amount  ?budget2769_unit ?budget2769_asof

       ?box_amount_pref ?box_unit_pref ?box_asof_pref ?box_region_pref ?box_region_prefLabel
       ?box_amount_any  ?box_unit_any  ?box_asof_any  ?box_region_any  ?box_region_anyLabel

       ?bom_id ?imdb_id ?tmdb_id

WHERE {{
  VALUES (?rtid ?orig) {{ {values_block} }}
  ?film wdt:P1258 ?rtid .

  OPTIONAL {{ ?film wdt:P179  ?series. }}
  OPTIONAL {{ ?film wdt:P8345 ?franchise. }}

  # -------- External IDs --------
  OPTIONAL {{ ?film wdt:P1237 ?bom_id. }}   # Box Office Mojo film ID
  OPTIONAL {{ ?film wdt:P345  ?imdb_id. }}  # IMDb ID
  OPTIONAL {{ ?film wdt:P4947 ?tmdb_id. }}  # TMDb movie ID

  # -------- Budget (P2130 preferred rank) --------
  OPTIONAL {{
    ?film p:P2130 ?budgetStmtPref .
    ?budgetStmtPref wikibase:rank wikibase:PreferredRank .
    ?budgetStmtPref psv:P2130 ?budgetNodePref .
    ?budgetNodePref wikibase:quantityAmount ?budget_amount_pref .
    ?budgetNodePref wikibase:quantityUnit   ?budget_unit_pref .
    OPTIONAL {{ ?budgetStmtPref pq:P585 ?budget_asof_pref . }}
  }}

  # -------- Budget (P2130 any rank, only if no preferred matched above) --------
  OPTIONAL {{
    FILTER(NOT EXISTS {{ ?film p:P2130 ?_bs1 . ?_bs1 wikibase:rank wikibase:PreferredRank . }})
    ?film p:P2130 ?budgetStmtAny .
    ?budgetStmtAny psv:P2130 ?budgetNodeAny .
    ?budgetNodeAny wikibase:quantityAmount ?budget_amount_any .
    ?budgetNodeAny wikibase:quantityUnit   ?budget_unit_any .
    OPTIONAL {{ ?budgetStmtAny pq:P585 ?budget_asof_any . }}
  }}

  # -------- Budget fallback property (P2769), if P2130 missing --------
  OPTIONAL {{
    FILTER(NOT EXISTS {{ ?film wdt:P2130 ?_hasP2130 . }})
    ?film p:P2769 ?budgetStmt2769 .
    ?budgetStmt2769 psv:P2769 ?budgetNode2769 .
    ?budgetNode2769 wikibase:quantityAmount ?budget2769_amount .
    ?budgetNode2769 wikibase:quantityUnit   ?budget2769_unit .
    OPTIONAL {{ ?budgetStmt2769 pq:P585 ?budget2769_asof . }}
  }}

  # -------- Box office (P2142 preferred rank) --------
  OPTIONAL {{
    ?film p:P2142 ?boxStmtPref .
    ?boxStmtPref wikibase:rank wikibase:PreferredRank .
    ?boxStmtPref psv:P2142 ?boxNodePref .
    ?boxNodePref wikibase:quantityAmount ?box_amount_pref .
    ?boxNodePref wikibase:quantityUnit   ?box_unit_pref .
    OPTIONAL {{ ?boxStmtPref pq:P585  ?box_asof_pref . }}
    OPTIONAL {{ ?boxStmtPref pq:P1001 ?box_region_pref . }}
  }}

  # -------- Box office (P2142 any rank, only if no preferred matched above) --------
  OPTIONAL {{
    FILTER(NOT EXISTS {{ ?film p:P2142 ?_bo1 . ?_bo1 wikibase:rank wikibase:PreferredRank . }})
    ?film p:P2142 ?boxStmtAny .
    ?boxStmtAny psv:P2142 ?boxNodeAny .
    ?boxNodeAny wikibase:quantityAmount ?box_amount_any .
    ?boxNodeAny wikibase:quantityUnit   ?box_unit_any .
    OPTIONAL {{ ?boxStmtAny pq:P585  ?box_asof_any . }}
    OPTIONAL {{ ?boxStmtAny pq:P1001 ?box_region_any . }}
  }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
""".strip()


def post_sparql(query: str, *, user_agent: str = DEFAULT_USER_AGENT,
                max_retries: int = 5, base_sleep: float = 1.5) -> Dict[str, Any]:
    """
    POST a SPARQL query with basic retry/backoff on 429/5xx.
    :param query: SPARQL query string.
    :param user_agent: User-Agent header value.
    :param max_retries: max attempts on 429/5xx.
    :param base_sleep: base sleep time in seconds for backoff.
    :return: parsed JSON response.
    :raises: RuntimeError on repeated 429/5xx or other HTTP errors.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent
    }
    for attempt in range(max_retries):
        resp = requests.post(WD_ENDPOINT, data={"query": query}, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 502, 503, 504):
            sleep = base_sleep * (2 ** attempt) + 0.25 * attempt
            time.sleep(sleep)
            continue
        # Other client/server errors: raise with context
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"WDQS error {resp.status_code}: {resp.text[:400]}") from e
    raise RuntimeError(f"WDQS rate-limited or unavailable after {max_retries} attempts.")


def parse_bindings_to_df(js: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert WDQS JSON results to a tidy DataFrame keyed by original RT id (?orig).
    Handles coalescing of budget/box office qualifiers from multiple ranks/properties.
    :param js: parsed JSON from WDQS.
    :return: DataFrame with one row per original RT id.
    """
    rows = []
    for b in js.get("results", {}).get("bindings", []):
        def get(name: str) -> Optional[str]:
            v = b.get(name, {}).get("value")
            return v if v is not None and v != "" else None

        def last_qid(iri: Optional[str]) -> Optional[str]:
            return iri.split(FORWARD_SLASH)[-1] if iri else None

        # Coalesce budget amount/unit/asof with preference: P2130 preferred -> P2130 any -> P2769
        budget_amount = (
            get("budget_amount_pref") or
            get("budget_amount_any") or
            get("budget2769_amount")
        )
        budget_unit_iri = (
            get("budget_unit_pref") or
            get("budget_unit_any") or
            get("budget2769_unit")
        )
        budget_asof = (
            get("budget_asof_pref") or
            get("budget_asof_any") or
            get("budget2769_asof")
        )
        budget_source = None
        if get("budget_amount_pref") or get("budget_amount_any"):
            budget_source = "P2130"
        elif get("budget2769_amount"):
            budget_source = "P2769"

        # Coalesce box office amount/unit/asof/region with preference: preferred -> any
        box_amount = get("box_amount_pref") or get("box_amount_any")
        box_unit_iri = get("box_unit_pref") or get("box_unit_any")
        box_asof = get("box_asof_pref") or get("box_asof_any")
        box_region_iri = get("box_region_pref") or get("box_region_any")
        box_region_label = get("box_region_prefLabel") or get("box_region_anyLabel")

        rows.append({
            ROTTEN_ID_COL: get("orig"),
            WD_FILM_QID_COL: last_qid(get("film")),
            WD_FILM_LABEL_COL: get("filmLabel"),
            SERIES_QID_COL: last_qid(get("series")),
            SERIES_LABEL_COL: get("seriesLabel"),
            FRANCHISE_QID_COL: last_qid(get("franchise")),
            FRANCHISE_LABEL_COL: get("franchiseLabel"),

            # ---- External IDs ----
            BOM_FILM_ID_COL: get("bom_id"),
            IMDB_ID_COL: get("imdb_id"),
            TMDB_ID_COL: get("tmdb_id"),

            # ---- Money fields (raw) ----
            BUDGET_VALUE_COL: float(budget_amount) if budget_amount is not None else None,
            BUDGET_CUR_ID_COL: last_qid(budget_unit_iri),
            BUDGET_CUR_COL: None,  # unit label not returned directly here
            BUDGET_ASOF_COL: budget_asof,
            BUDGET_SOURCE_COL: budget_source,  # "P2130" or "P2769"

            BO_VAL_COL: float(box_amount) if box_amount is not None else None,
            BO_CUR_ID_COL: last_qid(box_unit_iri),
            BO_CUR_COL: None,  # unit label not returned directly here
            BO_ASOF_COL: box_asof,
            BO_REGION_ID_COL: last_qid(box_region_iri),
            BO_REGION_COL: box_region_label,
        })
    if not rows:
        return pd.DataFrame(columns=[
            ROTTEN_ID_COL, WD_FILM_QID_COL, WD_FILM_LABEL_COL,
            SERIES_QID_COL, SERIES_LABEL_COL,
            FRANCHISE_QID_COL, FRANCHISE_LABEL_COL,
            BOM_FILM_ID_COL, IMDB_ID_COL, TMDB_ID_COL,
            BUDGET_VALUE_COL, BUDGET_CUR_ID_COL, BUDGET_CUR_COL, BUDGET_ASOF_COL, BUDGET_SOURCE_COL,
            BO_VAL_COL, BO_CUR_ID_COL, BO_CUR_COL, BO_ASOF_COL,
            BO_REGION_ID_COL, BO_REGION_COL
        ])
    # There should be at most one film per rt_id. Therefore, if duplicates appear, keep the first.
    df = pd.DataFrame(rows).drop_duplicates(subset=[ROTTEN_ID_COL])
    return df


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    """
    Yield successive chunks of given size from iterable.
    0 < size <= len(iterable)
    1,2,3,... -> [1,2], [3,4],
    1,2,3,... -> [1,2,3], [4,5,6], ...
    1,2,3,... -> [1], [2], [3], ...
    :param iterable: input iterable.
    :param size: chunk size.
    :return: yields lists of chunked items.
    """
    chunk = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def rtid_variants(rt_id: str) -> List[str]:
    """
    Generate a small set of variant RT IDs to handle underscore/dash differences.
    Example: 'm/fast_five' -> ['m/fast_five', 'm/fast-five']
             'm/1010644-intolerance' -> ['m/1010644-intolerance', 'm/1010644_intolerance']
    :param rt_id: original RT ID string.
    :return: list of variant RT IDs (including original).
    """
    if not isinstance(rt_id, str):
        return []
    base = rt_id.rstrip(FORWARD_SLASH)
    # Flip underscores <-> dashes only in the slug part (after the first '/')
    if FORWARD_SLASH in base:
        prefix, slug = base.split(FORWARD_SLASH, 1)
    else:
        prefix, slug = "", base

    v = {base}
    if UNDERSCORE in slug:
        v.add(f"{prefix}/{slug.replace(UNDERSCORE, HYPHEN)}" if prefix else slug.replace(UNDERSCORE, HYPHEN))
    if HYPHEN in slug:
        v.add(f"{prefix}/{slug.replace(HYPHEN, UNDERSCORE)}" if prefix else slug.replace(HYPHEN, UNDERSCORE))
    return list(v)


def query_wikidata_for_rt_ids(rt_ids: List[str], *, batch_size: int = 150,
                              user_agent: str = DEFAULT_USER_AGENT) -> pd.DataFrame:
    """
    First pass: query exact RT IDs.
    Second pass: for any misses, try underscore/dash variants.
    :param rt_ids: list of Rotten Tomatoes ID strings (may contain duplicates/invalids).
    :param batch_size: max RT IDs per SPARQL query
                       (max 200 recommended by WDQS; smaller batches use less memory/time).
    :param user_agent: User-Agent header value for WDQS requests.
    :return: DataFrame with one row per original RT ID (if matched).
    """
    # Deduplicate & sanitize
    rt_ids_clean = sorted({rid for rid in rt_ids if isinstance(rid, str) and rid.strip()})
    results: List[pd.DataFrame] = []

    # ---- Pass 1: exact IDs ----
    for chunk in chunked(rt_ids_clean, batch_size):
        pairs = [(rid, rid) for rid in chunk]  # (variant, original)
        q = sparql_for_pairs(pairs)
        js = post_sparql(q, user_agent=user_agent)
        results.append(parse_bindings_to_df(js))

    df1 = pd.concat(results, ignore_index=True) if results else parse_bindings_to_df({})

    matched = set(df1[ROTTEN_ID_COL].dropna()) if not df1.empty else set()
    missing = [rid for rid in rt_ids_clean if rid not in matched]

    if not missing:
        return df1

    # ---- Pass 2: variants for misses ----
    results2: List[pd.DataFrame] = []
    # Divide batches by approx 1/3 size, since each RT ID may generate multiple variants.
    for chunk in chunked(missing, max(1, batch_size // 3)):
        pairs: List[Tuple[str, str]] = []
        for rid in chunk:
            for var in rtid_variants(rid):
                pairs.append((var, rid))
        # If no variants (very rare), skip.
        if not pairs:
            continue
        q = sparql_for_pairs(pairs)
        js = post_sparql(q, user_agent=user_agent)
        results2.append(parse_bindings_to_df(js))
    df2 = pd.concat(results2, ignore_index=True) if results2 else parse_bindings_to_df({})

    if df1.empty:
        return df2

    # Combine, preferring exact pass-1 hits
    combined = pd.concat([df1, df2[~df2[ROTTEN_ID_COL].isin(matched)]], ignore_index=True)
    combined = combined.drop_duplicates(subset=[ROTTEN_ID_COL])
    return combined


def attach_wikidata_franchise_ids(df: pd.DataFrame, rt_col: str = ROTTEN_TOMATOES_ID,
                                  *, id_mode: str = "both", batch_size: int = 150,
                                  user_agent: str = DEFAULT_USER_AGENT) -> pd.DataFrame:
    """
    Enrich a DataFrame with Wikidata franchise/series IDs and other film data by looking up Rotten Tomatoes IDs.
    :param df: input DataFrame containing a column with Rotten Tomatoes IDs.
    :param rt_col: name of the column in df containing Rotten Tomatoes IDs (strings).
    :param id_mode: one of {"series", "ip", "both"}:
        "series" - add a single "franchise_id" column with the series (P179) QID (if any);
        "ip" - add a single "franchise_id" column with the franchise (P8345) QID (if any);
        "both" - add both SERIES_QID_COL and FRANCHISE_QID_COL columns (no single "franchise_id" column).
    :param batch_size: max RT IDs per SPARQL query
                       (max 200 recommended by WDQS; smaller batches use less memory/time).
    :param user_agent: User-Agent header value for WDQS requests.
    :return: DataFrame with additional columns (may have NaNs if no match found).
    """
    if rt_col not in df.columns:
        raise KeyError(f"Column '{rt_col}' not in DataFrame.")

    rt_ids = df[rt_col].astype(str).tolist()
    lookups = query_wikidata_for_rt_ids(rt_ids, batch_size=batch_size, user_agent=user_agent)

    # Left-join back on original rt_id
    out = df.copy()
    out = out.merge(lookups, how="left", left_on=rt_col, right_on=ROTTEN_ID_COL)
    out.drop(columns=[ROTTEN_ID_COL], inplace=True)

    if id_mode in ("series", "ip"):
        out["franchise_id"] = out[SERIES_QID_COL] if id_mode == "series" else out[FRANCHISE_QID_COL]
    elif id_mode == "both":
        # No single franchise_id; keep both columns.
        pass
    else:
        raise ValueError("id_mode must be one of {'series', 'ip', 'both'}")

    return out


if __name__ == "__main__":
    t0 = time.time()
    # Do the enrichment (keep both series & franchise IDs)
    enriched = attach_wikidata_franchise_ids(
        RT_MOVIES_DF[[ROTTEN_TOMATOES_ID]],
        ROTTEN_TOMATOES_ID,
        id_mode="both"
    )

    enriched.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f} seconds.")
