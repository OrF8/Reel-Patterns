# wikidata_franchise_from_rt.py
# ------------------------------------------------------------
# Map Rotten Tomatoes IDs (P1258) -> Wikidata film item -> series (P179) & franchise (P8345)
# + Add money fields: budget (P2130, fallback P2769), box office (P2142) with currency/qualifiers
# + Add derived money metrics & labels
# Usage (at bottom): df_enriched = attach_wikidata_franchise_ids(df, "rotten_tomatoes_link", id_mode="both")
#
# Requires: pandas, requests
# ------------------------------------------------------------

from __future__ import annotations
import time
import math
from typing import Dict, List, Tuple, Iterable, Optional
import requests
import pandas as pd


WD_ENDPOINT = "https://query.wikidata.org/sparql"

# IMPORTANT: replace with your app/contact so WDQS is happy.
DEFAULT_USER_AGENT = "FranchiseMapper/1.0 (adirtuval@gmail.com)"

# ---- Derived-money-calculation defaults (documented in-field) ----
# Effective studio revenue share on worldwide gross when domestic/international split is unknown.
# Literature/anecdotal rules of thumb: ~50% domestic, ~40% international, ~25% China; here we use a single effective number.
DEFAULT_SHARE_EFFECTIVE = 0.45
# Prints & Advertising (marketing) heuristic when not publicly reported.
DEFAULT_P_AND_A_FACTOR = 0.50  # 50% of production budget


def _escape_literal(s: str) -> str:
    """Escape a string for SPARQL string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _build_values_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    Build a VALUES block mapping variant -> original:
        VALUES (?rtid ?orig) { ("variant1" "original1") ("variant2" "original1") ... }
    """
    rows = []
    for rtid, orig in pairs:
        rows.append(f'("{_escape_literal(rtid)}" "{_escape_literal(orig)}")')
    return " ".join(rows)


def _sparql_for_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    SPARQL: match films by Rotten Tomatoes ID (P1258), return film, series (P179), franchise (P8345),
    and money fields:
      - Budget (preferred rank P2130; fallback to any P2130; fallback to P2769)
      - Box office (preferred rank P2142; fallback to any P2142)
    For quantities, return amount & unit IRI, and qualifiers when present (P585 as-of, P1001 region).
    ?orig carries the original RT id to map results back to your rows.
    """
    values_block = _build_values_pairs(pairs)
    return f"""
SELECT ?orig ?film ?filmLabel ?series ?seriesLabel ?franchise ?franchiseLabel

       ?budget_amount_pref ?budget_unit_pref ?budget_asof_pref
       ?budget_amount_any  ?budget_unit_any  ?budget_asof_any
       ?budget2769_amount  ?budget2769_unit ?budget2769_asof

       ?box_amount_pref ?box_unit_pref ?box_asof_pref ?box_region_pref ?box_region_prefLabel
       ?box_amount_any  ?box_unit_any  ?box_asof_any  ?box_region_any  ?box_region_anyLabel

WHERE {{
  VALUES (?rtid ?orig) {{ {values_block} }}
  ?film wdt:P1258 ?rtid .

  OPTIONAL {{ ?film wdt:P179  ?series. }}
  OPTIONAL {{ ?film wdt:P8345 ?franchise. }}

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


def _post_sparql(query: str,
                 *,
                 user_agent: str = DEFAULT_USER_AGENT,
                 max_retries: int = 5,
                 base_sleep: float = 1.5) -> dict:
    """
    POST a SPARQL query with basic retry/backoff on 429/5xx.
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


def _parse_bindings_to_df(js: dict) -> pd.DataFrame:
    """Convert WDQS JSON results to a tidy DataFrame keyed by original RT id (?orig)."""
    rows = []
    for b in js.get("results", {}).get("bindings", []):
        def get(name: str) -> Optional[str]:
            v = b.get(name, {}).get("value")
            return v if v is not None and v != "" else None

        def last_qid(iri: Optional[str]) -> Optional[str]:
            return iri.split("/")[-1] if iri else None

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
            "rt_id": get("orig"),
            "wd_film_qid": last_qid(get("film")),
            "wd_film_label": get("filmLabel"),
            "series_qid": last_qid(get("series")),
            "series_label": get("seriesLabel"),
            "franchise_qid": last_qid(get("franchise")),
            "franchise_label": get("franchiseLabel"),

            # ---- Money fields (raw) ----
            "budget_value": float(budget_amount) if budget_amount is not None else None,
            "budget_currency_qid": last_qid(budget_unit_iri),
            "budget_currency": None,  # filled by label service implicitly via unit label; not returned directly here
            "budget_asof": budget_asof,
            "budget_source": budget_source,  # "P2130" or "P2769"

            "box_office_value": float(box_amount) if box_amount is not None else None,
            "box_office_currency_qid": last_qid(box_unit_iri),
            "box_office_currency": None,  # filled by label service implicitly via unit label; not returned directly here
            "box_office_asof": box_asof,
            "box_office_region_qid": last_qid(box_region_iri),
            "box_office_region": box_region_label,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "rt_id", "wd_film_qid", "wd_film_label",
            "series_qid", "series_label",
            "franchise_qid", "franchise_label",
            "budget_value", "budget_currency_qid", "budget_currency", "budget_asof", "budget_source",
            "box_office_value", "box_office_currency_qid", "box_office_currency", "box_office_asof",
            "box_office_region_qid", "box_office_region"
        ])
    # There should be at most one film per rt_id; if duplicates appear, keep the first.
    df = pd.DataFrame(rows).drop_duplicates(subset=["rt_id"])
    return df


def _chunked(iterable: Iterable, size: int) -> Iterable[List]:
    chunk = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _rtid_variants(rt_id: str) -> List[str]:
    """
    Generate a small set of variant RT IDs to handle underscore/dash differences.
    Example: 'm/fast_five' -> ['m/fast_five', 'm/fast-five']
             'm/1010644-intolerance' -> ['m/1010644-intolerance', 'm/1010644_intolerance']
    """
    if not isinstance(rt_id, str):
        return []
    base = rt_id.rstrip("/")
    # Flip underscores <-> dashes only in the slug part (after the first '/')
    if "/" in base:
        prefix, slug = base.split("/", 1)
    else:
        prefix, slug = "", base

    v = {base}
    if "_" in slug:
        v.add(f"{prefix}/{slug.replace('_', '-')}" if prefix else slug.replace("_", "-"))
    if "-" in slug:
        v.add(f"{prefix}/{slug.replace('-', '_')}" if prefix else slug.replace("-", "_"))
    return list(v)


def query_wikidata_for_rt_ids(rt_ids: List[str],
                              *,
                              batch_size: int = 150,
                              user_agent: str = DEFAULT_USER_AGENT) -> pd.DataFrame:
    """
    First pass: query exact RT IDs.
    Second pass: for any misses, try underscore/dash variants.
    Returns a dataframe with one row per ORIGINAL rt_id (deduped).
    """
    # Deduplicate & sanitize
    rt_ids_clean = sorted({rid for rid in rt_ids if isinstance(rid, str) and rid.strip()})
    results: List[pd.DataFrame] = []

    # ---- Pass 1: exact IDs ----
    for chunk in _chunked(rt_ids_clean, batch_size):
        pairs = [(rid, rid) for rid in chunk]  # (variant, original)
        q = _sparql_for_pairs(pairs)
        js = _post_sparql(q, user_agent=user_agent)
        results.append(_parse_bindings_to_df(js))
    df1 = pd.concat(results, ignore_index=True) if results else _parse_bindings_to_df({})

    matched = set(df1["rt_id"].dropna()) if not df1.empty else set()
    missing = [rid for rid in rt_ids_clean if rid not in matched]

    if not missing:
        return df1

    # ---- Pass 2: variants for misses ----
    results2: List[pd.DataFrame] = []
    for chunk in _chunked(missing, max(1, batch_size // 3)):
        pairs: List[Tuple[str, str]] = []
        for rid in chunk:
            for var in _rtid_variants(rid):
                pairs.append((var, rid))
        # If no variants (very rare), skip.
        if not pairs:
            continue
        q = _sparql_for_pairs(pairs)
        js = _post_sparql(q, user_agent=user_agent)
        results2.append(_parse_bindings_to_df(js))
    df2 = pd.concat(results2, ignore_index=True) if results2 else _parse_bindings_to_df({})

    if df1.empty:
        return df2

    # Combine, preferring exact pass-1 hits
    combined = pd.concat([df1, df2[~df2["rt_id"].isin(matched)]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["rt_id"])
    return combined


def attach_wikidata_franchise_ids(df: pd.DataFrame,
                                  rt_col: str = "rotten_tomatoes_link",
                                  *,
                                  id_mode: str = "both",
                                  batch_size: int = 150,
                                  user_agent: str = DEFAULT_USER_AGENT) -> pd.DataFrame:
    """
    Enrich `df` with:
      - wd_film_qid, wd_film_label
      - series_qid, series_label  (P179: "part of the series")
      - franchise_qid, franchise_label (P8345: "media franchise")
      - Money fields (raw):
          * budget_value, budget_currency_qid, budget_currency, budget_asof, budget_source
          * box_office_value, box_office_currency_qid, box_office_currency, box_office_asof,
            box_office_region_qid, box_office_region
      - Money fields (derived; heuristic constants documented in columns):
          * p_and_a_est, p_and_a_method, share_effective
          * studio_rentals_est, profit_est, roi_est
          * gross_to_budget_multiple, breakeven_world_est, breakeven_gap
          * money_success_label  (smash_hit / hit / marginal / underperformer / bomb)
      - Optional: franchise_id column (choose 'series' or 'ip' via id_mode)
    id_mode: 'series' | 'ip' | 'both'
    """
    if rt_col not in df.columns:
        raise KeyError(f"Column '{rt_col}' not in DataFrame.")

    rt_ids = df[rt_col].astype(str).tolist()
    lookups = query_wikidata_for_rt_ids(rt_ids, batch_size=batch_size, user_agent=user_agent)

    # Left-join back on original rt_id
    out = df.copy()
    out = out.merge(lookups, how="left", left_on=rt_col, right_on="rt_id")
    out.drop(columns=["rt_id"], inplace=True)

    # ----- Fill human-readable currency labels when possible -----
    # We didn't fetch unit labels directly as separate columns (SERVICE label fills them server-side),
    # so here we leave *_currency as None unless you want to post-process via another lookup.
    # You can map currency_qid -> code externally if needed.

    # ----- Derived money metrics -----
    # Safe numeric conversions
    def _to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    out["budget_value"] = out["budget_value"].apply(_to_float)
    out["box_office_value"] = out["box_office_value"].apply(_to_float)

    # Heuristic parameters (copy into columns for auditability)
    out["p_and_a_method"] = f"{int(DEFAULT_P_AND_A_FACTOR*100)}% of production budget (heuristic)"
    out["share_effective"] = DEFAULT_SHARE_EFFECTIVE

    # P&A estimate
    out["p_and_a_est"] = out["budget_value"] * DEFAULT_P_AND_A_FACTOR

    # Studio rentals estimate (effective share on worldwide gross)
    out["studio_rentals_est"] = out["box_office_value"] * DEFAULT_SHARE_EFFECTIVE

    # Profit estimate
    out["profit_est"] = out["studio_rentals_est"] - (
        out["budget_value"] + out["p_and_a_est"]
    )

    # ROI estimate on cash out (budget + P&A)
    denom = out["budget_value"] + out["p_and_a_est"]
    out["roi_est"] = out["profit_est"] / denom

    # Gross-to-budget multiple (naive)
    out["gross_to_budget_multiple"] = out["box_office_value"] / out["budget_value"]

    # Breakeven worldwide gross estimate and gap
    out["breakeven_world_est"] = (out["budget_value"] + out["p_and_a_est"]) / DEFAULT_SHARE_EFFECTIVE
    out["breakeven_gap"] = out["box_office_value"] - out["breakeven_world_est"]

    # Money success label (conservative, share-aware rubric)
    def _label(roi):
        if roi is None or (isinstance(roi, float) and (math.isnan(roi) or math.isinf(roi))):
            return None
        if roi >= 1.0:
            return "smash_hit"
        if roi >= 0.3:
            return "hit"
        if roi >= -0.2:
            return "marginal"
        if roi >= -0.5:
            return "underperformer"
        return "bomb"

    out["money_success_label"] = out["roi_est"].apply(_label)

    if id_mode in ("series", "ip"):
        out["franchise_id"] = out["series_qid"] if id_mode == "series" else out["franchise_qid"]
    elif id_mode == "both":
        # No single franchise_id; keep both columns.
        pass
    else:
        raise ValueError("id_mode must be one of {'series', 'ip', 'both'}")

    return out


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":

    rt_movies_df = pd.read_csv("/homes/adirt/repos/py/packages/NeedleInADataHaystack/data/rotten_tomatoes_movies.csv")
    # Do the enrichment (keep both series & franchise IDs)
    enriched = attach_wikidata_franchise_ids(rt_movies_df[["rotten_tomatoes_link"]], "rotten_tomatoes_link", id_mode="both")
    # If you want a single franchise_id, pick 'series' or 'ip':
    # enriched = attach_wikidata_franchise_ids(df, "rotten_tomatoes_link", id_mode="series")

    enriched.to_csv("./result.csv")

    # Show results
    # pd.set_option("display.max_columns", None)
    # print(enriched)
