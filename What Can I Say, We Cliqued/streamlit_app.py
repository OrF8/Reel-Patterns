# Reel Patterns: What can I say, We cliqued
#
# Requirements
#   pip install -r requirements.txt
#
# Run
#   python -m streamlit run "What Can I Say, We Cliqued\streamlit_app.py"

import itertools
import os
import networkx as nx
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from collections import Counter, defaultdict
from networkx.algorithms.community.quality import modularity
from typing import List, Tuple, Dict, Set, Union
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

MOVIE_TITLE_COL: str = "tconst"
ACTOR_NAME_COL: str = "name"
ACTOR_NAME_ID_COL: str = "nconst"
WEIGHT: str = "weight"
DEFAULT_SEED: int = 42
MIN_MARKER: int = 8
MAX_MARKER_ADDITION: int = 12
LEGEND_MARKER_SIZE: int = 8
DEFAULT_RESOLUTION: float = 1.0
LEGEND_COMM_FMT: str = f"Community {{id}}"
INVERSE_WEIGHT: str = "inv_w"
NODE_COL: str = "node"
COMMUNITY_COL: str = "community"
BETWEENNESS: str = "betweenness"
CLUSTERING_COL: str = "clustering"
BRIDGING_CENTRALITY: str = "bridging_centrality"
PARTICIPATION: str = "participation"
INTER_DEGREE_COL: str = "inter_degree"
INTRA_DEGREE_COL: str = "intra_degree"
DEGREE_COL: str = "degree"
EDGE_BETWEENNESS: str = "edge_betweenness"
INTER_COMMUNITY: str = "inter_community"
HSL_SATURATION: str = "70%"
HSL_LIGHT: str = "55%"
HSL_FMT: str = f"hsl({{hue}}, {HSL_SATURATION}, {HSL_LIGHT})"
DEFAULT_NODE_COLOR: str = "#5aa9e6"
EDGE_MODE: str = "lines"
EDGE_COLOR: str = "rgba(180,180,200,0.18)"
HIGHLIGHT_EDGE_COLOR: str = "rgba(255,220,90,0.9)"
EDGE_HOVER_INFO: str = "none"
NODE_MODE_IF_LABELS: str = "markers+text"
NODE_MODE_NO_LABELS: str = "markers"
NODE_POS: str = "top center"
NODE_HOVER_INFO: str = "text"
NODE_COLOR: str ="rgba(230,230,240,0.7)"
HIGHLIGHT_NODE_COLOR: str = "#ffd75a"
HIGHLIGHT_NODE_MARKER_COLOR: str = "#111"
FIGS_TEMPLATE_THEME: str = "plotly_dark"
GRAPH_BG_COLOR: str = "#0b0f17"
GRAPH_TITLE: str = "Co-appearance Graph (3D)"
GRAPH_X_ANCHOR: str = "center"
SIDEBAR_TITLE: str = "⚙️ Controls"
LOUVAIN_ALGO: str = "Louvain"
GN_ALGO: str = "Girvan–Newman"
CLIQUE_PERC_ALGO: str = "Clique Percolation"
BETWEENNESS_METRIC: str = "Betweenness"
BRIDGING_CENTRALITY_METRIC: str = "Bridging centrality"
PARTICIPATION_METRIC: str ="Participation"
WARD_LINKAGE: str = "ward"
SITE_TITLE: str  = "🎬 Reel Patterns: Co‑appearance Communities"
SITE_CAPTION: str = "Explore which actors and filmmakers tend to appear together, tune thresholds, and compare community algorithms."
# Explanations for bridge metrics
METRIC_EXPL: Dict[str, str] = {
            BETWEENNESS_METRIC: "high betweenness centrality — they lie on many shortest paths between different communities",
            BRIDGING_CENTRALITY_METRIC: "high bridging centrality - they connect otherwise tight groups",
            PARTICIPATION_METRIC: "high participation coefficient — their neighbors are spread across several communities",
        }
GN_CAPTION_FMT: str = f"""
**Bridge nodes**: The top-{{top_k_nodes}} nodes by {{metric_expl}}.\n
**Bridge edges**: Inter-community edges with the highest edge betweenness (top-{{top_k_nodes}}).\n
*Note:* When you isolate a single community via the legend, inter-community bridge edges are hidden.
"""
IS_LINUX: bool = os.name == "posix"
# Note: The streamlit app is ran from the root directory, so paths are relative to that
DATA_PATH: str = "data/collabs.csv" if IS_LINUX else "data\\collabs.csv"
DENDROGRAM_TITLE_FMT: str = f"Dendrogram Hierarchy (linkage = {{linkage_method}})"
BROWSE_COMM_OPTIONS_MULT_FMT: str = f"#{{i}} – {{size}} actors & filmmakers"
BROWSE_COMM_OPTIONS_SINGLE_FMT: str = f"#{{i}} – 1 actor or filmmaker"


# ---------
# UI CONFIG
# ---------
st.set_page_config(
    page_title="Reel Patterns – Actor Communities",
    page_icon="🎬",
    layout="wide",
)

# -------
# HELPERS
# -------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load the collaboration data from CSV.
    Loads with caching for faster cold starts.
    :return: The actor & filmmakers collaboration DataFrame.
    """
    # Load only necessary columns as strings
    return pd.read_csv(DATA_PATH, dtype=str, usecols=[MOVIE_TITLE_COL, ACTOR_NAME_COL, ACTOR_NAME_ID_COL])


@st.cache_data(show_spinner=False)
def create_id_to_name_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping from actor/filmmaker ID to name.
    :param df: The actor & filmmakers collaboration DataFrame.
    :return: A dictionary mapping nconst -> name.
    """
    return dict(zip(df[ACTOR_NAME_ID_COL], df[ACTOR_NAME_COL]))


@st.cache_data(show_spinner=False)
def build_graph(pairs: pd.DataFrame, min_edge_weight: int = 1) -> nx.Graph:
    """
    Build a co-appearance graph from (name, title) pairs.
    Each node is an actor/filmmaker; an edge (u, v) exists if u and v appeared together in at least
    `min_edge_weight` titles. Edge weight = number of shared titles.
    :param pairs: DataFrame with columns "name" and "tconst".
    :param min_edge_weight: Minimum number of shared titles to create an edge between two actors/filmmakers.
    :return: A graph where nodes are actors/filmmakers and edges represent co-appearances.
    """
    # Group actors by title
    actors_by_title = pairs.groupby(MOVIE_TITLE_COL)[ACTOR_NAME_ID_COL].apply(list).to_dict()
    edge_weights = Counter()
    for cast in actors_by_title.values():
        # unique actors within a title to avoid double counting
        cast = sorted(set(cast))
        for a, b in itertools.combinations(cast, 2):
            edge = (a, b)
            edge_weights[edge] += 1

    graph = nx.Graph()
    for (u, v), w in edge_weights.items():
        # Add edge only if weight meets the minimum threshold
        if w >= min_edge_weight:
            graph.add_edge(u, v, weight=int(w))
    return graph


def graph_to_feature_matrix(graph: nx.Graph, weight: str = WEIGHT) -> tuple[list[str], np.ndarray]:
    """
    Convert a graph to a feature matrix suitable for clustering.
    Each row corresponds to a node, and each column corresponds to the normalized weights of edges to other nodes.
    :param graph: The input graph (NetworkX Graph).
    :param weight: Edge attribute to use as weight (default WEIGHT).
    :return: Tuple of:
             - List of node labels (in the order corresponding to rows of the matrix).
             - 2D numpy array representing the feature matrix.
    """
    nodes = [id_to_name.get(n, n) for n in graph.nodes()]

    # Create an index mapping for nodes
    index = {n: i for i, n in enumerate(graph.nodes())}
    n = len(nodes)

    # Initialize adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, v, d in graph.edges(data=True):
        w = float(d.get(weight, 1.0))
        i, j = index[u], index[v]
        A[i, j] = w
        A[j, i] = w

    # L2 normalize rows (avoid divide-by-zero)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = A / norms
    return nodes, X


def top_n_subgraph(graph: nx.Graph, n: int) -> nx.Graph:
    """
    Returns the subgraph induced by the top N nodes with the highest strength (sum of edge weights).
    If the graph has fewer than N nodes, returns a copy of the original graph.
    :param graph: The input graph (NetworkX Graph).
    :param n: Number of top nodes to include in the subgraph.
    :return: A subgraph containing the top N nodes by strength.
    """
    if len(graph) <= n:
        return graph.copy()
    # Calculate strength (sum of edge weights) for each node
    strength = {node: sum(w.get(WEIGHT, 1) for _, _, w in graph.edges(node, data=True)) for node in graph.nodes()}
    # Sort nodes by strength and take top N
    top_nodes = [node for node, _ in sorted(strength.items(), key=lambda kv: kv[1], reverse=True)[:n]]
    return graph.subgraph(top_nodes).copy()


def run_louvain(graph: nx.Graph, resolution: float = DEFAULT_RESOLUTION, seed: Union[int, None] = DEFAULT_SEED,
                weight: str = WEIGHT) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    Runs Louvain community detection on the given graph.
    :param: graph: The input graph (NetworkX Graph).
    :param: resolution: Resolution parameter for community detection (default DEFAULT_RESOLUTION).
    :param: seed: Random seed for reproducibility (default DEFAULT_SEED).
    :param: weight: Edge attribute to use as weight (default WEIGHT).

    :return: tuple of:
             - membership: A mapping from node -> community_id (0, ..., k-1, size-desc order)
             - communities: A list of communities, where each community is represented as a set of nodes.
                            Sorted in descending order by community size.
    """
    # Run Louvain community detection
    comms = nx.algorithms.community.louvain_communities(graph, weight=weight, resolution=resolution, seed=seed)
    # Sort communities by size (descending)
    comms_sorted = sorted((set(c) for c in comms), key=len, reverse=True)
    # Map nodes to community indices (0, ..., k-1) by size order
    memberships = {node: index for index, community in enumerate(comms_sorted) for node in community}
    return memberships, comms_sorted


def run_girvan_newman(graph: nx.Graph, num_communities: int = 5) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    Runs Girvan–Newman community detection on the given graph.
    :param graph: The input graph (NetworkX Graph).
    :param num_communities: The target number of communities to find (approximate).
    :return: tuple of:
             - membership: A mapping from node -> community_id (0, ..., k-1, size-desc order)
             - communities: A list of communities, where each community is represented as a set of nodes.
                            Sorted in descending order by community size.
    """
    gn_gen = nx.algorithms.community.girvan_newman(graph)
    # Take the partition at depth yielding the desired number of communities (or last available)
    target = None
    for partition in gn_gen:
        if len(partition) >= num_communities:
            target = partition
            break
        target = partition
    if target is None:
        return {}, []
    comms = [set(c) for c in target]
    # Sort communities by size (descending)
    comms_sorted = sorted(comms, key=len, reverse=True)
    # Map nodes to 0, ..., k-1 by size order
    memberships = {node: index for index, c in enumerate(comms_sorted) for node in c}
    return memberships, comms_sorted


def run_k_clique(graph: nx.Graph, k: int = 3) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    Runs k-clique community detection on the given graph.
    :param graph: The input graph (NetworkX Graph)
    :param k: The size of the cliques to find (k)
    :return: tuple of:
             - membership: A mapping from node -> community_id (0, ..., k-1, size-desc order)
             - communities: A list of communities, where each community is represented as a set of nodes.
                            Sorted in descending order by community size.
    """
    # NetworkX k_clique_communities ignores weights; ensure simple graph
    simple_G = nx.Graph()
    simple_G.add_nodes_from(graph.nodes())
    simple_G.add_edges_from(graph.edges())
    comms = list(nx.algorithms.community.k_clique_communities(simple_G, k))
    comms_sorted = sorted([set(c) for c in comms], key=len, reverse=True)
    memberships = {node: index for index, c in enumerate(comms_sorted) for node in c}
    return memberships, comms_sorted


def compute_bridge_metrics(graph: nx.Graph, memberships: Dict[str, int], weight: str = WEIGHT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute various bridge metrics for nodes and edges in the graph, given community memberships.
    :param graph:
    :param memberships:
    :param weight:
    :return: tuple of:
             - df_nodes: node-level metrics (betweenness, clustering, bridging_centrality, participation, inter/intra degree)
             - df_edges: edge-level metrics (edge betweenness, whether it bridges communities)
    """
    # Use inverse weight as "distance" for shortest paths (heavier ties = shorter distance)
    for u, v, d in graph.edges(data=True):
        d[INVERSE_WEIGHT] = 1.0 / float(d.get(weight, 1))

    # Centralities
    node_b = nx.betweenness_centrality(graph, weight=INVERSE_WEIGHT, normalized=True)
    edge_b = nx.edge_betweenness_centrality(graph, weight=INVERSE_WEIGHT, normalized=True)

    # Local clustering (weighted)
    clust = nx.clustering(graph, weight=weight)

    # Bridging centrality: betweenness * (1 - clustering)
    bridging = {n: node_b.get(n, 0.0) * (1.0 - clust.get(n, 0.0)) for n in graph.nodes()}

    # Inter- vs intra-community degree
    deg: Dict[str, int] = dict(graph.degree())
    intra = {n: 0 for n in graph.nodes()}
    inter = {n: 0 for n in graph.nodes()}
    neigh_comm = {n: defaultdict(int) for n in graph.nodes()}

    # Count inter/intra edges and neighbors per community
    for u, v in graph.edges():
        cu, cv = memberships.get(u), memberships.get(v)
        if cu == cv:
            intra[u] += 1
            intra[v] += 1
        else:
            inter[u] += 1
            inter[v] += 1
        neigh_comm[u][cv] += 1
        neigh_comm[v][cu] += 1

    # Participation coefficient
    participation = {}
    for n in graph.nodes():
        if deg[n] == 0:
            participation[n] = 0.0
        else:
            # sum of squared fraction of neighbors in each community
            sum_sq = sum((cnt / deg[n]) ** 2 for cnt in neigh_comm[n].values())
            participation[n] = 1.0 - sum_sq

    # Build node DataFrame, sorted by bridging centrality
    df_nodes = pd.DataFrame({
        NODE_COL: list(graph.nodes()),
        COMMUNITY_COL: [memberships.get(n) for n in graph.nodes()],
        BETWEENNESS: [node_b.get(n, 0.0) for n in graph.nodes()],
        CLUSTERING_COL: [clust.get(n, 0.0) for n in graph.nodes()],
        BRIDGING_CENTRALITY: [bridging[n] for n in graph.nodes()],
        PARTICIPATION: [participation[n] for n in graph.nodes()],
        INTER_DEGREE_COL: [inter[n] for n in graph.nodes()],
        INTRA_DEGREE_COL: [intra[n] for n in graph.nodes()],
        DEGREE_COL: [deg[n] for n in graph.nodes()],
    }).sort_values(BRIDGING_CENTRALITY, ascending=False)

    # Build edge DataFrame, sorted by edge betweenness (inter-community edges first)
    edge_rows = []
    for (u, v), eb in edge_b.items():
        cu, cv = memberships.get(u), memberships.get(v)
        edge_rows.append({
            "u": u, "v": v,
            EDGE_BETWEENNESS: eb,
            INTER_COMMUNITY: cu != cv,
            WEIGHT: graph[u][v].get(weight, 1),
            "comm_u": cu, "comm_v": cv
        })

    df_edges = pd.DataFrame(edge_rows).sort_values(
        [INTER_COMMUNITY, EDGE_BETWEENNESS], ascending=[False, False]
    )

    return df_nodes, df_edges


def modularity_safe(graphs: nx.Graph, memberships: Dict[str, int]) -> Union[float, None]:
    """
    Compute modularity of the given partition safely (returns None on failure).
    :param graphs: The input graph (NetworkX Graph).
    :param memberships: A mapping from node -> community_id.
    :return: Modularity value (float) or None if computation fails.
    """
    comms = defaultdict(set)
    for n, c in memberships.items():
        comms[c].add(n)
    return float(modularity(graphs, comms.values(), weight=WEIGHT)) if comms else None


def generate_hsl_color(comm_id: int, total_groups: int) -> str:
    """
    Generate an HSL color string based on the community ID and total number of groups.
    :param comm_id: Community ID (integer).
    :param total_groups: Total number of groups (integer).
    :return: HSL color string.
    """
    return HSL_FMT.format(hue=int(360 * comm_id / max(1, total_groups)))


def make_3d_figure(graph: nx.Graph, memberships: Union[Dict[str, int], None],
                   highlight_nodes: Set[str] = frozenset(),
                   highlight_edges: Set[Tuple[str, str]] = frozenset()) -> go.Figure:
    """
    Create a 3D Plotly figure of the graph, with optional community coloring and highlighted nodes/edges.
    3D layout is computed using a force-directed algorithm.
    Nodes are sized by their strength (weighted degree).
    :param graph: The input graph (NetworkX Graph).
    :param memberships: A mapping from node -> community_id (or None for no coloring).
    :param highlight_nodes: Set of nodes to highlight (bridge nodes).
    :param highlight_edges: Set of edges to highlight (bridge edges).
    :return: A Plotly Figure object.
    """
    # 3D force-like layout (deterministic with seed)
    pos3d = nx.spring_layout(graph, dim=3, seed=DEFAULT_SEED, weight=WEIGHT)

    # Node sizes: weighted degree ("strength")
    strength = {
        u: sum(d.get(WEIGHT, 1) for _, _, d in graph.edges(u, data=True))
        for u in graph.nodes()
    }

    # Scale sizes to a reasonable range
    min_s, max_s = (min(strength.values()), max(strength.values())) if strength else (0, 1)
    def scale_size(s, s_min=min_s, s_max=max_s):
        """
        Scale node size based on strength to a range of [8, 20].
        :param s: The strength of the node.
        :param s_min: The minimum strength in the graph.
        :param s_max: The maximum strength in the graph.
        :return: Scaled size (float).
        """
        # map to a reasonable marker size range
        if s_max == s_min:
            return (MIN_MARKER + MAX_MARKER_ADDITION) / 2
        return MIN_MARKER + MAX_MARKER_ADDITION * (s - s_min) / (s_max - s_min)

    traces = []

    # Highlighted edges + their endpoints (grouped under the same legend item)
    h_ex, h_ey, h_ez = [], [], []
    bridge_group = "bridges"

    for (u, v) in highlight_edges:
        if u in graph and v in graph:
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            h_ex += [x0, x1, None]
            h_ey += [y0, y1, None]
            h_ez += [z0, z1, None]

    if h_ex:
        traces.append(go.Scatter3d(
            x=h_ex, y=h_ey, z=h_ez, mode=EDGE_MODE,
            line=dict(width=4, color=HIGHLIGHT_EDGE_COLOR),
            hoverinfo=EDGE_HOVER_INFO, showlegend=True, name="Bridge edges",
            legendgroup=bridge_group
        ))

        # the nodes at the ends of the highlighted edges (auto-shown/hidden with the group)
        be_nodes = set()
        for (u, v) in highlight_edges:
            if u in graph and v in graph:
                be_nodes.add(u)
                be_nodes.add(v)

        bx = [pos3d[n][0] for n in be_nodes]
        by = [pos3d[n][1] for n in be_nodes]
        bz = [pos3d[n][2] for n in be_nodes]
        # labels for endpoints
        b_labels = [id_to_name.get(n, n) for n in be_nodes]

        traces.append(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode=NODE_MODE_IF_LABELS if show_labels else NODE_MODE_NO_LABELS,
            text=b_labels if show_labels else None,
            textposition=NODE_POS if show_labels else None,
            hovertext=b_labels,
            hoverinfo=NODE_HOVER_INFO,
            marker=dict(
                size=6,
                color=HIGHLIGHT_EDGE_COLOR,
                line=dict(width=1.2, color=HIGHLIGHT_NODE_MARKER_COLOR)
            ),
            name="Bridge edge nodes",
            showlegend=False,  # no separate legend item
            legendgroup=bridge_group
        ))

    n_comm: int = 0

    if memberships:
        # community-colored rendering (multiple traces)
        n_comm = max(memberships.values()) + 1

        for cid in range(n_comm):  # for each community, create a trace for its nodes and intra-community edges
            group_id = f"comm{cid}"
            color = generate_hsl_color(cid, n_comm)

            # nodes in this community
            nodes_in_comm = [n for n in graph.nodes() if memberships.get(n) == cid]
            if not nodes_in_comm:
                continue

            xs = [pos3d[n][0] for n in nodes_in_comm]
            ys = [pos3d[n][1] for n in nodes_in_comm]
            zs = [pos3d[n][2] for n in nodes_in_comm]
            sizes = [scale_size(strength[n]) for n in nodes_in_comm]
            text_labels = [id_to_name.get(n, n) for n in nodes_in_comm] if show_labels else None
            hover_texts = [
                f"<span style='color:{color}'>{id_to_name.get(n, n)} (Community {cid + 1})</span>"
                for n in nodes_in_comm
            ]

            # intra-community edges for this community
            ex, ey, ez = [], [], []
            for u, v in graph.edges(nodes_in_comm):
                if memberships.get(u) == cid and memberships.get(v) == cid:
                    x0, y0, z0 = pos3d[u]
                    x1, y1, z1 = pos3d[v]
                    ex += [x0, x1, None]
                    ey += [y0, y1, None]
                    ez += [z0, z1, None]

            # edges (no legend item, but grouped with nodes)
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez, mode=EDGE_MODE,
                line=dict(width=1, color=EDGE_COLOR),
                hoverinfo=EDGE_HOVER_INFO,
                showlegend=False, legendgroup=group_id
            ))

            # nodes (no legend)
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode=NODE_MODE_IF_LABELS if show_labels else NODE_MODE_NO_LABELS,
                text=text_labels, textposition=NODE_POS,
                hovertext=hover_texts, hoverinfo=NODE_HOVER_INFO,
                marker=dict(size=sizes, color=color, line=dict(width=0.5, color=NODE_COLOR)),
                name=LEGEND_COMM_FMT.format(id=cid+1), showlegend=False, legendgroup=group_id
            ))

            # legend-only dummy with controlled size
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=LEGEND_MARKER_SIZE, color=color,
                            line=dict(width=0.5, color=NODE_COLOR)),
                name=LEGEND_COMM_FMT.format(id=cid+1), showlegend=True, legendgroup=group_id
            ))

    else:
        # fallback: original single-trace rendering (no communities)
        ex, ey, ez = [], [], []
        for u, v in graph.edges():
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
            ez += [z0, z1, None]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode=EDGE_MODE,
            line=dict(width=1, color=EDGE_COLOR),
            hoverinfo=EDGE_HOVER_INFO, showlegend=False, name="Edges"
        ))

        nx_list = [pos3d[n][0] for n in graph.nodes()]
        ny_list = [pos3d[n][1] for n in graph.nodes()]
        nz_list = [pos3d[n][2] for n in graph.nodes()]
        node_sizes = [scale_size(strength[n]) for n in graph.nodes()]
        node_texts = [id_to_name.get(n, n) for n in graph.nodes()] if show_labels else None
        hover_texts = [f"{n}" for n in graph.nodes()]
        traces.append(go.Scatter3d(
            x=nx_list, y=ny_list, z=nz_list,
            mode=NODE_MODE_IF_LABELS if show_labels else NODE_MODE_NO_LABELS,
            text=node_texts, textposition=NODE_POS,
            hovertext=hover_texts, hoverinfo=NODE_HOVER_INFO,
            marker=dict(size=node_sizes, color=DEFAULT_NODE_COLOR, line=dict(width=0.5, color=NODE_COLOR)),
            name="Nodes", showlegend=False,
        ))

    # highlighted nodes (unchanged; stays as its own legend item)
    mask = [n in highlight_nodes for n in graph.nodes()]
    if any(mask):
        hx = [pos3d[n][0] for n, m in zip(graph.nodes(), mask) if m]
        hy = [pos3d[n][1] for n, m in zip(graph.nodes(), mask) if m]
        hz = [pos3d[n][2] for n, m in zip(graph.nodes(), mask) if m]
        highlighted_labels = [id_to_name.get(n, n) for n, m in zip(graph.nodes(), mask) if m] if show_labels else None
        highlighted_texts = [(f"⭐ <span style='color:{generate_hsl_color(memberships.get(n, -1), n_comm)}'>"
                              f"{id_to_name.get(n, n)} "
                              f"(Community {memberships.get(n, -1) + 1 if memberships else 'N/A'})</span>")
                             for n, m in zip(graph.nodes(), mask) if m]
        highlighted_sizes = [max(12, scale_size(strength[n]) + 6) for n, m in zip(graph.nodes(), mask) if m]
        traces.append(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode=NODE_MODE_IF_LABELS if show_labels else NODE_MODE_NO_LABELS,
            text=highlighted_labels, textposition=NODE_POS,
            hovertext=highlighted_texts,
            hoverinfo=NODE_HOVER_INFO,
            marker=dict(size=highlighted_sizes, color=HIGHLIGHT_NODE_COLOR,
                        line=dict(width=1.5, color=HIGHLIGHT_NODE_MARKER_COLOR)),
            name="Bridge nodes", showlegend=True
        ))

    figure = go.Figure(data=traces)
    figure.update_layout(
        width=1200,
        height=800,
        template=FIGS_TEMPLATE_THEME,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=GRAPH_BG_COLOR,
        ),
        title=dict(text=GRAPH_TITLE, x=0.5, xanchor=GRAPH_X_ANCHOR),
        uirevision="keep",
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02, x=0.5, xanchor=GRAPH_X_ANCHOR,
            itemclick="toggleothers", groupclick="togglegroup",
            itemsizing="trace"
        )
    )
    return figure



# ----------------
# SIDEBAR CONTROLS
# ----------------
st.sidebar.title(SIDEBAR_TITLE)
with st.sidebar:
    algo = st.selectbox("Community algorithm", [LOUVAIN_ALGO, GN_ALGO, CLIQUE_PERC_ALGO], index=0)

    if algo == LOUVAIN_ALGO:
        res = st.slider("Resolution parameter", 0.1, 2.0, DEFAULT_RESOLUTION, 0.1)
    if algo == GN_ALGO:
        show_bridges = st.checkbox("Highlight bridges 🔎", value=(algo == GN_ALGO))
        bridge_metric = st.selectbox("Bridge metric (nodes)", [BETWEENNESS_METRIC, BRIDGING_CENTRALITY_METRIC, PARTICIPATION_METRIC],
                                     index=1)
        top_k_nodes = st.slider("Top bridge nodes", 1, 100, 20, 1)
    if algo == CLIQUE_PERC_ALGO:
        cp_k = st.slider("k‑clique size (k)", 2, 10, 3, 1)

    top_n = st.slider("Number of actors and filmmakers (top‑N by total collaborations)", 25, 500, 200, 25)
    min_w = st.slider("Minimum shared titles between actors and filmmakers (min. edge weight)", 1, 10, 2, 1)

    show_labels = st.checkbox("Show names on nodes", value=False)

    show_dendro = st.checkbox("Show dendrogram", value=False)
    if show_dendro:
        linkage_method = st.selectbox("Dendrogram linkage", ["average", "complete", "single", WARD_LINKAGE], index=0)

st.title(SITE_TITLE)
st.caption(SITE_CAPTION)

# Load the dataset
df = load_data()

# Build a lookup from nconst to name, for name labels
id_to_name = create_id_to_name_map(df)

# ------------
# DATA & GRAPH
# ------------
with st.spinner("Building graph…"):
    G_full = build_graph(df, min_edge_weight=min_w)
    if len(G_full) == 0:
        st.warning("No nodes after filtering. Try lowering the min edge weight.")
        st.stop()
    G = top_n_subgraph(G_full, top_n)

# -------------------
# COMMUNITY DETECTION
# -------------------
with st.spinner(f"Running {algo}…"):
    membership = None
    communities = []

    if algo == LOUVAIN_ALGO:
        membership, communities = run_louvain(G, resolution=res)
    elif algo == GN_ALGO:
        membership, communities = run_girvan_newman(G)
    else:  # Clique Percolation
        membership, communities = run_k_clique(G, k=cp_k)

# Some nodes may be unassigned (e.g., CP finds none) → put them in their own group
if membership:
    unassigned = [n for n in G.nodes() if n not in membership]
    next_id = (max(membership.values()) + 1) if membership else 0
    for i, n in enumerate(unassigned):
        membership[n] = next_id + i

    # Ensure communities includes everything in `membership`
    if not communities:
        # Reconstruct from membership if the algorithm didn't return communities
        comm_map = defaultdict(set)
        for node, cid in membership.items():
            comm_map[cid].add(node)
        communities = list(comm_map.values())
    else:
        # Append singleton communities for newly assigned nodes
        communities.extend([{n} for n in unassigned])

    # Sort by size (largest first) to keep UI consistent
    communities = sorted(communities, key=len, reverse=True)

# ----------------------------------------------
# BRIDGE METRICS (only after we have membership)
# ----------------------------------------------
bridge_nodes = set()
bridge_edges = set()
df_nodes = None
df_edges = None

if algo == GN_ALGO and show_bridges and membership:
    df_nodes, df_edges = compute_bridge_metrics(G, membership)

    metric_map = {
        BETWEENNESS_METRIC: BETWEENNESS,
        BRIDGING_CENTRALITY_METRIC: BRIDGING_CENTRALITY,
        PARTICIPATION_METRIC: PARTICIPATION,
    }
    metric_col = metric_map[bridge_metric]

    # top‑k bridge nodes by chosen metric
    bridge_nodes = set(df_nodes.nlargest(top_k_nodes, metric_col)[NODE_COL])

    # top-k inter‑community edges by edge betweenness
    top_e = df_edges[df_edges[INTER_COMMUNITY]].nlargest(top_k_nodes, EDGE_BETWEENNESS)
    bridge_edges = set(tuple(sorted((r.u, r.v))) for r in top_e.itertuples(index=False))

# ---------------------------
# METRICS & COMMUNITY BROWSER
# ---------------------------
left, right = st.columns([2, 1])
with left:
    mod = modularity_safe(G, membership) if membership else None
    st.metric("Nodes", f"{G.number_of_nodes():,}")
    st.metric("Edges", f"{G.number_of_edges():,}")
    if membership:
        st.metric("Communities", f"{len(set(membership.values())):,}")
    if mod is not None:
        st.metric("Modularity (weighted)", f"{mod:.3f}")

with right:
    if communities:
        options = [
            BROWSE_COMM_OPTIONS_MULT_FMT.format(i=i+1, size=len(c)) if len(c) > 1
            else BROWSE_COMM_OPTIONS_SINGLE_FMT.format(i=i+1)
            for i, c in enumerate(communities)
        ]
        sel = st.selectbox("Browse communities", options) if options else None
        if sel:
            idx = options.index(sel)
            st.write(
                ", ".join(sorted([id_to_name.get(n, n) for n in communities[idx]])[:100]) +
                (" …" if len(communities[idx]) > 100 else "")
            )

# -------------
# VISUALIZATION
# -------------
with st.spinner("Rendering 3D network…"):
    fig = make_3d_figure(
        G, membership,
        highlight_nodes=bridge_nodes,
        highlight_edges=bridge_edges,
    )
    st.plotly_chart(fig, use_container_width=False, config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso3d", "select2d", "lasso2d"],
    })

    st.caption(
        "Tip: If labels are cluttered, hide them and use hover tooltips. Increase min. edge weight to focus on stronger ties."
    )

    # Explain what "Bridge nodes/edges" mean (shown only when GN + bridges enabled)
    if algo == GN_ALGO and 'show_bridges' in globals() and show_bridges:
        st.caption(
            GN_CAPTION_FMT.format(
                top_k_nodes=top_k_nodes,
                metric_expl=METRIC_EXPL[bridge_metric]
            )
        )

if show_dendro:
    with st.spinner("Clustering & rendering dendrogram…"):
        labels, X = graph_to_feature_matrix(G, weight=WEIGHT)

        # Filter out rows with all zeros or invalid values
        valid_rows = np.isfinite(X).all(axis=1) & (np.linalg.norm(X, axis=1) > 0)
        X = X[valid_rows]
        labels = [label for i, label in enumerate(labels) if valid_rows[i]]

        # Distance metric: cosine by default; ward requires euclidean
        if linkage_method == WARD_LINKAGE:
            dist_func = lambda Y: pdist(Y, metric="euclidean")
        else:
            dist_func = lambda Y: pdist(Y, metric="cosine")

        # Build dendrogram
        dendro_fig = ff.create_dendrogram(
            X,
            labels=labels,
            distfun=dist_func,
            orientation="left",
            linkagefun=lambda D: linkage(D, method=linkage_method),
        )
        dendro_fig.update_layout(
            width=900,
            height=18 * len(labels) + 100,  # auto-height by number of nodes
            template=FIGS_TEMPLATE_THEME,
            margin=dict(l=0, r=0, t=30, b=0),
            title=DENDROGRAM_TITLE_FMT.format(linkage_method=linkage_method)
        )
        # Make axis text a bit smaller if many labels
        dendro_fig.update_xaxes(visible=True, showgrid=False)
        dendro_fig.update_yaxes(tickfont=dict(size=10))

        st.plotly_chart(dendro_fig, use_container_width=True, config={"displaylogo": False})

