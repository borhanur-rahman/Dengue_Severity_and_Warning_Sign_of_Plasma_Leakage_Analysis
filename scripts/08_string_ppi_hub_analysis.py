#!/usr/bin/env python3

"""Step 08 — automated STRING PPI and hub-gene analysis.

Purpose
-------
Construct a STRING protein-protein interaction (PPI) network independently
for each dengue analysis axis and rank candidate hub genes using
cytoHubba-compatible network metrics.

The two analysis axes are:

    1. warning_sign
    2. severity

Input
-----
Per-axis enriched-gene lists prepared by Step 07:

scripts_outcomes/runs/<RUN_ID>/ppi/
    warning_sign_ppi_genes.txt
    severity_ppi_genes.txt

The genes are fixed before this step. No additional feature selection or
enrichment analysis is performed here.

Output
------
Results are written to:

scripts_outcomes/runs/<RUN_ID>/ppi_hub_analysis/<axis>/

For each axis, the script produces:

    string_nodes.tsv
    string_edges.tsv
    string_network.graphml
    hub_scores_all.tsv
    hub_top10_MCC.tsv
    hub_top10_Degree.tsv
    hub_top10_MNC.tsv
    analysis_summary.tsv
    unmapped_or_disconnected_input_genes.txt

A cross-axis summary is also written to:

    scripts_outcomes/runs/<RUN_ID>/ppi_hub_analysis/
        ppi_hub_analysis_summary.tsv

STRING network retrieval
------------------------
Networks are retrieved from the STRING API using:

    organism: Homo sapiens (NCBI taxonomy ID 9606)
    network type: functional
    minimum interaction score: ppi.string_min_score

The STRING minimum score is read from config/paths.yaml. API calls use
retry/backoff handling and a fixed request timeout.

Hub-ranking methods
-------------------
The primary ranking method is read from:

    ppi.cytohubba_method

The script calculates three cytoHubba-compatible network measures:

MCC
    Maximal Clique Centrality. For each node, the score is the sum of
    (|C|-1)! over all maximal cliques C containing that node.

Degree
    Number of direct interaction partners in the retained PPI network.

MNC
    Maximum Neighborhood Component, defined here as the size of the largest
    connected component in the subgraph induced by a node's neighbours.

MCC is the configured primary method in the current pipeline. Degree and MNC
are retained as secondary/sensitivity rankings.

Candidate hub genes
-------------------
Top-ranked genes are computationally identified network hubs and should be
described as candidate hub genes unless independently validated.

Optional Cytoscape export
-------------------------
If Cytoscape is running locally and py4cytoscape is installed, the script
can additionally import and style the GraphML network, apply a layout,
export a PNG image, and save a Cytoscape session.

Failure or absence of Cytoscape/py4cytoscape does not affect STRING network
retrieval or hub-score calculations.

Reproducibility
---------------
The run configuration records:

    - analysis axes
    - STRING minimum score
    - primary hub-ranking method
    - number of reported top hubs
    - Homo sapiens taxonomy ID

STRING is an externally maintained resource. For final manuscript reporting,
also record the STRING database/version and network retrieval date used for
the final analysis.
"""



from __future__ import annotations



from math import factorial

from pathlib import Path

import time



import networkx as nx

import numpy as np

import pandas as pd

import requests



from paths import CFG, seed_everything, resolve, read_dir, run_dir





seed_everything()



RUN_ID = None

AXES = list(CFG["ppi"]["axes"])

STRING_MIN_SCORE = float(CFG["ppi"]["string_min_score"])

PRIMARY_METHOD = str(CFG["ppi"]["cytohubba_method"]).upper()

TOP_HUBS = 10

SPECIES_TAXON = 9606

STRING_API = "https://string-db.org/api"

CALLER_IDENTITY = "Dengue_Project"

REQUEST_TIMEOUT = 120

MAX_RETRIES = 5





def read_gene_list(path: Path) -> list[str]:

    if not path.exists():

        raise FileNotFoundError(f"Missing input: {path}. Run Step 07 first.")

    genes = [x.strip() for x in path.read_text().splitlines() if x.strip()]

    return list(dict.fromkeys(genes))





def string_request(endpoint: str, identifiers: list[str]) -> list[dict]:

    """POST a gene list to STRING with retry/backoff."""

    url = f"{STRING_API}/json/{endpoint}"

    payload = {

        "identifiers": "\r".join(identifiers),

        "species": SPECIES_TAXON,

        "required_score": int(round(STRING_MIN_SCORE * 1000)),

        "network_type": "functional",

        "caller_identity": CALLER_IDENTITY,

    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)

            response.raise_for_status()

            return response.json()

        except (requests.RequestException, ValueError) as exc:

            if attempt == MAX_RETRIES - 1:

                raise RuntimeError(

                    f"STRING request failed after {MAX_RETRIES} attempts: {exc}"

                ) from exc

            wait = 2 ** attempt

            print(f"  STRING request failed; retrying in {wait}s", flush=True)

            time.sleep(wait)

    return []





def retrieve_string_network(genes: list[str]) -> pd.DataFrame:

    records = string_request("network", genes)

    if not records:

        return pd.DataFrame(

            columns=["preferredName_A", "preferredName_B", "score"]

        )

    edges = pd.DataFrame(records)

    required = {"preferredName_A", "preferredName_B", "score"}

    missing = required - set(edges.columns)

    if missing:

        raise RuntimeError(f"Unexpected STRING response; missing columns: {missing}")

    edges = edges.sort_values(

        ["score", "preferredName_A", "preferredName_B"],

        ascending=[False, True, True],

    )

    edges = edges.drop_duplicates(

        subset=["preferredName_A", "preferredName_B"], keep="first"

    )

    return edges.reset_index(drop=True)





def build_graph(genes: list[str], edges: pd.DataFrame) -> nx.Graph:

    graph = nx.Graph()

    graph.add_nodes_from(genes, input_gene=True)

    for row in edges.itertuples(index=False):

        a = str(row.preferredName_A)

        b = str(row.preferredName_B)

        if a == b:

            continue

        score = float(row.score)

        if graph.has_edge(a, b):

            graph[a][b]["combined_score"] = max(

                graph[a][b]["combined_score"], score

            )

        else:

            graph.add_edge(a, b, combined_score=score)

    return graph





def mcc_scores(graph: nx.Graph) -> dict[str, int]:

    """Maximal Clique Centrality used by cytoHubba."""

    scores = {str(node): 0 for node in graph.nodes}

    for clique in nx.find_cliques(graph):

        weight = factorial(len(clique) - 1)

        for node in clique:

            scores[str(node)] += weight

    return scores





def mnc_score(graph: nx.Graph, node: str) -> int:

    neighbours = list(graph.neighbors(node))

    if not neighbours:

        return 0

    neighbourhood = graph.subgraph(neighbours)

    return max((len(c) for c in nx.connected_components(neighbourhood)), default=0)





def score_nodes(graph: nx.Graph) -> pd.DataFrame:

    mcc = mcc_scores(graph)

    rows = []

    for node in graph.nodes:

        rows.append(

            {

                "gene": str(node),

                "MCC": mcc[str(node)],

                "Degree": int(graph.degree(node)),

                "MNC": int(mnc_score(graph, node)),

                "weighted_degree": float(

                    sum(d.get("combined_score", 0.0)

                        for _, _, d in graph.edges(node, data=True))

                ),

                "is_isolated": graph.degree(node) == 0,

            }

        )

    return pd.DataFrame(rows)





def rank_scores(scores: pd.DataFrame, method: str) -> pd.DataFrame:

    if method not in scores.columns:

        raise ValueError(f"Unsupported hub method: {method}")

    ranked = scores.sort_values(

        [method, "Degree", "weighted_degree", "gene"],

        ascending=[False, False, False, True],

    ).reset_index(drop=True)

    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    return ranked





def write_graphml(graph: nx.Graph, scores: pd.DataFrame, path: Path) -> None:

    attrs = scores.set_index("gene").to_dict(orient="index")

    for node in graph.nodes:

        for key, value in attrs[str(node)].items():

            if isinstance(value, (np.integer, np.floating, np.bool_)):

                value = value.item()

            graph.nodes[node][key] = value

        graph.nodes[node]["name"] = str(node)

    nx.write_graphml(graph, path)





def optional_cytoscape_export(axis: str, graphml: Path, axis_dir: Path) -> str:

    """Import and style the network when a local Cytoscape session is running."""

    try:

        import py4cytoscape as p4c



        p4c.cytoscape_ping()

    except Exception as exc:

        return f"skipped: Cytoscape/py4cytoscape unavailable ({exc})"



    try:

        p4c.import_network_from_file(str(graphml))

        p4c.rename_network(f"{axis}_STRING_PPI")

        p4c.layout_network("force-directed")



        style_name = f"Dengue_{axis}_MCC"

        defaults = {

            "NODE_SHAPE": "ELLIPSE",

            "NODE_SIZE": 35,

            "NODE_LABEL": "name",

            "NODE_LABEL_FONT_SIZE": 12,

            "NODE_FILL_COLOR": "#4C78A8",

            "EDGE_TRANSPARENCY": 110,

            "EDGE_WIDTH": 1.5,

        }

        try:

            p4c.create_visual_style(style_name, defaults=defaults)

        except Exception:

            pass

        p4c.set_visual_style(style_name)



        node_table = p4c.get_table_columns(columns=["name", "MCC", "Degree"])

        if "MCC" in node_table and len(node_table["MCC"]):

            low = float(pd.Series(node_table["MCC"]).min())

            high = float(pd.Series(node_table["MCC"]).max())

            if high > low:

                p4c.set_node_size_mapping("MCC", [low, high], [30, 85])

                p4c.set_node_color_mapping(

                    "MCC", [low, (low + high) / 2, high],

                    ["#DCEAF7", "#F6C85F", "#D62728"],

                    mapping_type="c",

                )



        p4c.export_image(

            str(axis_dir / f"{axis}_STRING_MCC_network"),

            type="PNG", overwrite_file=True,

        )

        p4c.save_session(str(axis_dir / f"{axis}_STRING_MCC.cys"))

        return "completed"

    except Exception as exc:

        return f"network created but Cytoscape export failed ({exc})"





def analyse_axis(axis: str, ppi_dir: Path, out_root: Path) -> dict:

    gene_file = ppi_dir / f"{axis}_ppi_genes.txt"

    genes = read_gene_list(gene_file)

    axis_dir = out_root / axis

    axis_dir.mkdir(parents=True, exist_ok=True)



    print(f"\n[{axis}] requesting STRING network for {len(genes)} genes", flush=True)

    edges = retrieve_string_network(genes)

    graph = build_graph(genes, edges)

    scores = score_nodes(graph)



    edge_out = edges.copy()

    edge_out.to_csv(axis_dir / "string_edges.tsv", sep="\t", index=False)

    scores.to_csv(axis_dir / "string_nodes.tsv", sep="\t", index=False)

    scores.to_csv(axis_dir / "hub_scores_all.tsv", sep="\t", index=False)



    for method in ("MCC", "Degree", "MNC"):

        rank_scores(scores, method).head(TOP_HUBS).to_csv(

            axis_dir / f"hub_top{TOP_HUBS}_{method}.tsv", sep="\t", index=False

        )



    graphml = axis_dir / "string_network.graphml"

    write_graphml(graph, scores, graphml)



    isolated = sorted(str(n) for n in graph.nodes if graph.degree(n) == 0)

    (axis_dir / "unmapped_or_disconnected_input_genes.txt").write_text(

        "\n".join(isolated) + ("\n" if isolated else "")

    )



    cytoscape_status = optional_cytoscape_export(axis, graphml, axis_dir)

    primary = rank_scores(scores, PRIMARY_METHOD).head(TOP_HUBS)

    print(f"  nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "

          f"isolated={len(isolated)}")

    print(f"  top {PRIMARY_METHOD}: {', '.join(primary['gene'])}")

    print(f"  Cytoscape: {cytoscape_status}")



    return {

        "axis": axis,

        "input_genes": len(genes),

        "network_nodes": graph.number_of_nodes(),

        "network_edges": graph.number_of_edges(),

        "isolated_nodes": len(isolated),

        "string_min_score": STRING_MIN_SCORE,

        "primary_method": PRIMARY_METHOD,

        "top_hubs": ";".join(primary["gene"].tolist()),

        "cytoscape_status": cytoscape_status,

    }





def main() -> None:

    rid = resolve(RUN_ID)

    ppi_dir = read_dir(rid, "ppi")

    out_root = run_dir(

        rid,

        "ppi_hub_analysis",

        config={

            "axes": AXES,

            "string_min_score": STRING_MIN_SCORE,

            "cytohubba_method": PRIMARY_METHOD,

            "top_hubs": TOP_HUBS,

            "species_taxon": SPECIES_TAXON,

        },

    )



    print("=" * 70)

    print("STEP 08 — STRING PPI AND HUB ANALYSIS")

    print("=" * 70)

    print(f"  reading: {ppi_dir}")

    print(f"  writing: {out_root}")

    print(f"  STRING minimum score: {STRING_MIN_SCORE:.3f}")

    print(f"  primary hub method: {PRIMARY_METHOD}")



    summaries = [analyse_axis(axis, ppi_dir, out_root) for axis in AXES]

    pd.DataFrame(summaries).to_csv(

        out_root / "ppi_hub_analysis_summary.tsv", sep="\t", index=False

    )



    print(f"\n[SAVED] -> {out_root}")

    print("[DONE]")





if __name__ == "__main__":

    main()