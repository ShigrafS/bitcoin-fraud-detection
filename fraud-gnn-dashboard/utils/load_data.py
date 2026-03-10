import pandas as pd
import networkx as nx

def load_graph(file):
    # just in case we need to load a csv
    df = pd.read_csv(file)

    G = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target"
    )

    return G
