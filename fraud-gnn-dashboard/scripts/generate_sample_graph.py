import pandas as pd
import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

FEATURES_FILE = "data/elliptic_txs_features.csv"
EDGES_FILE = "data/elliptic_txs_edgelist.csv"
CLASSES_FILE = "data/elliptic_txs_classes.csv"

OUTPUT_FILE = "sample_data/sample_graph.pt"

SAMPLE_SIZE = 1000

def load_full_graph():
    print("Loading dataset...")

    features = pd.read_csv(FEATURES_FILE, header=None)
    edges = pd.read_csv(EDGES_FILE)
    classes = pd.read_csv(CLASSES_FILE)

    node_ids = features[0].values
    x = torch.tensor(features.iloc[:, 2:].values, dtype=torch.float)

    id_map = {node_id: i for i, node_id in enumerate(node_ids)}

    # Optimize slow replace by mapping each column directly
    source_col = edges.columns[0]
    target_col = edges.columns[1]
    edges[source_col] = edges[source_col].map(id_map)
    edges[target_col] = edges[target_col].map(id_map)
    
    # Drop any edges where the node wasn't found in id_map
    edges = edges.dropna()
    
    edge_index = edges.values.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    labels = classes.set_index("txId").loc[node_ids]["class"]

    y = labels.replace({
        "unknown": -1,
        1: 1,
        2: 0,
        "1": 1,
        "2": 0
    }).fillna(-1).values

    y = torch.tensor(y, dtype=torch.long)
    time_steps = torch.tensor(features[1].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, time_step=time_steps)

    return data

def create_sample(data, target_step=30):
    total_nodes = data.x.shape[0]

    # Sample nodes from a specific time step to preserve edge connections!
    mask = data.time_step == target_step
    sampled_nodes = mask.nonzero().flatten()
    
    # If the time step doesn't exist (e.g. dummy data), fallback
    if len(sampled_nodes) == 0:
        print(f"Time step {target_step} not found, falling back to random sampling.")
        sampled_nodes = torch.tensor(random.sample(range(total_nodes), min(SAMPLE_SIZE, total_nodes)))

    edge_index, _ = subgraph(
        sampled_nodes,
        data.edge_index,
        relabel_nodes=True
    )

    x = data.x[sampled_nodes]
    y = data.y[sampled_nodes]
    # Optionally keep time steps in sample
    ts = data.time_step[sampled_nodes]

    sample_graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=ts
    )

    return sample_graph

def main():
    full_graph = load_full_graph()

    print("Creating sample graph...")

    sample_graph = create_sample(full_graph)

    torch.save(sample_graph, OUTPUT_FILE)

    print("Sample graph saved to:", OUTPUT_FILE)
    print("Nodes:", sample_graph.x.shape[0])
    print("Edges:", sample_graph.edge_index.shape[1])

if __name__ == "__main__":
    main()
