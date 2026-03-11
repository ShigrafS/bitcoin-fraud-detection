import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
os.makedirs("sample_data", exist_ok=True)

                         
num_nodes = 5000
node_ids = np.arange(100000, 100000 + num_nodes)

              
features = np.zeros((num_nodes, 167))
features[:, 0] = node_ids
features[:, 1] = 1           

features_df = pd.DataFrame(features)
features_df.to_csv("data/elliptic_txs_features.csv", index=False, header=False)

                      
num_edges = 15000
sources = np.random.choice(node_ids, num_edges)
targets = np.random.choice(node_ids, num_edges)
edges_df = pd.DataFrame({"source": sources, "target": targets})
edges_df.to_csv("data/elliptic_txs_edgelist.csv", index=False)

                        
classes = np.random.choice(["unknown", "1", "2"], num_nodes)
classes_df = pd.DataFrame({"txId": node_ids, "class": classes})
classes_df.to_csv("data/elliptic_txs_classes.csv", index=False)

print("Dummy data generated.")
