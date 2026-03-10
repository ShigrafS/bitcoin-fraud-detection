# 💳 Bitcoin Fraud Detection with Graph Neural Networks

A comprehensive machine learning project designed to detect illicit transactions within the Bitcoin network using Graph Neural Networks (GNNs). This project leverages the **Elliptic Dataset**, a large-scale transaction graph, and explores the efficacy of various state-of-the-art GNN architectures.

---

## 🚀 Overview

The transparent and pseudonymous nature of blockchains makes them attractive for illicit activities (e.g., money laundering, ransomware routing). This repository contains experiments and interactive tools to detect fraudulent nodes by analyzing the topological structure of transaction graphs. 

### Why Graph Neural Networks?
Traditional tabular machine learning models (like Random Forests or MLPs) only look at isolated transaction features (e.g., timestamp, amount). GNNs excel because they inherently capture the **flow of funds**—learning representations based on a node's neighborhood and transaction history.

---

## 🧠 Models Implemented

We explore and compare 6 distinct Graph Neural Network architectures, varying in their aggregation and message-passing strategies:

1. **GCN (Graph Convolutional Network)**: A powerful baseline that aggregates feature information from direct neighbors.
2. **GAT (Graph Attention Network)**: Utilizes attention mechanisms to dynamically weigh the importance of different neighbor nodes.
3. **GraphSAGE (Graph Sample and Aggregation)**: A highly scalable architecture that learns functions to sample and aggregate structural information from a node's local neighborhood.
4. **GIN (Graph Isomorphism Network)**: Designed to be as powerful as the Weisfeiler-Lehman graph isomorphism test, making it highly expressive for distinguishing graph topologies.
5. **MPNN (Message Passing Neural Network)**: A generalized spatial formulation of GNNs that explicitly models message functions and node updates.
6. **GTN (Graph Transformer Network)**: Adapts Transformer architectures to graph structures, capturing complex and long-range dependencies across the network.

---

## 📁 Repository Structure

- **`fraud-gnn-dashboard/`**: The core interactive Streamlit dashboard allowing users to visualize transaction subgraphs, inspect nodes, and compare model metrics interactively.
- **`models/`**: PyTorch implementations of the 6 GNN architectures (`gcn.py`, `gat.py`, `graphsage.py`, `gin.py`, `mpnn.py`, `gtn.py`).
- **`notebooks/`**: Detailed Jupyter Notebooks (`*.ipynb`) containing exploratory data analysis, graph building, and training loops for individual model evaluation.
- **`utils/`**: Helper modules for loading data, running mock inferences, calculating confusion matrices, and generating complex Plotly network visualizations.

---

## 📊 The Elliptic Dataset

The [Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) is a graph mapping Bitcoin transactions.
- **Nodes**: 203,769 Bitcoin transactions.
- **Edges**: 234,355 directed edges representing the flow of Bitcoin.
- **Features**: 166 features per node (local transaction features + aggregated neighborhood features).
- **Labels**: 
  - `1`: Illicit (Fraud) - 2% of the data
  - `2`: Licit (Normal) - 21% of the data
  - `Unknown`: Unlabeled - 77% of the data

### Running the Project
To explore the data visually or compare models, please visit the `fraud-gnn-dashboard/` directory and refer to its specific `README.md`.
