import streamlit as st
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import plotly.express as px
import os

from utils.inference import run_model
from utils.visualization import plot_graph
from utils.metrics import model_comparison

st.set_page_config(page_title="Fraud Detection GNN", layout="wide")

st.title("💳 Fraud Detection with Graph Neural Networks")

st.markdown("""
Dataset: **Elliptic Bitcoin Dataset**

Models:
- GCN
- GAT
- GraphSAGE
- GIN
- MPNN
- GTN
""")

# Sidebar
st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN", "GTN"]
)

# Dataset Selection
st.sidebar.markdown("---")
st.sidebar.header("Dataset")
dataset_source = st.sidebar.radio("Dataset Source", ["Local Sample", "Upload Custom"])

graph_path = None
uploaded_file = None

if dataset_source == "Local Sample":
    sample_files = [f for f in os.listdir("sample_data") if f.endswith(".pt")]
    if not sample_files:
        st.sidebar.error("No samples found in `sample_data/`! Run `scripts/generate_sample_graph.py` first.")
    else:
        selected_sample = st.sidebar.selectbox("Select Sample Graph", sample_files)
        graph_path = os.path.join("sample_data", selected_sample)
else:
    uploaded_file = st.sidebar.file_uploader("Upload PyTorch Geometric Graph (.pt)", type=["pt"])

run_button = st.sidebar.button("Run Detection")

# Layout
col1, col2 = st.columns([2, 1])

if run_button:
    try:
        if dataset_source == "Local Sample" and graph_path:
            graph = torch.load(graph_path, weights_only=False)
        elif dataset_source == "Upload Custom" and uploaded_file:
            graph = torch.load(uploaded_file, weights_only=False)
        else:
            st.warning("Please select a dataset to proceed.")
            st.stop()
            
    except Exception as e:
        st.error(f"Failed to load the graph: {str(e)}")
        st.stop()

    fraud_nodes, metrics, probabilities = run_model(model_choice, graph)

    fig = plot_graph(graph, fraud_nodes)

    with col1:
        st.subheader("Transaction Graph")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Metrics")
        st.metric("Accuracy", metrics["accuracy"])
        st.metric("F1 Score", metrics["f1"])
        st.metric("ROC AUC", metrics["roc"])
    
    # Optional extensions from the user instructions
    st.markdown("---")
    st.subheader("Model Comparison")
    st.plotly_chart(model_comparison())

    st.markdown("---")
    st.subheader("Fraud Probability Distribution")
    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Node", "Fraud Probability"]
    )
    fig_hist = px.histogram(prob_df, x="Fraud Probability")
    st.plotly_chart(fig_hist)
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Dataset Summary")
        st.write("Total Nodes:", graph.x.shape[0])
        st.write("Total Edges:", graph.edge_index.shape[1])
        st.write("Fraud Nodes Detected:", len(fraud_nodes))

    with res_col2:
        st.subheader("Node Inspector")
        # Generate NetworkX graph to find neighbors
        G = to_networkx(graph, to_undirected=True)
        selected_node = st.selectbox("Inspect Node", list(G.nodes()))
        
        st.write("Node:", selected_node)
        st.write("Fraud Probability:", round(probabilities[selected_node], 4))
        # getting connected neighbors might be large, cap it
        neighbors = list(G.neighbors(selected_node))
        st.write("Connected Nodes:", neighbors[:10], "..." if len(neighbors) > 10 else "")
