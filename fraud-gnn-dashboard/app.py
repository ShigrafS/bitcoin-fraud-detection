import streamlit as st
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import plotly.express as px
import os

from utils.inference import run_model
from utils.visualization import plot_graph
from utils.metrics import model_comparison, radar_chart, plot_confusion_matrix

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

         
st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN", "GTN", "All Models (Compare)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Settings")
prob_threshold = st.sidebar.slider("Fraud Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                   help="Nodes with probability above this threshold will jump out visually.")

                   
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
    if model_choice == "All Models (Compare)":
        all_metrics = {}
                                            
        models_to_run = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN", "GTN"]
        for m in models_to_run:
            _, m_metrics, m_probs, m_cm = run_model(m, graph)
            all_metrics[m] = m_metrics
            
                                          
        best_model = "GCN"
        fraud_nodes, metrics, probabilities, cm = run_model(best_model, graph)
        st.info(f"Visualizing results for {best_model} by default when running all models.")
    else:
        fraud_nodes, metrics, probabilities, cm = run_model(model_choice, graph)
        all_metrics = {model_choice: metrics}

                               
    filtered_fraud_nodes = [node for node, prob in probabilities.items() if prob >= prob_threshold]

    fig = plot_graph(graph, filtered_fraud_nodes, probabilities)

    with col1:
        st.subheader("Transaction Graph")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Metrics")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Accuracy", metrics["accuracy"])
            st.metric("F1 Score", metrics["f1"])
            st.metric("ROC AUC", metrics["roc"])
        with m_col2:
            st.metric("Precision", metrics.get("precision", "N/A"))
            st.metric("Recall", metrics.get("recall", "N/A"))
            
        st.plotly_chart(plot_confusion_matrix(cm, "Confusion Matrix"), use_container_width=True)
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Dataset Summary")
        G_summary = to_networkx(graph, to_undirected=True)
        degrees = [d for n, d in G_summary.degree()]
        avg_deg = sum(degrees) / len(degrees) if degrees else 0
        
                      
        y_vals = graph.y.tolist()
        normal_cnt = y_vals.count(0)
        fraud_cnt = y_vals.count(1)
        unknown_cnt = y_vals.count(-1)
        
        st.write(f"**Total Nodes:** {graph.x.shape[0]}")
        st.write(f"**Total Edges:** {graph.edge_index.shape[1]}")
        st.write(f"**Average Degree:** {avg_deg:.2f}")
        st.write(f"**Normal Nodes:** {normal_cnt} | **Fraud Logs:** {fraud_cnt} | **Unknown:** {unknown_cnt}")

    with res_col2:
        st.subheader("Node Inspector")
        G = to_networkx(graph, to_undirected=True)
        selected_node = st.selectbox("Inspect Node", list(G.nodes()))
        
        prob = probabilities.get(selected_node, 0)
        label_id = int(graph.y[selected_node].item())
        label_str = {1: "Fraud", 0: "Normal", -1: "Unknown"}.get(label_id, "Unknown")
        ts = int(graph.time_step[selected_node].item()) if hasattr(graph, 'time_step') else "N/A"
        deg = dict(G.degree()).get(selected_node, 0)
        
                                                       
        community_id = list(nx.node_connected_component(G, selected_node))[0]
        
        st.markdown(f"""
        **Node:** `{selected_node}`  
        **Label:** `{label_str}`  
        **Fraud Probability:** `{prob:.4f}`  
        **Time Step:** `{ts}`  
        **Degree:** `{deg}`  
        **Community ID:** `{community_id}`  
        """)
        
        neighbors = list(G.neighbors(selected_node))
        st.write("Connected Nodes:", neighbors[:10], "..." if len(neighbors) > 10 else "")

    st.markdown("---")
    st.subheader("Model Comparison")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.plotly_chart(model_comparison(all_metrics), use_container_width=True)
    with comp_col2:
        radar_fig = radar_chart(all_metrics)
        if radar_fig:
             st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Fraud Probability Distribution (Top 10)")
    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Node", "Fraud Probability"]
    )
    
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fig_hist = px.histogram(prob_df, x="Fraud Probability", nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with dist_col2:
        top_10 = prob_df.sort_values(by="Fraud Probability", ascending=False).head(10)
        st.dataframe(top_10, use_container_width=True)
        
                      
        csv_data = prob_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_data,
            file_name='fraud_predictions.csv',
            mime='text/csv',
        )
