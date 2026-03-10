# 💳 Fraud Detection with Graph Neural Networks Dashboard

A fully interactive Streamlit application designed to visualize and analyze Bitcoin transaction fraud subgraphs using various Graph Neural Network (GNN) architectures on the Elliptic dataset.

## ✨ Features

- **Multi-Model Support**: Easily toggle between testing standard GNNs including **GCN, GAT, GraphSAGE, GIN, MPNN**, and **GTN**.
- **Run All / Compare**: Execute all models sequentially to generate comparative radar charts and multi-bar performance graphics.
- **Organic Graph Visualization**: Features a highly aesthetic, force-directed network layout. Node sizes scale by degree (transaction connections) and colors represent a continuous probability gradient stretching from Green (Safe) to Red (High Fraud Prob).
- **Deep Node Inspection**: Hover over nodes to instantly view `ID`, `Label`, `Time Step`, `Degree`, and `Fraud Probability`.
- **Threshold Filtering**: An interactive slider lets you instantly hide low-probability noise, visually isolating the most suspicious transaction chains.
- **Rich Diagnostic Metrics**: Beyond standard Accuracy, it provides operational metrics including F1 Score, ROC AUC, Precision, Recall, and a visual **Confusion Matrix** heatmap for evaluating False Positives/Negatives.
- **Data Exporting**: 1-click CSV download configuration to export predicted scores out of the dashboard.

---

## 🛠️ Setup Instructions

### 1. Install Requirements
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Add the Dataset
Download the [Elliptic Dataset from Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). 
Extract it and place the following three main files strictly inside the `data/` directory:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

*(Note: These files are heavily ignored by Git due to their size)*

### 3. Generate Sample Graph
Because the full Elliptic dataset consists of >200,000 nodes, it cannot be rendered interactively in a web browser without severe lag. 
Before running the dashboard, please generate a structural subgraph by running:
```bash
python scripts/generate_sample_graph.py
```
This script intelligently extracts nodes from a specific temporal `time_step` (e.g., `30`), ensuring that the resulting `.pt` sample graph retains its dense, organic edge connectivity rather than generating a disconnected random sample.

---

## 🚀 Running the App

To run the Streamlit dashboard on Windows, simply execute the included batch script:
```cmd
./run.bat
```

> **Windows Note**: If you occasionally see a `Fatal error in launcher: Unable to create process` when directly calling `streamlit run app.py` on Windows (due to broken pip executable paths), the `run.bat` script bypasses this reliably by utilizing the `python -m streamlit` module runner instead. 

### Manual Execution:
Alternatively, you can run it manually via terminal:
```bash
python -m streamlit run app.py
```
