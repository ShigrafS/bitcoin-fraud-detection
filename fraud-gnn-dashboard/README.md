# 💳 Fraud Detection with Graph Neural Networks Dashboard

This dashboard visualizes and analyzes Bitcoin transaction fraud using Graph Neural Networks on the Elliptic dataset.

## Setup Instructions

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add the Dataset**
   Download the Elliptic dataset and place the following files inside the `data/` directory:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`

3. **Generate Sample Graph**
   The full dataset is too large to render interactively, so generate a sample graph first:
   ```bash
   python scripts/generate_sample_graph.py
   ```

## Running the App

To run the Streamlit dashboard on Windows, simply double-click or run:
```cmd
run.bat
```

> **Note**: If you see a `Fatal error in launcher: Unable to create process` when running `streamlit run app.py` on Windows, you should always use `run.bat` (which runs `python -m streamlit run app.py`).
