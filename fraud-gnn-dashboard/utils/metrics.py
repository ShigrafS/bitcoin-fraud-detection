import pandas as pd
import plotly.express as px

def model_comparison():
    data = {
        "Model": ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN", "GTN"],
        "Accuracy": [0.92, 0.94, 0.93, 0.91, 0.92, 0.95],
        "F1": [0.89, 0.91, 0.90, 0.88, 0.89, 0.92],
        "ROC": [0.94, 0.96, 0.95, 0.93, 0.94, 0.97]
    }

    df = pd.DataFrame(data)

    fig = px.bar(df, x="Model", y="Accuracy")

    return fig
