import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx

def plot_graph(graph_data, fraud_nodes):
    # PyG data to networkx for plotting
    G = to_networkx(graph_data, to_undirected=True)

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    colors = []

    for node in G.nodes():
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        if node in fraud_nodes:
            colors.append("red")
        else:
            colors.append("green")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(size=10, color=colors)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    return fig
