import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx

def plot_graph(graph_data, fraud_nodes):
    # PyG data to networkx for plotting
    G = to_networkx(graph_data, to_undirected=True)

    # Use a layout that spreads nodes naturally from the center
    # k regulates the distance between nodes; larger k means further apart
    pos = nx.spring_layout(G, k=0.15, iterations=60, seed=42)

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
        line=dict(width=0.3, color="#000000"),  # Darker, thinner edges like screenshot 2
        hoverinfo='none',
        mode='lines',
        opacity=0.6
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
            # Most nodes blue, matching the screenshot 2 look
            colors.append("blue")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=6, 
            color=colors,
            line=dict(width=0.5, color="white") # subtle border
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="white"
                    ))
    return fig
