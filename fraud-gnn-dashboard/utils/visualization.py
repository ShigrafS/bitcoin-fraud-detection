import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx

def plot_graph(graph_data, fraud_nodes, probabilities=None):
    # PyG data to networkx for plotting
    G = to_networkx(graph_data, to_undirected=True)
    degrees = dict(G.degree())

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
    sizes = []
    hover_text = []

    node_label_map = {1: "Fraud", 0: "Normal", -1: "Unknown"}
    
    for idx, node in enumerate(G.nodes()):
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        
        # Color based on probability gradient if available
        if probabilities and node in probabilities:
            colors.append(probabilities[node])
        else:
            colors.append(1.0 if node in fraud_nodes else 0.0)
            
        # Hover info
        deg = degrees.get(node, 0)
        prob = probabilities.get(node, 0) if probabilities else (1.0 if node in fraud_nodes else 0.0)
        
        # Map label
        label_id = int(graph_data.y[idx].item())
        label_str = node_label_map.get(label_id, "Unknown")
        
        # Time step (default to 1 if not present)
        ts = int(graph_data.time_step[idx].item()) if hasattr(graph_data, 'time_step') else 1
        
        hover_text.append(
            f"<b>Node: {node}</b><br>"
            f"Label: {label_str}<br>"
            f"Time Step: {ts}<br>"
            f"Degree: {deg}<br>"
            f"Fraud Prob: {prob:.4f}"
        )
        
        # Node size based on degree
        size = min(30, max(6, deg * 2))
        sizes.append(size)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=hover_text,
        marker=dict(
            size=sizes, 
            color=colors,
            colorscale='RdYlGn_r', # Red is high prob, Green is low
            showscale=True,
            colorbar=dict(title="Fraud Prob", thickness=15),
            cmin=0.0,
            cmax=1.0,
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
