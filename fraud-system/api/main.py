import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from pipelines.inference_pipeline import InferencePipeline

# FastAPI App initialization
app = FastAPI(
    title="GraphGuard - Fraud Detection API",
    description="Production-grade API for transaction fraud scoring, Louvain clustering, and pattern analysis.",
    version="1.0.0"
)

# Global Inference Pipeline instance
pipeline = None

class PredictRequest(BaseModel):
    node_id: Optional[int] = Field(None, description="The specific transaction node ID in the graph.", example=230420)
    features: Optional[Dict[str, float]] = Field(None, description="Key-value mapping of all 165 transaction features.")
    model_name: Optional[str] = Field("voting", description="ML model for custom prediction ('rf', 'xgb', 'voting', 'stacking').")

class NeighborInfo(BaseModel):
    id: int
    score: float
    label: str
    relationship: str

class PredictResponse(BaseModel):
    node_id: Optional[int] = None
    fraud_score: float = Field(..., description="Calculated probability of transaction fraud.")
    label: str = Field(..., description="'Fraudulent' or 'Legitimate'")
    cluster_id: Optional[int] = None
    fraud_type: Optional[str] = None
    explanation: Optional[str] = None
    time_step: Optional[int] = None
    neighbors: Optional[List[NeighborInfo]] = None

class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    mean_score: float
    max_score: float
    risk_level: str
    pattern_type: str

class ClusterResponse(BaseModel):
    total_clusters: int
    high_risk_clusters_count: int
    clusters: List[ClusterSummary]

@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing Inference Pipeline...")
    pipeline = InferencePipeline()
    print("Pipeline initialization complete.")

@app.get("/health")
def health():
    """Returns application health status."""
    global pipeline
    if pipeline is None or pipeline.G is None:
        return {"status": "unhealthy", "message": "Graph not initialized"}
    return {
        "status": "healthy",
        "loaded_models": list(pipeline.models.keys()),
        "loaded_gnns": list(pipeline.gnn_sessions.keys()),
        "graph_nodes": pipeline.G.number_of_nodes(),
        "graph_edges": pipeline.G.number_of_edges()
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predicts fraud score and label.
    Supports either node_id (lookup in graph + GNN prediction) or custom input features (tabular ML ensemble).
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    if request.node_id is not None:
        # Single Node Inference Mode
        try:
            res = pipeline.predict_single_node(request.node_id)
            return PredictResponse(
                node_id=res["node_id"],
                fraud_score=res["fraud_score"],
                label=res["label"],
                cluster_id=res["cluster_id"],
                fraud_type=res["fraud_type"],
                explanation=res["explanation"],
                time_step=res["time_step"],
                neighbors=res["neighbors"]
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")
            
    elif request.features is not None:
        # Custom Input Mode
        try:
            res = pipeline.predict_custom_input(request.features, model_name=request.model_name)
            return PredictResponse(
                fraud_score=res["fraud_score"],
                label=res["label"]
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'node_id' or 'features' in request payload.")

@app.post("/cluster", response_model=ClusterResponse)
@app.post("/cluster-analysis", response_model=ClusterResponse)
def cluster_analysis():
    """
    Triggers/Retrieves Louvain community detection.
    Returns a summary of all detected clusters (risk levels, size, average score, structural pattern).
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    if not pipeline.cluster_stats:
        raise HTTPException(status_code=500, detail="Louvain partitioning has not been initialized.")
        
    from models.fraud_type import classify_batch_communities
    from collections import defaultdict
    
    clusters_dict = defaultdict(list)
    for node, cid in pipeline.partition.items():
        clusters_dict[cid].append(node)
        
    # Classify all patterns dynamically
    cluster_patterns, metrics_summary = classify_batch_communities(pipeline.G, clusters_dict)
    
    clusters_list = []
    high_risk_count = 0
    
    for cid, stats in pipeline.cluster_stats.items():
        mean_score = stats["mean"]
        
        # Risk level determination
        if mean_score > 0.6:
            risk_level = "Critical"
            high_risk_count += 1
        elif mean_score > 0.3:
            risk_level = "Elevated"
        else:
            risk_level = "Legitimate"
            
        pattern = cluster_patterns.get(cid, "chain/mixed")
        
        clusters_list.append(ClusterSummary(
            cluster_id=cid,
            size=stats["size"],
            mean_score=mean_score,
            max_score=stats["max"],
            risk_level=risk_level,
            pattern_type=pattern
        ))
        
    # Sort by risk (mean score descending)
    clusters_list = sorted(clusters_list, key=lambda c: c.mean_score, reverse=True)
    
    return ClusterResponse(
        total_clusters=len(pipeline.cluster_stats),
        high_risk_clusters_count=high_risk_count,
        clusters=clusters_list
    )

@app.get("/model-info")
def model_info():
    """Returns metadata about the loaded ML and GNN models."""
    global pipeline
    if pipeline is None:
        return {"status": "uninitialized"}
    return {
        "tabular_models": list(pipeline.models.keys()),
        "gnn_models": list(pipeline.gnn_sessions.keys()),
        "p99_latency_sla": "100ms",
        "target_accuracy": "97.4%",
        "target_f1": "93.1%"
    }

@app.get("/cluster-graph/{cluster_id}")
def get_cluster_graph(cluster_id: int):
    """
    Returns nodes and edges for a specific cluster to render in the visualization.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    from collections import defaultdict
    clusters = defaultdict(list)
    for node, cid in pipeline.partition.items():
        clusters[cid].append(node)
        
    if cluster_id not in clusters:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
        
    nodes_in_cluster = clusters[cluster_id]
    subG = pipeline.G.subgraph(nodes_in_cluster)
    
    # Format nodes
    nodes_list = []
    for n in subG.nodes():
        score = pipeline.node_probs.get(n, 0.0)
        if score > 0.6:
            label = "Fraudulent"
            color = "Red"
        elif score > 0.3:
            label = "Suspicious"
            color = "Orange"
        else:
            label = "Legitimate"
            color = "Green"
            
        nodes_list.append({
            "id": int(n),
            "score": float(score),
            "label": label,
            "color": color
        })
        
    # Format edges
    edges_list = []
    for u, v in subG.edges():
        edges_list.append({
            "source": int(u),
            "target": int(v)
        })
        
    return {
        "cluster_id": cluster_id,
        "nodes": nodes_list,
        "edges": edges_list
    }


@app.get("/subgraph/{time_step}")
def get_subgraph(time_step: int, max_nodes: Optional[int] = 1000):
    """
    Returns nodes and edges for a specific time step to render in the visualization.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    if pipeline.features_df is None or pipeline.G is None:
        raise HTTPException(status_code=500, detail="Graph or features not initialized in pipeline")

    # Get nodes belonging to the specified time step
    # features_df has columns: 0 (id), 1 (time_step)
    feat_step = pipeline.features_df[pipeline.features_df.iloc[:, 1] == time_step]
    if feat_step.empty:
        raise HTTPException(status_code=404, detail=f"Time step {time_step} not found.")
        
    step_nodes = list(feat_step.iloc[:, 0].values)
    
    # Limit nodes if there are too many to keep Plotly fast and responsive
    if max_nodes and len(step_nodes) > max_nodes:
        import random
        random.seed(42) # Reproducible subset
        step_nodes = random.sample(step_nodes, max_nodes)
        
    step_nodes_set = set(step_nodes)
    subG = pipeline.G.subgraph(step_nodes_set)
    
    # Build dictionary for fast O(1) lookup
    class_map = dict(zip(pipeline.classes_df['txId'], pipeline.classes_df['class']))
    
    # Format nodes
    nodes_list = []
    for n in subG.nodes():
        score = pipeline.node_probs.get(n, 0.0)
        # Class mapping from classes_df: '1' is fraud, '2' is normal, 'unknown'
        orig_class = class_map.get(n, 'unknown')
        
        # Human-readable labels
        label_str = {
            '1': 'Fraudulent',
            '2': 'Legitimate',
            'unknown': 'Unknown'
        }.get(str(orig_class), 'Unknown')
        
        if score > 0.6:
            color = "Red"
        elif score > 0.3:
            color = "Orange"
        else:
            color = "Green"
            
        nodes_list.append({
            "id": int(n),
            "score": float(score),
            "label": label_str,
            "color": color
        })
        
    # Format edges
    edges_list = []
    for u, v in subG.edges():
        edges_list.append({
            "source": int(u),
            "target": int(v)
        })
        
    return {
        "time_step": time_step,
        "nodes": nodes_list,
        "edges": edges_list
    }


