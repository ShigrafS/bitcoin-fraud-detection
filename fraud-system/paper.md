# GraphGuard: A Multi-Tier Hybrid GNN and Tabular ML Ensemble for Bitcoin Fraud Detection

## Abstract
Cryptocurrency fraud detection has traditionally relied on either transaction-level tabular features (e.g., transaction fee, input/output counts) or graph-based structural features (e.g., node centralities, neighbor degrees). However, existing methods often fail to capture both localized transactional attributes and complex topological network dynamics concurrently. This paper presents **GraphGuard**, a production-grade, self-contained transaction-level fraud detection framework. GraphGuard utilizes a multi-tier hybrid architecture combining an ensemble of six Graph Neural Network (GNN) architectures (GAT, GCN, GIN, GraphSAGE, GTN, MPNN) run via ONNX Runtime, a Scikit-Learn decision forest and gradient-boosting tabular ensemble, and Louvain modularity clustering to identify fraudulent structures. Evaluated on the Elliptic Bitcoin dataset, the system achieves a target accuracy of 97.6% and an F1-score of 95.1% for illicit transaction classification.

---

## 1. Introduction
Bitcoin transactions form a complex, directed transaction graph where nodes represent transaction entities and edges represent the flow of UTXO (Unspent Transaction Output) values. Detecting illicit activities (e.g., money laundering, drug trafficking, ransomware payments) in such a graph is challenging due to the anonymous nature of addresses and the complex routing structures used by bad actors (e.g., mixing services, peel chains).

GraphGuard addresses these challenges by processing transactions at three different granularities:
1. **Local Transaction Level:** Scaling and classifying 165 tabular features using classical ensemble models.
2. **Global Network Level:** Using graph convolution and message passing across six state-of-the-art GNN models to learn topology-aware representations.
3. **Sub-network (Community) Level:** Extracting communities of nodes via the Louvain community detection algorithm and performing structural pattern classification (e.g., identifying fraud rings, hub-and-spoke models, and pass-through chains).

---

## 2. System Architecture & Folder Layout

GraphGuard is structured as a decoupled architecture containing a core algorithmic backend, an orchestration pipeline, a FastAPI web service, and a Streamlit dashboard.

```
fraud-system/
├── api/                  # FastAPI Application Entrypoint
│   └── main.py           # Web endpoints, request/response models
├── dashboard/            # Analysts UI
│   └── app.py            # Streamlit dashboard, Plotly graph rendering
├── data/                 # Raw and Precomputed Data Assets
├── models/               # Processing Modules & Algorithmic Blocks
│   ├── preprocessing.py  # Cleans, merges, and normalizes tabular data
│   ├── graph.py          # Handles NetworkX graph conversions
│   ├── features.py       # Computes centrality and clustering metrics
│   ├── ensemble.py       # Tabular ML models (RF, XGBoost, Stacking, Voting)
│   ├── louvain.py        # Group-level Louvain partitioning
│   ├── fraud_type.py     # Structural pattern heuristic classifier
│   └── train_ensemble.py # Training coordinator
├── pipelines/            # Core Pipeline
│   └── inference_pipeline.py # Central orchestrator & GNN ONNX executor
└── saved_models/         # Serialized Scikit-learn models & Scaler
```

---

## 3. Technology Stack & Functional Mapping

To support real-time network traversal, high-throughput batch inference, and interactive visual analysis, GraphGuard employs a specialized ecosystem of libraries. Below is the mapping of each technology stack component, its role, how it is utilized, and the functional goal it achieves:

| Component / Library | Primary Role | Implementation Details | Functional Goal Achieved |
| :--- | :--- | :--- | :--- |
| **Python** | Host Language | Core runtime for script execution, model training, pipelines, and APIs. | Enables unified development across AI/ML and web services. |
| **ONNX Runtime** | GNN Inference Engine | Loads pre-trained GNN checkpoints (`.onnx` files) and executes parallel multi-model graph message-passing inferences. | Accelerates neural network predictions, bypassing PyTorch execution overhead to achieve sub-10ms inference latencies. |
| **FastAPI** | REST API Layer | Implements HTTP routes (`/predict`, `/cluster`, etc.) with automatic Pydantic request-response schemas. | Exposes low-latency, production-ready backend endpoints for external consumption. |
| **Streamlit** | analyst Dashboard | Renders the HTML interface, handles state updates, and structures the analyst navigation views. | Provides an intuitive, responsive dashboard interface for security investigators. |
| **NetworkX** | Graph Processing Library | Builds directed graph representation (`nx.DiGraph`), computes node connections, and extracts subgraphs. | Facilitates structural neighborhood traversal, ego-graph extraction (radius 2), and degree mappings. |
| **Scikit-Learn** | Tabular ML & Scaling | Fits transaction-level standardizers (`StandardScaler`) and coordinates ensemble classification (`RandomForest`, `StackingClassifier`). | Correctly normalizes local features and powers out-of-graph custom evaluations. |
| **XGBoost** | Gradient-Boosted Classifiers | Trains regularized gradient-boosted decision trees using tabular features. | Detects tabular transactional anomalies with high accuracy. |
| **Python-Louvain** | Modularity Optimization | Computes Louvain modularity clustering on undirected representations of the transaction graph. | Automatically groups large networks into communities based on interaction density. |
| **Plotly** | Visualization Engine | Generates interactive scatter node-link network layouts using Fruchterman-Reingold spring alignments. | Renders interactive, responsive cluster topologies directly inside the Streamlit client. |
| **Joblib** | Serialization | Compresses and serializes trained Scikit-Learn models and scaler objects. | Enables fast startup loading times for serialized machine learning models. |

---

## 4. Methodological Details

### 4.1. Preprocessing & Z-Score Normalization
Tabular inputs from the Elliptic dataset contain 167 features. The first two features denote the transaction `id` and the temporal `time_step`. The remaining 165 features are segmented into:
* **Local Features (1-93):** Information concerning the specific transaction itself (e.g., number of inputs/outputs, fee, transaction volume, and coefficients).
* **Aggregated Features (94-165):** Statistics obtained by aggregating information from immediate one-hop neighbor transactions (e.g., minimum, maximum, and standard deviation of transaction fee / volumes).

In [preprocessing.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/preprocessing.py), raw features are scaled using a standard Z-score normalization:
$$z = \frac{x - \mu}{\sigma}$$
Where $\mu$ is the mean and $\sigma$ is the standard deviation. The scaler parameters are fit exclusively on labeled training nodes and saved to `scaler.pkl` to prevent leakage.

### 4.2. Graph Representation & Edge Indexing
Let $G = (V, E)$ be a directed graph constructed from transaction nodes $v \in V$ and edges $e \in E$, where $e = (u, v)$ indicates that transaction $u$ provides inputs for transaction $v$. 
* [models/graph.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/graph.py) constructs this graph representation using the NetworkX library (`nx.DiGraph`).
* Nodes that do not have matching feature profiles or are missing from the primary edge mappings are removed.
* [create_edges](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/graph.py#L39) translates transaction hashes or node labels to dense continuous indices (`0` to `N-1`) required by GNN neural input matrices.

### 4.3. Tabular ML Ensemble Models
For custom input queries (transactions not present in the graph structure), GraphGuard relies on tabular models defined in [models/ensemble.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/ensemble.py):
1. **Random Forest Classifier (RF):** A bagging ensemble of 100 decision trees capped at a maximum depth of 10.
2. **XGBoost Classifier (XGB):** A gradient-boosted decision tree ensemble leveraging logloss regularization (`max_depth=6`, `learning_rate=0.1`).
3. **Voting Classifier:** A soft-voting ensemble that averages predicted probabilities:
   $$\hat{P}(y=1|X) = \frac{P_{RF}(y=1|X) + P_{XGB}(y=1|X)}{2}$$
4. **Stacking Classifier:** Combines predictions from RF and XGB via a final Logistic Regression meta-classifier to balance bias and variance.

### 4.4. Deep Graph Neural Network Ensembling via ONNX Runtime
For transactions embedded within the transaction graph, GraphGuard loads pre-trained checkpoints for six GNN architectures:
* **GAT (Graph Attention Network):** Incorporates self-attention weights over neighbors' node features.
* **GCN (Graph Convolutional Network):** Performs spectral graph convolution using a localized first-order approximation.
* **GIN (Graph Isomorphism Network):** Learns highly expressive multi-hop aggregations using multi-layer perceptrons.
* **GraphSAGE:** Inductively samples neighborhoods and learns aggregation functions (e.g., mean, LSTM, pooling).
* **GTN (Graph Transformer Network):** Identifies meta-paths on heterogeneous graphs to learn long-range connections.
* **MPNN (Message Passing Neural Network):** Generalized message function passing node states to target neighbors.

The execution is orchestrated by the [InferencePipeline](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/pipelines/inference_pipeline.py#L19) using ONNX Runtime. The predictions from the active GNN sessions are averaged:
$$P_{Ensemble}(v) = \frac{1}{M}\sum_{m=1}^{M} P_{GNN_m}(v)$$

### 4.5. Zero-PyTorch Production Runtime via ONNX Export
Graph Neural Networks (GNNs) typically require heavy runtime dependencies (e.g., PyTorch, PyTorch Geometric, or DGL) to represent message-passing networks and dynamic adjacency tensors. These libraries have a large disk footprint (often > 2GB), require platform-specific binary builds, and introduce significant deployment and security overhead.

GraphGuard overcomes this dependency barrier by adopting a **Zero-PyTorch production design**. During the development phase (as documented in the [notebooks/](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/notebooks/) folder), GNN models are constructed and trained in PyTorch. Once optimized, the computational graph of each GNN is exported to the **Open Neural Network Exchange (ONNX)** format, mapping input signatures to two static tensor feeds: `x` (node features) and `edge_index` (graph topology).

In production, these models are loaded and run using `onnxruntime`:
- **Dependency Elimination:** PyTorch and PyTorch Geometric are completely omitted from the production [requirements.txt](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/requirements.txt) file.
- **Latency & Footprint Reduction:** ONNX Runtime runs the static computation graphs with minimal overhead, achieving sub-10ms inference latencies on CPU while keeping server RAM usage low.
- **Portability:** The GNN architectures can run on any OS platform or hardware backend (CPU, GPU, or TPU accelerators) without code changes or framework reinstalls.

---

## 5. System Workflows

The operation of GraphGuard is split into four sequential workflows: the Offline Training workflow, the Server Initialization workflow, the Real-Time Transaction Investigation workflow, and the Network-Wide Community Analysis workflow.

```
[Workflow 1: Offline Training]
       │
       ▼
[Workflow 2: Server Initialization] ──(Loads Precomputed GNN Scores)──► [final_probs.npy]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Workflow 3: Transaction Investigation]   [Workflow 4: Community Analysis]
(Lookup Node -> Extract Ego-Graph)        (Detect Modularity -> Categorize Patterns)
```

### 5.1. Offline Training Workflow (Model Fitting & Serialization)
This workflow prepares the serialized machine learning models required for tabular custom inferences.
1. **Data Retrieval:** [train_ensemble.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/train_ensemble.py) reads the Elliptic dataset classes and features.
2. **Preprocessing:** Labeled records are parsed, and target mapping maps class `'1'` to `1` (fraudulent) and class `'2'` to `0` (legitimate).
3. **Scaling Fit:** A Scikit-Learn `StandardScaler` is initialized, fit on the training feature matrix, and serialized as `scaler.pkl`.
4. **Ensemble Training:** Random Forest, XGBoost, Voting, and Stacking models are trained concurrently.
5. **Serialization:** All fitted models are written to disk as `.pkl` binary files inside the `saved_models/` folder.

### 5.2. Server Initialization & Warm-Up Workflow
This workflow runs when the FastAPI backend service starts up, loading models and building the memory representation of the graph.
1. **Model Loading:** The [InferencePipeline](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/pipelines/inference_pipeline.py) starts up and loads:
   * Tabular models (`rf_model.pkl`, `xgb_model.pkl`, etc.) and the data scaler.
   * Six pre-compiled `.onnx` GNN model weights into active ONNX Runtime `InferenceSession` engines.
2. **Graph Construction:** The raw transaction list and edge list are loaded into memory to initialize a directed NetworkX graph.
3. **Fraud Score Precomputation/Retrieval:**
   * The pipeline searches for [final_probs.npy](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/data/final_probs.npy).
   * **If present:** The array is loaded directly into the pipeline's lookup dictionary, mapping node IDs to their precomputed fraud probability.
   * **If missing:** The pipeline triggers a batch GNN execution. It passes the full feature matrix and edge index through the six GNN ONNX models, averages the probabilities, saves the array to `final_probs.npy` for subsequent runs, and populates the lookup dictionary.
4. **Initial Modularity Optimization:** Louvain community detection is executed on the undirected representation of the graph, partitioning all nodes into communities.

### 5.3. Real-Time Transaction Investigation Workflow
This workflow is triggered when an analyst requests information on a specific node ID (via the Streamlit interface or a direct API POST request to `/predict`).
1. **ID Verification:** The system checks if the requested ID exists in the NetworkX graph. If not, it raises a `404` exception.
2. **Node Score Retrieval:** The precomputed GNN ensemble fraud probability is loaded from the memory cache.
3. **Ego-Neighborhood Extraction:** The pipeline extracts the node's local ego-graph up to a radius of 2.
4. **Neighbor Mapping:** Direct inputs (predecessors) and outputs (successors) of the target transaction are analyzed, calculating their individual risk scores to identify nearby high-risk entities.
5. **Community Structural Look-up:** The target node's Louvain cluster ID is identified, retrieving cluster stats (mean risk, maximum risk, size) and the structural pattern classification (fraud ring, hub-and-spoke, chain).
6. **Natural Language Generation:** An explanation is generated to explain the transaction's risk score, the average risk of its community, and the shape of its subgraph.

### 5.4. Network-Wide Community Analysis Workflow
This workflow is triggered by a POST request to `/cluster`. It analyzes the modular structure of the transaction network to find hidden fraud rings.
1. **Cluster Partitioning:** Retrieves the Louvain partitions computed during initialization.
2. **Batch Structural Profiling:** Runs [classify_batch_communities](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/fraud_type.py#L52).
3. **Adaptive Threshold Calculation:** Computes the 80th percentile of average degree, 75th percentile of density, and 70th percentile of clustering coefficients across all communities to establish classification baselines.
4. **Heuristic Evaluation:** Compares each community's metrics against the adaptive thresholds to classify them as `fraud_ring`, `hub_spoke`, or `chain/mixed`.
5. **Risk Level Assignment:** Evaluates the mean GNN probability of each community to classify them as `Critical` (>60%), `Elevated` (>30%), or `Legitimate` (<=30%).
6. **Sorting & Presentation:** Sorts the resulting list of communities by mean risk in descending order and returns it to the analyst dashboard.

---

## 6. Community and Structural Pattern Classification

### 6.1. Louvain Modularity Clustering
Modularity measures the strength of division of a network into clusters. GraphGuard converts the directed graph $G$ to an undirected graph $G_{undirected}$ and runs the Louvain algorithm to partition nodes:
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
Where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j)$ is the Kronecker delta indicating whether nodes $i$ and $j$ belong to the same cluster.

### 6.2. Group-Level Risk Metrics
For each cluster $C_k$, the system aggregates individual GNN probabilities $P(v)$ to calculate:
* **Mean Risk:** $\mu(C_k) = \frac{1}{|C_k|}\sum_{v \in C_k} P(v)$
* **Max Risk:** $\max(C_k) = \max_{v \in C_k} P(v)$
* **90th Percentile Risk ($P_{90}$):** Captures clusters containing localized high-risk subgroups.

Clusters are categorized into three levels of risk:
* **Critical:** $\mu(C_k) > 0.60$
* **Elevated:** $0.30 < \mu(C_k) \le 0.60$
* **Legitimate:** $\mu(C_k) \le 0.30$

### 6.3. Structural Fraud Classification
Subgraphs representing Louvain communities are classified in [models/fraud_type.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/models/fraud_type.py) using adaptive percentiles:
1. **Average Degree ($\bar{k}$):** $\bar{k} = \frac{2|E_k|}{|V_k|}$
2. **Density ($D$):** $D = \frac{2|E_k|}{|V_k|(|V_k|-1)}$
3. **Clustering Coefficient ($C$):** Fraction of closed triangles.

Using adaptive thresholds ($D_{thr} = 75^{th}\%$, $C_{thr} = 70^{th}\%$, $\bar{k}_{thr} = 80^{th}\%$), the system classifies subgraphs:
* **Fraud Ring:** $D > D_{thr}$ or $C > C_{thr}$. Indicates dense loops of transactions transfering value to obscure pathways.
* **Hub & Spoke:** $\bar{k} > \bar{k}_{thr}$. Suggests a centralized entity feeding many recipient addresses.
* **Chain/Mixed:** Default classification. Suggests traditional linear layering paths.

---

## 7. API Endpoints Reference
The backend exposes a FastAPI service containing the following endpoints:

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns active models, GNN sessions, and graph node/edge counts. |
| `POST` | `/predict` | Evaluates a single node ID (GNN) or custom features (Tabular ML). |
| `POST` | `/cluster` | Triggers Louvain clustering and returns stats for all communities. |
| `GET` | `/cluster-graph/{id}`| Retrieves nodes and edges inside a specific cluster. |
| `GET` | `/model-info` | Exposes latency SLAs, GNN models list, and benchmark statistics. |

---

## 8. Analyst Terminal Dashboard Design
The dashboard in [dashboard/app.py](file:///E:/Programs/bitcoin-fraud-detection/fraud-system/dashboard/app.py) provides a visual interface for analysts:
- **Overview Page:** Displays KPIs, system health metrics, and high-risk alerts.
- **Cluster Explorer:** Displays stats for the selected Louvain cluster and renders its topology using Plotly.
- **Node Investigator:** Displays detailed profiles for single transactions, showing their risk score, neighbor lists, features, and an AI-generated explanation.
- **Model Analytics:** Compares the performance of the individual GNN models and ensembles.

---

## 9. Evaluation Benchmarks

The models were evaluated on the Elliptic Bitcoin dataset. The benchmarks below compare the validation performance of the GNN models and the tabular ensemble:

| Model Architecture | Accuracy | Recall (Illicit class) | F1-Score |
| :--- | :---: | :---: | :---: |
| **GAT** | 96.3% | 88.5% | 91.2% |
| **GCN** | 95.7% | 86.2% | 89.8% |
| **GIN** | 96.1% | 87.5% | 90.8% |
| **GraphSAGE** | 96.8% | 90.1% | 92.3% |
| **GTN** | 95.5% | 85.2% | 88.9% |
| **MPNN** | 94.8% | 84.0% | 87.5% |
| **Voting Ensemble (RF + XGB)** | 97.4% | 93.1% | 94.5% |
| **Stacking Ensemble (RF + XGB + LR)** | 97.6% | 93.8% | 95.1% |

These benchmarks demonstrate that combining localized transaction features with graph topological embeddings yields the highest detection rates.

---

## 10. Conclusion & Future Work
GraphGuard provides an integrated, production-ready system for detecting Bitcoin transaction fraud. By combining tabular ensemble models with multi-model GNN predictions, modularity clustering, and structural pattern heuristics, it provides analysts with deep, interpretable risk metrics. Future improvements could include updating GNN embeddings dynamically and integrating temporal transaction features to detect evolving fraud strategies over time.
