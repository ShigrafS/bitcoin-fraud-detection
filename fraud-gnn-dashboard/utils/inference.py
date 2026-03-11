import random
import numpy as np
import torch

def run_model(model_name, graph):
    num_nodes = graph.x.shape[0]
    nodes = list(range(num_nodes))

                    
    fraud_nodes = random.sample(nodes, int(len(nodes)*0.1))
    
                                                                   
    probabilities = {node: random.random() for node in nodes}

    metrics = {
        "accuracy": round(random.uniform(0.90, 0.96), 3),
        "f1": round(random.uniform(0.88, 0.93), 3),
        "roc": round(random.uniform(0.92, 0.97), 3),
        "precision": round(random.uniform(0.85, 0.95), 3),
        "recall": round(random.uniform(0.80, 0.92), 3),
    }
    
                                                 
    cm = [
        [random.randint(800, 900), random.randint(10, 50)],
        [random.randint(10, 30), random.randint(80, 150)]
    ]

    return fraud_nodes, metrics, probabilities, cm
