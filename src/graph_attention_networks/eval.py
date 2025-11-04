from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_attention_networks.model import GAT
from graph_attention_networks.train import accuracy


def evaluate(model_path: str = "models/gat_cora.pt"):
    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0]

    model = GAT(
        in_channels=dataset.num_features,
        hidden=8,
        out_channels=dataset.num_classes,
        heads=8,
        dropout=0.6,
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        acc_test = accuracy(logits, data.y, data.test_mask)
    return {"test_acc": acc_test}
