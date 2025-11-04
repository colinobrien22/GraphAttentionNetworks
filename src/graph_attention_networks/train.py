# src/graph_attention_networks/train.py
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_attention_networks.model import GAT


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits, y, mask):
    pred = logits.argmax(dim=-1)
    correct = int((pred[mask] == y[mask]).sum())
    total = int(mask.sum())
    return correct / total if total else 0.0


def train(
    seed: int = 0,
    epochs: int = 200,
    hidden: int = 8,
    heads: int = 8,
    dropout: float = 0.6,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
):
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    set_seed(seed)

    dataset = Planetoid(
        root="./data",
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0]

    model = GAT(
        in_channels=dataset.num_features,
        hidden=hidden,
        out_channels=dataset.num_classes,
        heads=heads,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.0
    best_snapshot = {"epoch": 0, "val": 0.0, "test": 0.0}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            acc_train = accuracy(logits, data.y, data.train_mask)
            acc_val = accuracy(logits, data.y, data.val_mask)
            acc_test = accuracy(logits, data.y, data.test_mask)

        if acc_val > best_val:
            best_val = acc_val
            best_snapshot = {"epoch": epoch, "val": acc_val, "test": acc_test}
            torch.save(model.state_dict(), "models/gat_cora.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:03d}] loss={loss.item():.4f} | "
                f"train={acc_train:.3f} val={acc_val:.3f} test={acc_test:.3f}"
            )

    with open("results/metrics.json", "w") as f:
        json.dump(best_snapshot, f, indent=2)

    return best_snapshot
