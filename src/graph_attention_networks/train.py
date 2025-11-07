import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_attention_networks.model import GAT


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == y[mask]).sum().item()
    total = int(mask.sum())
    return correct / max(total, 1)


def _append_epoch_metrics(
    epoch: int, acc_train: float, acc_val: float, acc_test: float
):
    Path("results").mkdir(exist_ok=True)
    csv_path = Path("results/metrics_epoch.csv")
    if not csv_path.exists():
        csv_path.write_text("epoch,acc_train,acc_val,acc_test\n")
    with csv_path.open("a") as f:
        f.write(f"{epoch},{acc_train:.4f},{acc_val:.4f},{acc_test:.4f}\n")


def _plot_learning_curve(
    csv_path: str = "results/metrics_epoch.csv", out_path: str = "results/curve.png"
):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["acc_train"], label="train")
    plt.plot(df["epoch"], df["acc_val"], label="val")
    plt.plot(df["epoch"], df["acc_test"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(
    hidden: int = 8,
    heads: int = 8,
    dropout: float = 0.6,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    epochs: int = 200,
    seed: int = 0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    model = GAT(
        in_channels=dataset.num_features,
        hidden=hidden,
        out_channels=dataset.num_classes,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_snapshot = {"epoch": -1, "val": 0.0, "test": 0.0}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            acc_tr = accuracy(logits, data.y, data.train_mask)
            acc_va = accuracy(logits, data.y, data.val_mask)
            acc_te = accuracy(logits, data.y, data.test_mask)

        _append_epoch_metrics(epoch, acc_tr, acc_va, acc_te)

        # Save best-by-val
        if acc_va > best_val:
            best_val = acc_va
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/gat_cora.pt")
            best_snapshot = {"epoch": epoch, "val": acc_va, "test": acc_te}
            Path("results").mkdir(exist_ok=True)
            with open("results/metrics.json", "w") as f:
                json.dump(best_snapshot, f)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[{epoch:03d}] loss={loss.item():.4f} acc(train/val/test)={acc_tr:.3f}/{acc_va:.3f}/{acc_te:.3f}"
            )

    # Plot at the end
    _plot_learning_curve()

    return best_snapshot
