from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_attention_networks.model import GAT
from graph_attention_networks.train import accuracy


def _save_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    Path("results").mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


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
        test_mask = data.test_mask
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = logits.argmax(dim=-1)[test_mask].cpu().numpy()
        acc_test = float(accuracy(logits, data.y, test_mask))

    # save confusion matrix image
    _save_confusion_matrix(y_true, y_pred, "results/confusion_matrix.png")
    return {"test_acc": acc_test}
