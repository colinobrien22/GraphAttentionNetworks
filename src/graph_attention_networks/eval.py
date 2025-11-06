from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_attention_networks.model import GAT
from graph_attention_networks.train import accuracy


def _save_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    Path("results").mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # normalized percentages
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Normalized)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # Annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]*100:.1f}%",
                ha="center",
                va="center",
                color="black" if cm[i, j] < 0.5 else "white",
                fontsize=8,
            )
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
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_mask = data.test_mask
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = logits.argmax(dim=-1)[test_mask].cpu().numpy()
        acc_test = float(accuracy(logits, data.y, test_mask))

    # save classification report
    Path("results").mkdir(exist_ok=True)
    with open("results/report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, digits=3))

    # save confusion matrix
    _save_confusion_matrix(y_true, y_pred, "results/confusion_matrix.png")

    return {"test_acc": acc_test}
