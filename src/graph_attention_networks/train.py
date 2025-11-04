from pathlib import Path


def train(seed: int = 0, epochs: int = 200):
    """
    TODO:
      - Load Cora dataset (PyG Planetoid + NormalizeFeatures)
      - Define GAT model
      - Train on train_mask, validate on val_mask, report test_mask
      - Save metrics to results/ and weights to models/
    """
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    return {"val_acc": None, "test_acc": None}
