import argparse
import json
from pathlib import Path

import yaml

from graph_attention_networks.eval import evaluate
from graph_attention_networks.train import train


def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="ganet", description="Graph Attention Networks on Cora"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a GAT model on Cora")
    p_train.add_argument("--config", type=str, default="configs/default.yaml")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--hidden", type=int)
    p_train.add_argument("--heads", type=int)
    p_train.add_argument("--dropout", type=float)
    p_train.add_argument("--lr", type=float)
    p_train.add_argument("--weight_decay", type=float)
    p_train.add_argument("--seed", type=int)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    p_eval.add_argument("--model", type=str, default="models/gat_cora.pt")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = _load_cfg(args.config)
        # apply CLI overrides
        for k in ["epochs", "hidden", "heads", "dropout", "lr", "weight_decay", "seed"]:
            v = getattr(args, k)
            if v is not None:
                cfg[k] = v

        Path("results").mkdir(exist_ok=True)
        print("Config:", json.dumps(cfg, indent=2))
        best = train(**cfg)
        print("Best snapshot:", json.dumps(best, indent=2))

    elif args.cmd == "eval":
        res = evaluate(model_path=args.model)
        print("Evaluation:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
