# 🧠 Graph Attention Networks (GAT) on Cora
Node classification on the [Cora citation network](https://relational.fel.cvut.cz/dataset/CORA) using a Graph Attention Network implemented with PyTorch Geometric.

![CI](https://github.com/colinobrien22/GraphAttentionNetworks/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Quick Start

```bash
# clone + setup
git clone git@github.com:colinobrien22/GraphAttentionNetworks.git
cd GraphAttentionNetworks
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

### Usage

Train (YAML config):
```bash
python3 -m graph_attention_networks.cli train --config configs/default.yaml

Train (override from CLI)
python3 -m graph_attention_networks.cli train --epochs 100 --hidden 16 --heads 4

Evaluate
python3 -m graph_attention_networks.cli eval --model models/gat_cora.pt




<img src="results/curve.png" width="420"/> <img src="results/confusion_matrix.png" width="420"/>
