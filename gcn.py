"""Jittor warmup track 1: Cora node classification."""

import argparse
import json
import os
import os.path as osp
import pickle
from math import log

if os.environ.get("USE_CUDA") == "0":
    os.environ.setdefault("cache_path", "/tmp/jittor_cpu_cache_jdjt")

import jittor as jt
from jittor import nn
import numpy as np


DEFAULT_CONFIG = {
    "model": "gcnii",
    "data_path": "data/cora.pkl",
    "output": "result.json",
    "seed": 42,
    "use_cuda": 1,
    "hidden_dim": 64,
    "num_layers": 64,
    "alpha": 0.1,
    "theta": 0.5,
    "dropout": 0.6,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 2000,
    "patience": 200,
    "log_interval": 100,
}


class GraphData:
    """Container for graph tensors and sparse adjacency formats."""


class GCNIINet(nn.Module):
    """GCNII model used for the submitted prediction file."""

    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers,
        alpha,
        theta,
        dropout,
    ):
        super().__init__()
        from jittor_geometric.nn.conv.gcn2_conv import GCN2Conv

        self.dropout = dropout
        self.alpha = alpha
        self.theta = theta
        self.lin1 = nn.Linear(num_features, hidden_dim)
        self.convs = nn.ModuleList(
            [GCN2Conv(hidden_dim, hidden_dim, spmm=True) for _ in range(num_layers)]
        )
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def execute(self, data):
        x = nn.dropout(data.x, self.dropout, is_train=self.training)
        x = x_0 = nn.relu(self.lin1(x))

        for layer_idx, conv in enumerate(self.convs):
            x = nn.dropout(x, self.dropout, is_train=self.training)
            beta = log(self.theta / (layer_idx + 1) + 1)
            x = nn.relu(conv(x, x_0, data.csc, data.csr, self.alpha, beta))

        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.lin2(x)
        return nn.log_softmax(x, dim=1)


class APPNPCpuNet(nn.Module):
    """Small dense APPNP-style fallback for CPU smoke tests."""

    def __init__(self, num_features, num_classes, hidden_dim, dropout, alpha):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = nn.Linear(num_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def execute(self, data, prop_steps):
        h = nn.dropout(data.x, self.dropout, is_train=self.training)
        h = nn.relu(self.lin1(h))
        h = nn.dropout(h, self.dropout, is_train=self.training)
        h = self.lin2(h)

        out = h
        for _ in range(prop_steps):
            out = (1.0 - self.alpha) * (data.norm_adj @ out) + self.alpha * h
        return out


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a Jittor model on Cora and write result.json."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config path.")
    parser.add_argument(
        "--model",
        choices=["gcnii", "appnp_cpu"],
        default=DEFAULT_CONFIG["model"],
        help="Use gcnii for the submitted solution, appnp_cpu for CPU smoke tests.",
    )
    parser.add_argument("--data-path", default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--output", default=DEFAULT_CONFIG["output"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--use-cuda", type=int, choices=[0, 1], default=DEFAULT_CONFIG["use_cuda"]
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--theta", type=float, default=DEFAULT_CONFIG["theta"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"]
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument(
        "--log-interval", type=int, default=DEFAULT_CONFIG["log_interval"]
    )
    return parser


def parse_args():
    parser = build_parser()
    partial_args, _ = parser.parse_known_args()

    if partial_args.config:
        if not osp.exists(partial_args.config):
            raise FileNotFoundError(
                f"Missing config: {partial_args.config}. "
                "Use configs/default.json or pass command line arguments."
            )
        with open(partial_args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        parser.set_defaults(**config)

    return parser.parse_args()


def set_seed(seed):
    """Set Jittor and NumPy random seeds."""
    jt.misc.set_global_seed(seed)
    np.random.seed(seed)


def load_raw_data(data_path):
    """Load the official Cora pickle file."""
    if not osp.exists(data_path):
        raise FileNotFoundError(
            f"Missing dataset: {data_path}. "
            "Place the official cora.pkl at data/cora.pkl or pass --data-path."
        )

    with open(data_path, "rb") as f:
        return pickle.load(f)


def normalize_features_np(x):
    row_sum = x.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-12] = 1.0
    return x / row_sum


def prepare_gcnii_data(raw):
    """Build Jittor tensors and sparse formats for GCNII."""
    from jittor_geometric.nn.conv.gcn_conv import gcn_norm
    from jittor_geometric.ops import cootocsc, cootocsr

    data = GraphData()
    data.x = jt.array(normalize_features_np(raw["x"].astype(np.float32)))
    data.y = jt.array(raw["y"].astype(np.int64))
    data.edge_index = jt.array(raw["edge_index"].astype(np.int64))
    data.train_mask = jt.array(raw["train_mask"])
    data.val_mask = jt.array(raw["val_mask"])
    data.test_mask = jt.array(raw["test_mask"])

    num_nodes = data.x.shape[0]
    edge_index, edge_weight = gcn_norm(
        data.edge_index,
        None,
        num_nodes,
        improved=False,
        add_self_loops=True,
    )

    with jt.no_grad():
        data.csc = cootocsc(edge_index, edge_weight, num_nodes)
        data.csr = cootocsr(edge_index, edge_weight, num_nodes)

    return data


def build_dense_norm_adj(edge_index, num_nodes):
    row, col = edge_index
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[row, col] = 1.0
    adj += np.eye(num_nodes, dtype=np.float32)

    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def prepare_appnp_cpu_data(raw):
    """Build dense CPU tensors for a quick no-GPU smoke test."""
    x_np = normalize_features_np(raw["x"].astype(np.float32))
    y_np = raw["y"].astype(np.int64)

    data = GraphData()
    data.x = jt.array(x_np)
    data.y = jt.array(y_np)
    data.norm_adj = jt.array(
        build_dense_norm_adj(raw["edge_index"].astype(np.int64), x_np.shape[0])
    )
    data.train_indices = jt.array(np.where(y_np != -1)[0].astype(np.int64))
    data.val_indices = jt.array(np.where(raw["val_mask"])[0].astype(np.int64))
    data.test_indices_np = np.where(raw["test_mask"])[0].astype(np.int64)
    return data


def save_prediction_dict(result, output_path):
    output_dir = osp.dirname(osp.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"预测结果已保存到 {output_path}")
    print(f"共预测 {len(result)} 个测试节点")


def run_gcnii(args, raw):
    data = prepare_gcnii_data(raw)
    model = GCNIINet(
        num_features=raw["num_features"],
        num_classes=raw["num_classes"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        alpha=args.alpha,
        theta=args.theta,
        dropout=args.dropout,
    )
    optimizer = nn.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = jt.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=1e-5
    )

    best_val_acc = 0
    best_model_state = None
    counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(data)
        loss = nn.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        optimizer.step(loss)
        scheduler.step()

        model.eval()
        with jt.no_grad():
            logits = model(data)
            accs = []
            for mask in [data.train_mask, data.val_mask]:
                pred = jt.argmax(logits[mask], dim=1)[0]
                acc = (pred == data.y[mask]).float32().mean().item()
                accs.append(acc)
            train_acc, val_acc = accs

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if epoch % args.log_interval == 0:
            log_str = (
                "Epoch: {:03d}, Loss: {:.4f}, "
                "Train Acc: {:.4f}, Val Acc: {:.4f}"
            )
            print(log_str.format(epoch, loss.item(), train_acc, val_acc))

        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\n最终结果: Val Acc: {best_val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with jt.no_grad():
        logits = model(data)
        pred = jt.argmax(logits, dim=1)[0]

    test_indices = jt.where(data.test_mask)[0]
    result = {str(int(idx)): int(pred[int(idx)]) for idx in test_indices}
    save_prediction_dict(result, args.output)


def run_appnp_cpu(args, raw):
    data = prepare_appnp_cpu_data(raw)
    model = APPNPCpuNet(
        raw["num_features"],
        raw["num_classes"],
        args.hidden_dim,
        args.dropout,
        args.alpha,
    )
    optimizer = nn.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(data, args.num_layers)
        loss = nn.cross_entropy_loss(
            logits[data.train_indices], data.y[data.train_indices]
        )
        optimizer.step(loss)

        if epoch % args.log_interval == 0:
            model.eval()
            with jt.no_grad():
                logits = model(data, args.num_layers)
                pred = jt.argmax(logits[data.val_indices], dim=1)[0]
                val_acc = (
                    pred == data.y[data.val_indices]
                ).float32().mean().item()
            print(
                "Epoch: {:03d}, Loss: {:.4f}, Val Acc: {:.4f}".format(
                    epoch, loss.item(), val_acc
                )
            )

    model.eval()
    with jt.no_grad():
        logits = model(data, args.num_layers)
        pred = jt.argmax(logits, dim=1)[0].numpy().astype(int)

    result = {str(int(idx)): int(pred[idx]) for idx in data.test_indices_np}
    save_prediction_dict(result, args.output)


def main():
    args = parse_args()
    jt.flags.use_cuda = args.use_cuda
    set_seed(args.seed)

    print("Config:")
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")

    raw = load_raw_data(args.data_path)
    if args.model == "gcnii":
        run_gcnii(args, raw)
    else:
        run_appnp_cpu(args, raw)


if __name__ == "__main__":
    main()
