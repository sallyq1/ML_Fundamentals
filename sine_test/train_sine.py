"""Train 1-hidden-layer tanh MLPs on a noiseless sine; dump learned curves."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_TRAIN = 256
N_HIDDEN = 6
EPOCHS = 4000
LR = 0.04
X_MIN, X_MAX = -math.pi, math.pi
N_PLOT = 81
# Epoch 0 = random init, before any Adam step.
SNAPSHOT_EPOCHS = (0, 10, 50, 100, 250, 500, 1500, 4000)


def sample_x(rng: np.random.Generator, n: int = N_TRAIN) -> np.ndarray:
    return rng.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float64)


def sine(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


class MLP:
    """yhat = W2 @ tanh(W1 x + b1) + b2  (full batch, Adam)."""

    def __init__(self, rng: np.random.Generator, n_hidden: int = N_HIDDEN) -> None:
        # Xavier-ish
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(1), size=(n_hidden, 1))
        self.b1 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(n_hidden), size=(1, n_hidden))
        self.b2 = np.zeros((1, 1))
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1" : self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.W1 @ x.T + self.b1  # (H, N)
        h = np.tanh(n)
        yhat = self.W2 @ h + self.b2  # (1, N)
        return n, h, yhat

    def adam_step(self, grads: dict[str, np.ndarray], lr: float = LR) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1**self.t)
            vhat = self.v[k] / (1 - b2**self.t)
            p = self.params()[k]
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def train_mse(self, x: np.ndarray, y_row: np.ndarray) -> float:
        _, _, yhat = self.forward(x)
        err = yhat - y_row
        return float(np.mean(err * err))

    def fit(self, x: np.ndarray, y: np.ndarray, xs_plot: np.ndarray) -> tuple[float, list[dict]]:
        n_samples = x.shape[0]
        y_row = y.T
        snapshots: list[dict] = []
        snap_set = set(SNAPSHOT_EPOCHS)

        def maybe_snap(epoch: int, mse: float) -> None:
            if epoch in snap_set:
                print(f"    epoch {epoch:4d}  mse={mse:.6f}")
                snapshots.append(snapshot_mlp(self, epoch, mse, xs_plot))

        maybe_snap(0, self.train_mse(x, y_row))
        last_mse = 0.0
        for step in range(1, EPOCHS + 1):
            n, h, yhat = self.forward(x)
            err = yhat - y_row
            last_mse = float(np.mean(err * err))
            dy = (2.0 / n_samples) * err
            dW2 = dy @ h.T
            db2 = dy.sum(axis=1, keepdims=True)
            dh = self.W2.T @ dy
            dn = dh * (1.0 - h * h)
            dW1 = dn @ x
            db1 = dn.sum(axis=1, keepdims=True)
            self.adam_step({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
            if step in snap_set:
                last_mse = self.train_mse(x, y_row)
                maybe_snap(step, last_mse)
        return last_mse, snapshots


def _coeff(v: float, digits: int = 8) -> str:
    """ASCII coefficient for Desmos (no unicode minus, no scientific notation)."""
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "0"
    return s


def _affine(w: float, b: float) -> str:
    b_op = "+" if b >= 0 else "-"
    return f"{_coeff(w)}x {b_op} {_coeff(abs(b))}"


def desmos_equation(run: dict, name: str = "y") -> str:
    """One Desmos expression: y = B + sum W_i tanh(w_i x + b_i)."""
    parts: list[str] = []

    def add_term(coef: float, body: str | None) -> None:
        piece = _coeff(abs(coef)) if body is None else f"{_coeff(abs(coef))}*{body}"
        if not parts:
            parts.append(piece if coef >= 0 else f"- {piece}")
        else:
            op = "+" if coef >= 0 else "-"
            parts.append(f"{op} {piece}")

    B = float(run["B"])
    if abs(B) > 1e-12:
        add_term(B, None)
    for u in run["units"]:
        add_term(float(u["W"]), f"tanh({_affine(float(u['w']), float(u['b']))})")
    rhs = " ".join(parts) if parts else "0"
    return f"{name} = {rhs}"


def snapshot_mlp(mlp: MLP, epoch: int, mse: float, xs: np.ndarray) -> dict:
    """Compact checkpoint: weights, Desmos line, and ŷ on the plot grid."""
    x_col = xs.reshape(-1, 1)
    _, _, yhat = mlp.forward(x_col)
    units = [
        {
            "i": i,
            "w": float(mlp.W1[i, 0]),
            "b": float(mlp.b1[i, 0]),
            "W": float(mlp.W2[0, i]),
        }
        for i in range(mlp.W1.shape[0])
    ]
    B = float(mlp.b2[0, 0])
    payload = {"B": B, "units": units}
    return {
        "epoch": epoch,
        "mse": round(mse, 6),
        "B": round(B, 8),
        "units": units,
        "yhat": [round(float(v), 4) for v in yhat.ravel()],
        "desmos": desmos_equation(payload, name=f"y_{epoch}"),
    }


def weights_markdown(snap: dict) -> list[str]:
    lines = [
        "| unit | w (into tanh) | b (hidden bias) | W (output weight) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for u in snap["units"]:
        lines.append(
            f"| {int(u['i']) + 1} | {_coeff(u['w'])} | {_coeff(u['b'])} | {_coeff(u['W'])} |"
        )
    lines.append(f"| output bias B |  |  | {_coeff(snap['B'])} |")
    return lines


def desmos_unit_lines(run: dict) -> list[str]:
    """Optional per-neuron expressions so each tanh can be plotted alone."""
    lines = []
    for u in run["units"]:
        i = int(u["i"]) + 1
        W, w, b = float(u["W"]), float(u["w"]), float(u["b"])
        lines.append(f"u_{i}={_coeff(W)}*tanh({_affine(w, b)})")
    sum_u = " + ".join(f"u_{i}" for i in range(1, len(run["units"]) + 1))
    B = float(run["B"])
    b_op = "+" if B >= 0 else "-"
    lines.append(f"y = {sum_u} {b_op} {_coeff(abs(B))}")
    return lines



def centers_steepness(W1: np.ndarray, b1: np.ndarray) -> list[dict]:
    out = []
    for i in range(W1.shape[0]):
        w = float(W1[i, 0])
        b = float(b1[i, 0])
        center = float(-b / w) if abs(w) > 1e-8 else None
        out.append({"w": w, "b": b, "center": center, "steepness": abs(w)})
    return out


def pack_run(
    name: str,
    init_seed: int,
    data_seed: int,
    mlp: MLP,
    x_train: np.ndarray,
    mse: float,
    snapshots: list[dict],
) -> dict:
    xs = np.linspace(X_MIN, X_MAX, N_PLOT)
    x_col = xs.reshape(-1, 1)
    n, h, yhat = mlp.forward(x_col)
    contrib = (mlp.W2.T * h)  # (H, P) each row is W_i tanh(...)
    units = []
    for i in range(N_HIDDEN):
        units.append(
            {
                "i": i,
                "w": float(mlp.W1[i, 0]),
                "b": float(mlp.b1[i, 0]),
                "W": float(mlp.W2[0, i]),
                "tanh": [round(float(v), 4) for v in h[i]],
                "contrib": [round(float(v), 4) for v in contrib[i]],
            }
        )
    packed = {
        "name": name,
        "init_seed": init_seed,
        "data_seed": data_seed,
        "mse": round(mse, 6),
        "B": round(float(mlp.b2[0, 0]), 8),
        "x": [round(float(v), 4) for v in xs],
        "target": [round(float(v), 4) for v in sine(xs)],
        "yhat": [round(float(v), 4) for v in yhat.ravel()],
        "units": units,
        "geometry": centers_steepness(mlp.W1, mlp.b1),
        "n_train": int(x_train.shape[0]),
        "x_train_min": round(float(x_train.min()), 4),
        "x_train_max": round(float(x_train.max()), 4),
        "desmos": desmos_equation({"B": float(mlp.b2[0, 0]), "units": units}),
        "snapshots": snapshots,
    }
    return packed


def plot_runs(runs: list[dict], path: Path, show: bool = True) -> None:
    """Overlay MLP ŷ vs sin(x) for each run; residual on the right."""
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    for i, r in enumerate(runs):
        xs = r["x"]
        ax = axes[i, 0]
        ax.plot(xs, r["target"], color="green", lw=2, label="sin(x)")
        ax.plot(xs, r["yhat"], color="black", lw=1.8, ls="--", label="MLP ŷ")
        ax.set_title(f"{r['name']}   mse={r['mse']:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axhline(0, color="0.85", lw=0.8)
        if i == 0:
            ax.legend(loc="upper right")

        resid = [yh - t for yh, t in zip(r["yhat"], r["target"])]
        axr = axes[i, 1]
        axr.plot(xs, resid, color="tab:red", lw=1.5)
        axr.axhline(0, color="0.85", lw=0.8)
        axr.set_title("residual  ŷ − sin(x)")
        axr.set_xlabel("x")
        axr.set_ylabel("ŷ − sin(x)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_progress(runs: list[dict], path: Path, show: bool = False) -> None:
    """sin(x) vs the network at each snapshot, one row per run."""
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
    cmap = plt.colormaps["viridis"]
    for i, r in enumerate(runs):
        ax = axes[i, 0]
        xs = r["x"]
        snaps = r.get("snapshots") or []
        ax.plot(xs, r["target"], color="green", lw=2.2, label="sin(x)", zorder=3)
        for j, s in enumerate(snaps):
            frac = j / max(len(snaps) - 1, 1)
            ax.plot(xs, s["yhat"], color=cmap(frac), lw=1.3, label=f"epoch {s['epoch']}")
        ax.set_title(f"{r['name']}   final mse={r['mse']:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def train_one(init_seed: int, data_seed: int, label: str) -> dict:
    print(f"\n=== {label}  init={init_seed}  data={data_seed} ===")
    data_rng = np.random.default_rng(data_seed)
    x = sample_x(data_rng)
    y = sine(x)
    init_rng = np.random.default_rng(init_seed)
    mlp = MLP(init_rng)
    xs_plot = np.linspace(X_MIN, X_MAX, N_PLOT)
    mse, snapshots = mlp.fit(x, y, xs_plot)
    return pack_run(label, init_seed, data_seed, mlp, x, mse, snapshots)


def main() -> None:
    # Four experimental cells:
    # A: seed 1, data 1
    # B: seed 2, data 1   (different seeds, same inputs)
    # C: seed 1, data 1   (same seeds, same inputs) — replay of A
    # D: seed 1, data 2   (same seeds, different inputs)
    # E: seed 2, data 2   (different seeds, different inputs)
    runs = [
        train_one(1, 10, "seed1_data1"),
        train_one(2, 10, "seed2_data1"),
        train_one(1, 10, "seed1_data1_replay"),
        train_one(1, 20, "seed1_data2"),
        train_one(2, 20, "seed2_data2"),
    ]
    out = {
        "setup": {
            "target": "sin(x)",
            "domain": [X_MIN, X_MAX],
            "n_train": N_TRAIN,
            "n_hidden": N_HIDDEN,
            "noise": 0,
            "sampling": "i.i.d. uniform, not a grid",
            "architecture": "yhat = B + sum_i W_i * tanh(w_i * x + b_i)",
            "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        },
        "runs": runs,
    }
    path = Path(__file__).with_name("sine_runs.json")
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwrote {path}")
    for r in runs:
        print(f"  {r['name']:22s} mse={r['mse']:.6f}  B={r['B']}")
    here = Path(__file__).resolve().parent
    plot_runs(runs, here / "sine_fit.png", show=False)
    plot_progress(runs, here / "sine_progress.png", show=False)


if __name__ == "__main__":
    main()
