# What the network looks like after 1, 2, and 3 hidden layers.

# The 1D "output after layer k" is a linear mix of that layer's nodes
# (the best mix that matches y). Layer 3 uses the trained output head.

import numpy as np
import torch
import matplotlib.pyplot as plt

from generate_training_data import generate_training_data
from generate_world import get_y_outputs
from neural_network import convert_numpy_to_tensors, train_neural_network


WORLD = "world1"


def hidden_activations(model, x):
    h1 = model.relu(model.hidden(x))
    h2 = model.relu(model.hidden2(h1))
    h3 = model.relu(model.hidden3(h2))
    y_hat = model.output(h3)
    return h1, h2, h3, y_hat


def _readout_weights(H, y):
    A = np.column_stack([H, np.ones(len(H))])
    beta, *_ = np.linalg.lstsq(A, np.ravel(y), rcond=None)
    return beta


def _apply_readout(H, beta):
    A = np.column_stack([H, np.ones(len(H))])
    return A @ beta


def collect_curves(model, x_train, y_train, world_type=WORLD, start_window=-5, end_window=5, step_count=1000):
    x_grid = np.linspace(start_window, end_window, step_count).reshape(-1, 1)
    xt, _ = convert_numpy_to_tensors(x_train, y_train)
    xg = torch.tensor(x_grid, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        h1_tr, h2_tr, _, _ = hidden_activations(model, xt)
        h1, h2, h3, y_hat = hidden_activations(model, xg)

    h1_tr, h2_tr = h1_tr.numpy(), h2_tr.numpy()
    h1, h2, h3 = h1.numpy(), h2.numpy(), h3.numpy()

    # If we stopped after layer 1 or 2, this is the 1D curve those nodes can already draw
    y_after_l1 = _apply_readout(h1, _readout_weights(h1_tr, y_train))
    y_after_l2 = _apply_readout(h2, _readout_weights(h2_tr, y_train))
    return {
        "x": x_grid.ravel(),
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "y_after_l1": y_after_l1,
        "y_after_l2": y_after_l2,
        "y_hat": y_hat.numpy().ravel(),
        "y_true": get_y_outputs(world_type, x_grid).ravel(),
    }


def plot_output_after_each_layer(curves, x_train, y_train):
    x = curves["x"]
    rows = [
        (curves["y_after_l1"], "After layer 1 only", "tab:blue"),
        (curves["y_after_l2"], "After layer 2", "tab:orange"),
        (curves["y_hat"], "After layer 3", "black"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    for ax, (y_depth, title, color) in zip(axes, rows):
        ax.scatter(x_train, y_train, s=6, c="lightgray", alpha=0.4, zorder=1)
        ax.plot(x, curves["y_true"], color="green", lw=2, label="true function", zorder=2)
        ax.plot(x, y_depth, color=color, lw=2.4, label="output at this depth", zorder=3)
        ax.set_title(title)
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("x")
    fig.suptitle("How the polyline changes as each hidden layer is applied", fontsize=13)
    fig.tight_layout()
    fig.savefig("output_after_each_layer.png", dpi=140)
    plt.show()


def plot_layer_stack(curves, x_train, y_train):
    n_hidden = curves["h1"].shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_hidden, endpoint=False))
    x = curves["x"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    rows = [
        (axes[0], curves["h1"], "Layer 1"),
        (axes[1], curves["h2"], "Layer 2"),
        (axes[2], curves["h3"], "Layer 3"),
    ]
    for ax, acts, title in rows:
        for j in range(n_hidden):
            ax.plot(x, acts[:, j], color=colors[j], lw=1)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylabel("activation")
        ax.set_title(title)

    ax = axes[3]
    ax.scatter(x_train, y_train, s=6, c="lightgray", alpha=0.45, label="training data", zorder=1)
    ax.plot(x, curves["y_true"], color="green", lw=2, label="true function", zorder=2)
    ax.plot(x, curves["y_hat"], color="black", lw=2.4, label="model output (mix of layer 3)", zorder=3)
    ax.set_title("Final Output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("layer_stack.png", dpi=140)
    plt.show()


def main():
    x_train, y_train = generate_training_data("world1", 8826, 1000, "gaussian", "uniform", 0.3)
    model = train_neural_network(x_train, y_train, epochs=20000, learning_rate=0.01)
    curves = collect_curves(model, x_train, y_train)
    plot_output_after_each_layer(curves, x_train, y_train)
    plot_layer_stack(curves, x_train, y_train)


if __name__ == "__main__":
    main()
