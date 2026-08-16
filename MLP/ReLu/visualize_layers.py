
# What the network looks like after 1, 2, and 3 hidden layers.

# The 1D "output after layer k" is a linear mix of that layer's nodes
# (the best mix that matches y). Layer 3 uses the trained output head.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from generate_world import get_y_outputs


N_HIDDEN = 8  # small enough to read, big enough to actually fit
WORLD = "world1"
X_MIN, X_MAX = -5.0, 5.0


class SmallDeepReLU(nn.Module):
    def __init__(self, n_hidden=N_HIDDEN):
        super().__init__()
        self.hidden = nn.Linear(1, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        self.output = nn.Linear(n_hidden, 1)
        self._init_weights()

    def _init_weights(self):
        # Tiny deep ReLUs often "die" (everything clipped to 0). A little positive
        # bias keeps more sticks off the floor at the start of training.
        for layer in (self.hidden, self.hidden2, self.hidden3):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        return self.output(x)

    def hidden_states(self, x):
        z1 = self.hidden(x)
        h1 = self.relu(z1)
        z2 = self.hidden2(h1)
        h2 = self.relu(z2)
        z3 = self.hidden3(h2)
        h3 = self.relu(z3)
        y_hat = self.output(h3)
        return (z1, h1), (z2, h2), (z3, h3), y_hat


def make_training_data(n=1000, seed=8826, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=(n, 1))
    y = get_y_outputs(WORLD, x) + rng.normal(0.0, noise, size=x.shape)
    return x, y


def train(x, y, epochs=6000, lr=0.01):
    model = SmallDeepReLU()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for epoch in range(epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | MSE Loss: {loss.item():.4f}")
    return model


def _readout_weights(H, y):
    A = np.column_stack([H, np.ones(len(H))])
    beta, *_ = np.linalg.lstsq(A, np.ravel(y), rcond=None)
    return beta


def _apply_readout(H, beta):
    A = np.column_stack([H, np.ones(len(H))])
    return A @ beta


def collect_curves(model, x_train, y_train, n_grid=800):
    x_grid = np.linspace(X_MIN, X_MAX, n_grid).reshape(-1, 1)
    xt = torch.tensor(x_train, dtype=torch.float32)
    xg = torch.tensor(x_grid, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        (_, h1_tr), (_, h2_tr), (_, h3_tr), _ = model.hidden_states(xt)
        (z1, h1), (z2, h2), (z3, h3), y_hat = model.hidden_states(xg)
    to_np = lambda t: t.detach().cpu().numpy()
    h1_tr, h2_tr = to_np(h1_tr), to_np(h2_tr)
    h1, h2, h3 = to_np(h1), to_np(h2), to_np(h3)
    # If we stopped after layer 1 or 2, this is the 1D curve those nodes can already draw
    y_after_l1 = _apply_readout(h1, _readout_weights(h1_tr, y_train))
    y_after_l2 = _apply_readout(h2, _readout_weights(h2_tr, y_train))
    return {
        "x": x_grid.ravel(),
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "z2": to_np(z2),
        "y_after_l1": y_after_l1,
        "y_after_l2": y_after_l2,
        "y_hat": to_np(y_hat).ravel(),
        "y_true": get_y_outputs(WORLD, x_grid).ravel(),
    }


def plot_output_after_each_layer(curves, x_train, y_train):
    x = curves["x"]
    rows = [
        (curves["y_after_l1"], "After layer 1 only", "tab:blue"),
        (curves["y_after_l2"], "After layer 2 ", "tab:orange"),
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
    colors = plt.cm.tab10(np.linspace(0, 1, N_HIDDEN, endpoint=False))
    x = curves["x"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    rows = [
        (axes[0], curves["h1"], "Layer 1"),
        (axes[1], curves["h2"], "Layer 2"),
        (axes[2], curves["h3"], "Layer 3"),
    ]
    for ax, acts, title in rows:
        for j in range(N_HIDDEN):
            ax.plot(x, acts[:, j], color=colors[j], lw=2, label=f"node {j}")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylabel("activation")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=N_HIDDEN, fontsize=8)

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
    x_train, y_train = make_training_data()
    model = train(x_train, y_train)
    curves = collect_curves(model, x_train, y_train)
    plot_output_after_each_layer(curves, x_train, y_train)
    plot_layer_stack(curves, x_train, y_train)


if __name__ == "__main__":
    main()
