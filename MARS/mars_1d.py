
# 1D MARS: greedy (hinge) regression

# y_hat(x) = b0 + sum_j b_j * ReLU(+=(x - t_j))

# Forward: add the hinge that cuts MSE the most
# Backward: drop hinges that GCV says are not worth it

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_RELU_DIR = Path(__file__).resolve().parents[1] / "MLP" / "ReLu"
if str(_RELU_DIR) not in sys.path:
    sys.path.insert(0, str(_RELU_DIR))


def _load_relu_module(name: str):
    path = _RELU_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Couldn't load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generate_training_data = _load_relu_module("generate_training_data")
_generate_world = _load_relu_module("generate_world")
generate_training_data = _generate_training_data.generate_training_data
get_y_outputs = _generate_world.get_y_outputs


def hinge(x, kind, knot):
    if kind == "plus":
        return np.maximum(x - knot, 0.0)
    if kind == "minus":
        return np.maximum(knot - x, 0.0)
    raise ValueError(kind)


def _design(x, terms):
    cols = [np.ones(len(x))]
    for kind, knot in terms:
        cols.append(hinge(x, kind, knot))
    return np.column_stack(cols)


def _fit_ls(H, y):
    beta, *_ = np.linalg.lstsq(H, y, rcond=None)
    y_hat = H @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    return beta, rss, y_hat


def _gcv(rss, n, n_basis, penalty):
    # Friedman: charge extra for each knot/hinge so we don't keep junk
    m_eff = 1.0 + penalty * max(n_basis - 1, 0)
    denom = 1.0 - m_eff / n
    if denom <= 1e-8:
        return np.inf
    return (rss / n) / (denom ** 2)


class Mars1D:
    def __init__(self, max_terms=30, penalty=2.0, n_candidate_knots=80):
        self.max_terms = max_terms
        self.penalty = penalty
        self.n_candidate_knots = n_candidate_knots
        self.terms = []  # list of (kind, knot)  kind in {"plus", "minus"}
        self.beta = None

    def fit(self, x, y):
        x = np.ravel(x).astype(float)
        y = np.ravel(y).astype(float)
        n = len(x)
        if n < 3:
            self.terms = []
            self.beta = np.array([float(np.mean(y))])
            return self

        candidates = np.unique(x)
        if len(candidates) > self.n_candidate_knots:
            pick = np.linspace(0, len(candidates) - 1, self.n_candidate_knots).astype(int)
            candidates = candidates[pick]

        terms = []
        used = set()
        H = np.ones((n, 1))
        _, rss, _ = _fit_ls(H, y)

        max_hinges = min(self.max_terms, n - 2)
        for _ in range(max_hinges):
            best = None
            for t in candidates:
                for kind in ("plus", "minus"):
                    key = (kind, float(t))
                    if key in used:
                        continue
                    col = hinge(x, kind, t)
                    if col.max() < 1e-12:
                        continue
                    H_try = np.column_stack([H, col])
                    if np.linalg.cond(H_try) > 1e10:
                        continue
                    beta, rss_try, _ = _fit_ls(H_try, y)
                    if best is None or rss_try < best[0]:
                        best = (rss_try, kind, float(t), H_try, beta)

            if best is None:
                break
            rss_try, kind, t, H_new, beta = best
            if rss - rss_try < 1e-10:
                break
            used.add((kind, t))
            terms.append((kind, t))
            H, rss = H_new, rss_try
            self.beta = beta

        self.terms, self.beta = self._prune(x, y, terms)
        return self

    def _prune(self, x, y, terms):
        n = len(x)
        H = _design(x, terms)
        beta, rss, _ = _fit_ls(H, y)
        best_terms = list(terms)
        best_beta = beta
        best_gcv = _gcv(rss, n, H.shape[1], self.penalty)

        improved = True
        while improved and best_terms:
            improved = False
            for drop in range(len(best_terms)):
                trial = best_terms[:drop] + best_terms[drop + 1 :]
                H_try = _design(x, trial)
                beta_try, rss_try, _ = _fit_ls(H_try, y)
                gcv_try = _gcv(rss_try, n, H_try.shape[1], self.penalty)
                if gcv_try < best_gcv - 1e-12:
                    best_gcv = gcv_try
                    best_terms = trial
                    best_beta = beta_try
                    improved = True
                    break
        return best_terms, best_beta

    def predict(self, x):
        x = np.ravel(x).astype(float)
        H = _design(x, self.terms)
        return H @ self.beta

    @property
    def n_hinges(self):
        return len(self.terms)


def moving_average(y, window=31):
    w = int(window)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, len(y) if len(y) % 2 == 1 else len(y) - 1)
    w = max(w, 3)
    kernel = np.ones(w) / w
    pad = w // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def monotonic_knots(x, y, smooth_window=51, min_sep=0.35, min_prominence=0.45):
    # Endpoints + extrema of a smoothed curve, tiny noise wiggles are ignored
    order = np.argsort(np.ravel(x))
    xs = np.ravel(x)[order]
    ys = np.ravel(y)[order]
    ys_s = moving_average(ys, smooth_window)
    d = np.diff(ys_s)
    sign = np.sign(d)
    sign[sign == 0] = 1
    flips = np.where(sign[:-1] * sign[1:] < 0)[0] + 1

    extrema = []
    for i in flips:
        if extrema and abs(ys_s[i] - ys_s[extrema[-1]]) < min_prominence:
            continue
        if extrema and (xs[i] - xs[extrema[-1]]) < min_sep:
            continue
        extrema.append(i)

    knots = [float(xs[0])]
    for i in extrema:
        t = float(xs[i])
        if t - knots[-1] >= min_sep:
            knots.append(t)
    if float(xs[-1]) - knots[-1] >= min_sep:
        knots.append(float(xs[-1]))
    else:
        knots[-1] = float(xs[-1])
    return np.array(knots), xs, ys_s


class LinearPiece:
    # Fallback when a slice has too few points for MARS
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.terms = []
        self.beta = np.array([0.0])

    def fit(self, x, y):
        x = np.ravel(x).astype(float)
        y = np.ravel(y).astype(float)
        if len(x) < 2:
            self.a, self.b = 0.0, float(np.mean(y)) if len(y) else 0.0
        else:
            self.a, self.b = np.polyfit(x, y, 1)
        self.beta = np.array([self.b, self.a])
        return self

    def predict(self, x):
        return self.a * np.ravel(x).astype(float) + self.b

    @property
    def n_hinges(self):
        return 0


class PartitionedMars:
    # 1) Smooth the scatter.
    # 2) Split at monotonic sign changes (peaks / valleys).
    # 3) Run 1D MARS on each interval (adaptive hinge count).
    # 4) Linear correction so adjacent pieces meet (C0 path).

    def __init__(self, max_terms_per_slice=16, penalty=2.0, smooth_window=51, min_sep=0.35, min_prominence=0.45):
        self.max_terms_per_slice = max_terms_per_slice
        self.penalty = penalty
        self.smooth_window = smooth_window
        self.min_sep = min_sep
        self.min_prominence = min_prominence
        self.pieces = []  # (a, b, model)
        self.knots = None
        self.joint_y = None  # y at each knot so the path is continuous

    def fit(self, x, y):
        x = np.ravel(x).astype(float)
        y = np.ravel(y).astype(float)
        knots, _, _ = monotonic_knots(
            x, y, self.smooth_window, self.min_sep, self.min_prominence
        )
        self.knots = knots
        models = []

        for i in range(len(knots) - 1):
            a, b = knots[i], knots[i + 1]
            if i == len(knots) - 2:
                mask = (x >= a) & (x <= b)
            else:
                mask = (x >= a) & (x < b)

            xa, ya = x[mask], y[mask]
            cap = max(2, min(self.max_terms_per_slice, xa.size // 6))
            if xa.size < 8:
                model = LinearPiece().fit(xa, ya) if xa.size else LinearPiece().fit(
                    np.array([a, b]), np.zeros(2)
                )
            else:
                model = Mars1D(max_terms=cap, penalty=self.penalty).fit(xa, ya)
            models.append(model)

        # C0 stitch: one shared y at each knot (average of the two touching models).
        joint_y = np.zeros(len(knots))
        joint_y[0] = float(models[0].predict(np.array([knots[0]]))[0])
        joint_y[-1] = float(models[-1].predict(np.array([knots[-1]]))[0])
        for i in range(1, len(knots) - 1):
            t = knots[i]
            left = float(models[i - 1].predict(np.array([t]))[0])
            right = float(models[i].predict(np.array([t]))[0])
            joint_y[i] = 0.5 * (left + right)

        self.joint_y = joint_y
        self.pieces = [
            (knots[i], knots[i + 1], models[i]) for i in range(len(models))
        ]
        return self

    def predict(self, x):
        x = np.ravel(x).astype(float)
        out = np.zeros_like(x, dtype=float)
        n_pieces = len(self.pieces)
        for i, (a, b, model) in enumerate(self.pieces):
            if i == n_pieces - 1:
                mask = (x >= a) & (x <= b)
            else:
                mask = (x >= a) & (x < b)
            if not np.any(mask):
                continue
            f = model.predict(x[mask])
            fa = float(model.predict(np.array([a]))[0])
            fb = float(model.predict(np.array([b]))[0])
            v_a, v_b = self.joint_y[i], self.joint_y[i + 1]
            alpha = v_a - fa
            slope_fix = ((v_b - fb) - (v_a - fa)) / (b - a + 1e-12)
            out[mask] = f + alpha + slope_fix * (x[mask] - a)
        if self.pieces:
            out[x < self.knots[0]] = self.joint_y[0]
            out[x > self.knots[-1]] = self.joint_y[-1]
        return out

    @property
    def n_hinges(self):
        return int(sum(p[2].n_hinges for p in self.pieces))


def make_world1_data():
    return generate_training_data("world1", 8826, 1000, "gaussian", "uniform", 0.3)


def plot_mars_fit(x_train, y_train, predict_fn, title, knots=None, save_path=None):
    x_grid = np.linspace(-5, 5, 1000)
    y_true = get_y_outputs("world1", x_grid.reshape(-1, 1)).ravel()
    y_hat = predict_fn(x_grid)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=6, c="lightgray", alpha=0.45, label="training data")
    plt.plot(x_grid, y_true, color="green", lw=2, label="true function")
    plt.plot(x_grid, y_hat, color="black", lw=2.4, label="MARS fit")
    if knots is not None:
        for i, t in enumerate(np.ravel(knots)):
            plt.axvline(t, color="tab:red", ls=":", alpha=0.7, label="monotonic split" if i == 0 else None)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=140)
    plt.show()
