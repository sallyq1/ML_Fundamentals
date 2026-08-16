
# Monotone-partitioned MARS: smooth → split at peaks/valleys → MARS on each slice → stitch endpoints

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars_1d import PartitionedMars, make_world1_data, plot_mars_fit


def main():
    x, y = make_world1_data()
    model = PartitionedMars(
        max_terms_per_slice=16,
        penalty=2.0,
        smooth_window=51,
        min_sep=0.35,
        min_prominence=0.45,
    ).fit(x, y)

    mse = float(((model.predict(x) - y.ravel()) ** 2).mean())
    print(f"Split knots: {model.knots}")
    print(f"Slices: {len(model.pieces)} | total hinges: {model.n_hinges} | train MSE: {mse:.4f}")
    for i, (a, b, piece) in enumerate(model.pieces):
        print(
            f"  slice {i}: [{a:6.2f}, {b:6.2f}]  hinges={piece.n_hinges}  "
            f"stitch y=({model.joint_y[i]:.2f} -> {model.joint_y[i + 1]:.2f})"
        )

    out = Path(__file__).resolve().parent / "mars_with_splits_fit.png"
    plot_mars_fit(
        x,
        y,
        model.predict,
        title=f"Monotone-partitioned MARS  ({len(model.pieces)} slices, {model.n_hinges} hinges)",
        knots=model.knots,
        save_path=out,
    )


if __name__ == "__main__":
    main()
