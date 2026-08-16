# Global 1D MARS on world1: one pile of hinges over the whole x range

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars_1d import Mars1D, make_world1_data, plot_mars_fit


def main():
    x, y = make_world1_data()
    model = Mars1D(max_terms=40, penalty=2.0).fit(x, y)
    mse = float(((model.predict(x) - y.ravel()) ** 2).mean())
    print(f"Hinges kept: {model.n_hinges} | train MSE: {mse:.4f}")
    for kind, knot in model.terms:
        print(f"  {kind:5s}  knot={knot:7.3f}")

    out = Path(__file__).resolve().parent / "basic_mars_fit.png"
    plot_mars_fit(
        x,
        y,
        model.predict,
        title=f"Basic 1D MARS  ({model.n_hinges} hinges)",
        save_path=out,
    )


if __name__ == "__main__":
    main()
