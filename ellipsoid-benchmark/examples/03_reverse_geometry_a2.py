from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from hearing_ellipsoid_bench.io.eigenvalues import load_eigenvalues_txt
from hearing_ellipsoid_bench.validation.weyl import reverse_geometry_table
from hearing_ellipsoid_bench.viz.plots import plot_reverse_geometry


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
OUT = DATA / "outputs" / "reverse_geometry_a2"
OUT.mkdir(parents=True, exist_ok=True)

# Replace this with your freshly computed eigenvalue file from the larger machine.
EIG_FILES = {
    "ssg_current": DATA / "generated" / "ssg_a1_b1.5_c2.3_eigenvalues.txt",
}


def main() -> None:
    eigs_by_method = {
        name: load_eigenvalues_txt(path)
        for name, path in EIG_FILES.items()
        if Path(path).exists()
    }
    if not eigs_by_method:
        raise FileNotFoundError("No eigenvalue files found. Update EIG_FILES first.")

    k_values = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]
    df = reverse_geometry_table(eigs_by_method, a=1.0, b=1.5, c=2.3, k_values=k_values)
    df.to_csv(OUT / "reverse_geometry_a2_table.csv", index=False)

    fig = plot_reverse_geometry(df)
    fig.savefig(OUT / "reverse_geometry_a2_errors.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(df.tail())
    print(f"Saved outputs to: {OUT}")


if __name__ == "__main__":
    main()
