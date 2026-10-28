import numpy as np

def load_eigenvalues_txt(path: str, n_test: int | None = None) -> np.ndarray:
    raw = np.loadtxt(path, dtype=float)
    raw = np.asarray(raw).ravel()
    raw = raw[np.isfinite(raw) & (raw > 0)]
    raw.sort()
    if n_test is not None:
        raw = raw[:n_test]
    return raw