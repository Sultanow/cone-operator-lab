# c:/Users/sulta/AppData/Local/Programs/Python/Python310/Scripts/pip install physics-tenpy
# c:/Users/sulta/AppData/Local/Programs/Python/Python310/python ellipsoid_eigs_tenpy.py

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tenpy.linalg import np_conserved as npc
from tenpy.linalg.krylov_based import Arnoldi

# ------------------------------------------------------------
# 1. Konfiguration & Geometrie
# ------------------------------------------------------------
FILENAME_TXT = "eigenvalues_tenpy.txt"
FILENAME_PNG = "eigenvector_slice.png"

a, b, c = 1.0, 1.5, 2.3
nx = ny = nz = 41  # Gute Auflösung für schnelle Demo

print(f"Initialisiere Gitter {nx}x{ny}x{nz}...")

x = np.linspace(-a, a, nx)
y = np.linspace(-b, b, ny)
z = np.linspace(-c, c, nz)

hx, hy, hz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Ellipsoid-Maske
inside = (X/a)**2 + (Y/b)**2 + (Z/c)**2 <= 1.0

# Index-Mapping
idx = -np.ones_like(X, dtype=int)
counter = 0
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            if inside[i, j, k]:
                idx[i, j, k] = counter
                counter += 1
num_dofs = counter
print(f"Anzahl Freiheitsgrade (Matrixgröße): {num_dofs}")

# ------------------------------------------------------------
# 2. Sparse Matrix Bauen (Laplace)
# ------------------------------------------------------------
rows, cols, data = [], [], []
ix2, iy2, iz2 = 1/hx**2, 1/hy**2, 1/hz**2
diag_val = 2*(ix2 + iy2 + iz2)

# Nachbar-Offsets und Werte
offsets = [(-1,0,0, ix2), (1,0,0, ix2), 
           (0,-1,0, iy2), (0,1,0, iy2), 
           (0,0,-1, iz2), (0,0,1, iz2)]

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            p = idx[i, j, k]
            if p < 0: continue
            
            # Diagonale
            rows.append(p); cols.append(p); data.append(diag_val)
            
            # Nachbarn
            for di, dj, dk, val in offsets:
                ni, nj, nk = i+di, j+dj, k+dk
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    q = idx[ni, nj, nk]
                    if q >= 0:
                        rows.append(p); cols.append(q); data.append(-val)

L_sparse = sp.csr_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs))

# ------------------------------------------------------------
# 3. TenPy Wrapper & Solver
# ------------------------------------------------------------
class TenpySparseWrapper:
    def __init__(self, mat):
        self.mat = mat
        self.dtype = mat.dtype
        self.labels = ['p', 'p*'] # Labels sind wichtig für TenPy Checks

    def matvec(self, vec_npc):
        v_flat = vec_npc.to_ndarray().ravel()
        res_flat = self.mat.dot(v_flat)
        return npc.Array.from_ndarray_trivial(res_flat, labels=['p'])

print("Starte TenPy Arnoldi Solver...")
H_op = TenpySparseWrapper(L_sparse)

# Zufälliger Startvektor
rng = np.random.default_rng(42)
psi0_flat = rng.normal(size=num_dofs)
psi0_flat /= np.linalg.norm(psi0_flat)
psi0 = npc.Array.from_ndarray_trivial(psi0_flat, labels=['p'])

# Solver Optionen
options = {
    "num_ev": 20,
    "which": "SM",   # Smallest Magnitude (kleinste Energie)
    "N_max": 60,     # Krylov-Raum Größe
    "P_tol": 1e-8,   # Toleranz
    "max_iter": 1000
}

arn = Arnoldi(H_op, psi0, options)
evals, evecs, _ = arn.run()

# Sortieren (Realteil)
sort_perm = np.argsort(np.real(evals))
evals_sorted = np.real(np.array(evals)[sort_perm])
evecs_sorted = [evecs[i] for i in sort_perm]

# ------------------------------------------------------------
# 4. Export: Textdatei
# ------------------------------------------------------------
print(f"Exportiere Eigenwerte nach '{FILENAME_TXT}'...")
header_txt = f"Dirichlet Eigenwerte Ellipsoid (a={a}, b={b}, c={c})\nSolver: TenPy Arnoldi\nGrid: {nx}x{ny}x{nz}"
np.savetxt(FILENAME_TXT, evals_sorted, header=header_txt, fmt="%.8f")

# ------------------------------------------------------------
# 5. Export: PNG (Visualisierung Eigenvektor 0)
# ------------------------------------------------------------
print(f"Exportiere Plot nach '{FILENAME_PNG}'...")

# 1. Vektor zurück in 3D Grid mappen
vec_flat = evecs_sorted[0].to_ndarray().ravel() # Grundzustand
grid_3d = np.zeros((nx, ny, nz))
# Wir nehmen den Betrag, falls durch Numerik kleine imaginäre Teile entstanden sind,
# und np.real für das Vorzeichen
grid_3d[idx >= 0] = np.real(vec_flat) 

# 2. Schnitt durch die Mitte (z-Achse)
z_idx = nz // 2
slice_2d = grid_3d[:, :, z_idx]

# 3. Plotten
plt.figure(figsize=(8, 6))
# Transponieren (.T), damit x horizontal und y vertikal ist, origin='lower' für korrekte Orientierung
plt.imshow(slice_2d.T, origin='lower', extent=[-a, a, -b, b], cmap='viridis', interpolation='nearest')
plt.colorbar(label='Amplitude $\psi(x,y,0)$')
plt.title(f"Grundzustand (Eigenwert $\lambda_0 \\approx {evals_sorted[0]:.2f}$)\nSchnitt z=0")
plt.xlabel("x")
plt.ylabel("y")

# Ellipse einzeichnen zur Orientierung
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(a*np.cos(theta), b*np.sin(theta), 'w--', linewidth=1, label='Rand')
plt.legend(loc='upper right')

plt.savefig(FILENAME_PNG, dpi=300)
plt.close() # Fenster schließen, damit Skript sauber beendet
