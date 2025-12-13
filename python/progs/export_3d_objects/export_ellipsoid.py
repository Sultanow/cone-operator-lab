
# ------------------------------------------------------------
# Run the program
# cd c:/Users/sulta/git/cone-operator-lab/python/progs/export_3d_objects
# c:/Users/sulta/AppData/Local/Programs/Python/Python310/python export_ellipsoid.py
# ------------------------------------------------------------
import numpy as np

def ellipsoid_mesh(a, b, c, nu=200, nv=120):
    """
    Parametrisches Ellipsoid:
      x = a cos(u) sin(v)
      y = b sin(u) sin(v)
      z = c cos(v)
    u in [0, 2pi), v in [0, pi]
    """
    u = np.linspace(0.0, 2.0*np.pi, nu, endpoint=False)
    v = np.linspace(0.0, np.pi, nv, endpoint=True)

    uu, vv = np.meshgrid(u, v, indexing="xy")

    x = a * np.cos(uu) * np.sin(vv)
    y = b * np.sin(uu) * np.sin(vv)
    z = c * np.cos(vv)

    # Vertex-Liste (row-major)
    V = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Triangulation auf dem (u,v)-Grid
    def idx(i_v, i_u):
        return i_v * nu + i_u

    F = []
    for iv in range(nv - 1):
        for iu in range(nu):
            iu2 = (iu + 1) % nu  # wrap in u
            i0 = idx(iv, iu)
            i1 = idx(iv, iu2)
            i2 = idx(iv + 1, iu)
            i3 = idx(iv + 1, iu2)

            # Zwei Dreiecke pro Quad
            # Achtung: an den Polen degenerieren Dreiecke nicht, aber sind sehr klein.
            F.append([i0, i2, i1])
            F.append([i1, i2, i3])

    F = np.array(F, dtype=np.int32)
    return V, F

def vertex_normals(V, a, b, c):
    """
    Normale vom impliziten Ellipsoid:
      (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    Gradient ~ (2x/a^2, 2y/b^2, 2z/c^2)
    """
    n = np.stack([V[:,0]/(a*a), V[:,1]/(b*b), V[:,2]/(c*c)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n

def write_obj(path, V, F, N=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ellipsoid mesh\n")
        for v in V:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        if N is not None:
            for n in N:
                f.write(f"vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}\n")

        # OBJ ist 1-basiert
        if N is None:
            for tri in F:
                a, b, c = tri + 1
                f.write(f"f {a} {b} {c}\n")
        else:
            for tri in F:
                a, b, c = tri + 1
                # f v//vn v//vn v//vn
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")

if __name__ == "__main__":
    a, b, c = 1.0, 1.5, 2.3
    V, F = ellipsoid_mesh(a, b, c, nu=220, nv=140)
    N = vertex_normals(V, a, b, c)

    out = "ellipsoid_a1_b1p5_c2p3.obj"
    write_obj(out, V, F, N=N)
    print(f"Wrote {out} with {len(V)} vertices and {len(F)} triangles.")
