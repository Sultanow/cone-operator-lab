import gmsh

def generate_ellipsoid_mesh(a=1.0, b=1.5, c=2.3, lc=0.2, outfile="ellipsoid.msh"):
    gmsh.initialize()
    gmsh.model.add("ellipsoid")

    # Unit sphere
    sph = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    gmsh.model.occ.synchronize()

    # Scale sphere -> ellipsoid
    gmsh.model.occ.dilate([(3, sph)], 0, 0, 0, a, b, c)
    gmsh.model.occ.synchronize()

    # Mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("General.Terminal", 1)  # keine GUI

    gmsh.model.mesh.generate(3)
    gmsh.write(outfile)
    print(f"Mesh written to: {outfile}")

    gmsh.finalize()

if __name__ == "__main__":
    generate_ellipsoid_mesh()
