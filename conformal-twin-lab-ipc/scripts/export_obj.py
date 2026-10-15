"""Export meshes for external viewers (Mathematica, Blender, MeshLab, ParaView).
  python scripts/export_obj.py polyhedron OUTDIR            -> mizhaev_P.obj/.json (8 nonagons) and mizhaev_K.obj/.json (56 triangles)
  python scripts/export_obj.py RESULT_L?.npz OUTDIR         -> a surface from a sweep checkpoint, grouped by nonagon
OBJ files carry one group per nonagon ('g nonagon_k'); the JSON files hold vertices, faces and the nonagon of each face."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src'); sys.path.insert(0, SRC)
R = json.load(open(os.path.join(SRC, 'results.json'))); D = json.load(open(os.path.join(SRC, 'development.json')))
def write(prefix, V, faces, group):
    with open(prefix + '.obj', 'w') as fh:
        fh.write(f'# {os.path.basename(prefix)}: {len(V)} vertices, {len(faces)} faces, one group per nonagon\n')
        for v in V: fh.write('v %.10g %.10g %.10g\n' % tuple(v))
        for g in sorted(set(group)):
            fh.write(f'g nonagon_{g + 1}\n')
            for f, gg in zip(faces, group):
                if gg == g: fh.write('f ' + ' '.join(str(int(i) + 1) for i in f) + '\n')
    json.dump(dict(vertices=np.asarray(V, float).tolist(), faces=[[int(i) + 1 for i in f] for f in faces],
                   nonagon=[int(g) + 1 for g in group]), open(prefix + '.json', 'w'))
    print('wrote', prefix + '.obj', prefix + '.json')
what, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
if what == 'polyhedron':
    V = np.array(R['vertices'], float); F = [[x - 1 for x in f] for f in R['faces']]
    write(os.path.join(out, 'mizhaev_P'), V, F, list(range(8)))
    T = [[x - 1 for x in t] for t in R['triangles']]
    write(os.path.join(out, 'mizhaev_K'), V, T, [[k for k, f in enumerate(F) if set(t) <= set(f)][0] for t in T])
else:
    from fine_mesh import build
    Z = np.load(what); lev = int(Z['level']); M = build(lev, SRC)
    assert np.array_equal(Z['F'], M['F']), 'checkpoint and mesh do not match'
    group = np.array(D['face_of'])[M['face_tri']]
    write(os.path.join(out, os.path.splitext(os.path.basename(what))[0]), Z['X'], Z['F'], list(group))
