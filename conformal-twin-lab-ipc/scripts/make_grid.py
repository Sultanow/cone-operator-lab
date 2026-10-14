"""Write the parameter grid for the Slurm job array: one line of CLI arguments per task.
Usage: python scripts/make_grid.py [--smoke | --ipc | --hp] > grid.txt ;  #tasks = number of lines."""
import itertools, sys
smoke = '--smoke' in sys.argv
if '--hp' in sys.argv:
    rows = []
    for wf, ww in itertools.product([1.0, 0.2, 0.05], [1e-3, 1e-2, 1e-1, 1.0]):
        rows.append(dict(levels='1,2,3,4', w_fit=wf, w_will=ww, barrier='on', seed=0, noise=0.0,
                         iters=800, sched='1e4,1e5,1e6', solver='ipc', target='P'))
elif '--ipc' in sys.argv:
    rows = []
    for wf, ww, seed in itertools.product([1.0, 0.2, 0.05, 0.01], [1e-3, 1e-2], [0, 1]):
        rows.append(dict(levels='1,2,3,4', w_fit=wf, w_will=ww, barrier='on', seed=seed, noise=0.0 if seed == 0 else 0.3,
                         iters=800, sched='1,100,1e4,1e6', solver='ipc'))
elif smoke:
    rows = [dict(levels='1,2', w_fit=1.0, w_will=1e-3, barrier='off', seed=0, noise=0.0, iters=40, sched='1,100,1e4'),
            dict(levels='1,2', w_fit=0.2, w_will=1e-3, barrier='on', seed=1, noise=0.3, iters=40, sched='1,100,1e4')]
else:
    rows = []
    for wf, ww, bar, seed in itertools.product([1.0, 0.2, 0.05, 0.01], [1e-3, 1e-2, 1e-1], ['on', 'off'], [0, 1, 2, 3]):
        rows.append(dict(levels='2,3,4', w_fit=wf, w_will=ww, barrier=bar, seed=seed, noise=0.0 if seed == 0 else 0.3,
                         iters=400, sched='1,10,100,1e3,1e4,1e5,1e6'))
for r in rows:
    print(f"--levels {r['levels']} --w-fit {r['w_fit']:g} --w-will {r['w_will']:g} --barrier {r['barrier']} "
          f"--seed {r['seed']} --noise {r['noise']:g} --iters {r['iters']} --wc-schedule {r['sched']} --solver {r.get('solver', 'lbfgs')}"
          + (f" --target {r['target']}" if 'target' in r else ''))
