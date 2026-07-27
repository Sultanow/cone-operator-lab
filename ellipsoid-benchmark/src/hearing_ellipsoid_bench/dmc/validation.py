# =============================================================================
#  DMC ZELLE 1 — Diffusion Monte Carlo für Grundzustand-Eigenwert
#  Bridge-corrected, parallelisiert über 20 Cores
# =============================================================================
import numpy as np
import time
from multiprocessing import Pool
from hearing_ellipsoid_bench.core.types import AlgorithmResult

def dmc_single_run(args):
    """Eine DMC-Trajektorie. Gibt counts(t) array zurück."""
    a, b, c, n_walkers, n_steps, dt, seed = args
    rng = np.random.default_rng(seed)
    
    # Init: walker uniform im Ellipsoid via rejection sampling
    needed = n_walkers
    chunks_x, chunks_y, chunks_z = [], [], []
    while needed > 0:
        n_try = max(int(needed * 2.5), 1000)
        x = rng.uniform(-a, a, n_try)
        y = rng.uniform(-b, b, n_try)
        z = rng.uniform(-c, c, n_try)
        inside = (x/a)**2 + (y/b)**2 + (z/c)**2 < 1.0
        x_in = x[inside]; y_in = y[inside]; z_in = z[inside]
        take = min(len(x_in), needed)
        chunks_x.append(x_in[:take])
        chunks_y.append(y_in[:take])
        chunks_z.append(z_in[:take])
        needed -= take
    pos = np.column_stack([np.concatenate(chunks_x),
                            np.concatenate(chunks_y),
                            np.concatenate(chunks_z)])
    
    sigma = np.sqrt(2.0 * dt)
    inv_a2 = 1.0/a**2; inv_b2 = 1.0/b**2; inv_c2 = 1.0/c**2
    counts = np.empty(n_steps + 1, dtype=np.float64)
    counts[0] = len(pos)
    
    def signed_distance(p):
        """Pseudo-signed-distance to ellipsoid surface (>0 inside, <0 outside)."""
        x = p[:, 0]; y = p[:, 1]; z = p[:, 2]
        f = 1.0 - (x*x*inv_a2 + y*y*inv_b2 + z*z*inv_c2)
        gnorm = 2.0 * np.sqrt((x*inv_a2)**2 + (y*inv_b2)**2 + (z*inv_c2)**2)
        return f / np.maximum(gnorm, 1e-12)
    
    for step in range(n_steps):
        if len(pos) == 0:
            counts[step+1:] = 0
            break
        d_old = signed_distance(pos)
        pos_new = pos + sigma * rng.standard_normal(pos.shape)
        d_new = signed_distance(pos_new)
        
        # Hard kill: walker out
        alive_hard = d_new > 0
        # Bridge correction: Brownsche Bahn könnte zwischen Schritten Wand getroffen haben
        # P(touched | both endpoints inside) = exp(- d_old * d_new / dt)
        bridge_killed = np.zeros(len(pos), dtype=bool)
        both_in = alive_hard & (d_old > 0)
        if both_in.any():
            with np.errstate(over='ignore'):
                p_kill = np.exp(-d_old[both_in] * d_new[both_in] / dt)
            r = rng.random(both_in.sum())
            kills = np.where(both_in)[0][r < p_kill]
            bridge_killed[kills] = True
        
        survive = alive_hard & ~bridge_killed
        pos = pos_new[survive]
        counts[step+1] = len(pos)
    
    return counts


def fit_lambda1_from_counts(counts, dt, warmup_frac=0.30, fit_frac=0.50,
                             min_walkers=200):
    """Fit log N(t) = log N_0 - lambda_1 t über stabilen Bereich."""
    n = len(counts)
    warm = int(n * warmup_frac)
    end = warm + int(n * fit_frac)
    end = min(end, n)
    valid = counts[warm:end] > min_walkers
    if valid.sum() < 10:
        valid = counts[warm:end] > 20
    if valid.sum() < 5:
        return np.nan
    ts = (np.arange(warm, end) * dt)[valid]
    log_n = np.log(counts[warm:end][valid])
    slope, _ = np.polyfit(ts, log_n, 1)
    return -slope


def dmc_lambda1(a, b, c, n_walkers_per_run, n_steps, dt,
                n_runs, n_cores=20, base_seed=0):
    """Parallel DMC: n_runs separate trajectories, returns array of estimates."""
    args_list = [(a, b, c, n_walkers_per_run, n_steps, dt,
                  base_seed + run_id * 9973 + int(dt * 1e8))
                 for run_id in range(n_runs)]
    
    with Pool(processes=min(n_cores, n_runs)) as pool:
        all_counts = pool.map(dmc_single_run, args_list)
    
    estimates = np.array([fit_lambda1_from_counts(c, dt) for c in all_counts])
    return estimates


def dmc_with_dt_extrapolation(a, b, c,
                               n_walkers_per_run=1_000_000,
                               n_runs_per_dt=20,
                               dts=(0.004, 0.002, 0.001, 0.0005),
                               base_n_steps=400,
                               n_cores=20,
                               verbose=True):
    """Vollständiger DMC-Lauf mit dt->0 Extrapolation."""
    t0_total = time.perf_counter()
    
    if verbose:
        print(f"=== DMC with dt-extrapolation: a={a}, b={b}, c={c} ===")
        print(f"  walkers per run: {n_walkers_per_run:,}")
        print(f"  runs per dt:     {n_runs_per_dt}")
        print(f"  dt values:       {list(dts)}")
        print(f"  cores:           {n_cores}")
        total_walkers = n_walkers_per_run * n_runs_per_dt * len(dts)
        print(f"  total walkers:   {total_walkers:,}\n")
    
    results = {}
    for dt in dts:
        n_steps = int(base_n_steps * (0.001 / dt))  # konstante tau-Länge
        if verbose:
            print(f"  dt={dt} ({n_steps} steps) ...", flush=True)
        t0 = time.perf_counter()
        estimates = dmc_lambda1(a, b, c, n_walkers_per_run, n_steps, dt,
                                 n_runs=n_runs_per_dt, n_cores=n_cores,
                                 base_seed=int(dt * 1e9))
        results[dt] = estimates
        if verbose:
            valid = ~np.isnan(estimates)
            mean = estimates[valid].mean()
            sem = estimates[valid].std(ddof=1) / np.sqrt(valid.sum())
            print(f"    -> mean={mean:.5f}, sem={sem:.5f}, "
                  f"({valid.sum()}/{len(estimates)} valid runs, "
                  f"{time.perf_counter()-t0:.1f}s)")
    
    # dt -> 0 Extrapolation (linear)
    dts_arr = np.array(list(results.keys()))
    means = np.array([results[dt][~np.isnan(results[dt])].mean() for dt in dts_arr])
    sems = np.array([
        results[dt][~np.isnan(results[dt])].std(ddof=1) / np.sqrt((~np.isnan(results[dt])).sum())
        for dt in dts_arr
    ])
    
    coef, cov = np.polyfit(dts_arr, means, 1, cov=True, w=1.0/np.maximum(sems, 1e-6))
    slope, intercept = coef
    intercept_err = np.sqrt(cov[1, 1])
    
    if verbose:
        print(f"\n=== dt -> 0 extrapolation ===")
        print(f"  linear fit: lambda_1(dt) = {intercept:.5f} + {slope:.2f} * dt")
        print(f"  estimate:   lambda_1 = {intercept:.5f} +/- {intercept_err:.5f}")
        print(f"  total wall: {time.perf_counter()-t0_total:.1f}s")
    
    return {
        "lambda_1": intercept,
        "lambda_1_err": intercept_err,
        "slope": slope,
        "results_per_dt": results,
        "dts": dts_arr,
        "means": means,
        "sems": sems,
        "wall_time": time.perf_counter() - t0_total,
    }

def solve_dmc_lambda1(
    a: float,
    b: float,
    c: float,
    n_walkers_per_run: int = 1_000_000,
    n_runs_per_dt: int = 20,
    dts=(0.004, 0.002, 0.001, 0.0005),
    base_n_steps: int = 400,
    n_cores: int = 20,
    verbose: bool = True,
) -> "AlgorithmResult":
    """DMC-Ground-State-Solver im AlgorithmResult-Vertrag.

    Liefert ein AlgorithmResult mit:
        algorithm = "dmc_bridge_corrected"
        eigs      = np.array([lambda_1])
        meta      = {lambda_1_err, slope, dts, means, sems, ...}
    """
    from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues
    import numpy as np
    import time

    name = "dmc_bridge_corrected"
    t0 = time.perf_counter()

    try:
        res = dmc_with_dt_extrapolation(
            a=a,
            b=b,
            c=c,
            n_walkers_per_run=n_walkers_per_run,
            n_runs_per_dt=n_runs_per_dt,
            dts=tuple(dts),
            base_n_steps=base_n_steps,
            n_cores=n_cores,
            verbose=verbose,
        )

        lambda_1 = float(res["lambda_1"])
        eigs = clean_eigenvalues(np.array([lambda_1]))

        return AlgorithmResult(
            algorithm=name,
            eigs=eigs,
            runtime_sec=time.perf_counter() - t0,
            success=True,
            meta={
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "lambda_1": lambda_1,
                "lambda_1_err": float(res["lambda_1_err"]),
                "slope": float(res["slope"]),
                "dts": [float(x) for x in res["dts"]],
                "means": [float(x) for x in res["means"]],
                "sems": [float(x) for x in res["sems"]],
                "n_walkers_per_run": int(n_walkers_per_run),
                "n_runs_per_dt": int(n_runs_per_dt),
                "base_n_steps": int(base_n_steps),
                "n_cores": int(n_cores),
                "method": "bridge-corrected diffusion Monte Carlo with dt-to-zero extrapolation",
            },
        )

    except Exception as e:
        return AlgorithmResult(
            algorithm=name,
            eigs=clean_eigenvalues([]),
            runtime_sec=time.perf_counter() - t0,
            success=False,
            message=repr(e),
            meta={
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "n_walkers_per_run": int(n_walkers_per_run),
                "n_runs_per_dt": int(n_runs_per_dt),
                "base_n_steps": int(base_n_steps),
                "n_cores": int(n_cores),
            },
        )