# perfo_runs.py
import os
import numpy as np
import itertools
import typer
from typer_config import use_yaml_config

def _toeplitz_rect(g, h, gamma):
    i = np.arange(g)[:, None]
    j = np.arange(h)[None, :]
    return gamma ** np.abs(i - j)

def sample_X(rng, rows, rho, gamma,g,h):
    if np.isclose(rho, 0) and np.isclose(gamma,0):
        return rng.standard_normal((rows, g+h))
    Tg = _toeplitz_rect(g, g, gamma)
    Th = _toeplitz_rect(h, h, gamma)
    Tgh = _toeplitz_rect(g, h, gamma)
    S = np.block([[Tg, rho * Tgh], [rho * Tgh.T, Th]])
    L = np.linalg.cholesky(S)
    assert np.isfinite(L).all()

    Z = rng.standard_normal((rows, g + h))
    return Z @ L.T




app = typer.Typer()


@app.command()
@use_yaml_config()
def main(unbalanced:float,
        n: int,
        kappa: float,
        lam_start: float,
        lam_stop: float,
        lam_step: float,
        runs: int,
        steps: int,
        sigma: float,
        gamma: float,
        rho: float,
        b: float,
        c: float):
    
    # To use with slurm

    run_id = os.getenv("SLURM_ARRAY_TASK_ID", None)

    outdir = "results_perfo"



    param = {
        "sigma": sigma,
        "kappa": kappa,
        "rho": rho,
        "gamma": gamma,
        "unbalanced": unbalanced,
        "b": b,
        "c": c,
    }

    if run_id is not None:
        outdir = f"out/results_{os.getenv('SLURM_JOB_NAME', 'unkown')}_{os.getenv('SLURM_ARRAY_JOB_ID', 'unkown')}_{os.getenv('SLURM_PROCID', 'unkown')}"
        grid_parameters = {
            "sigma": [.2, .5, 1],
            "kappa": [1.1, 2],
            "rho": [0, .5],
            "gamma": [0, .5, .9],
            "unbalanced": [0.5, .8],
            "b": [0, 0.2],
            "c": [0, 0.2]
        }

        keys, values = zip(*sorted(grid_parameters.items()))
        params = []
        for point_values in itertools.product(*values):
            params.append({k: v for k, v in zip(keys, point_values)})

        run_id = int(run_id)

        param = params[run_id]

        sigma = param["sigma"]
        kappa = param["kappa"]
        rho = param["rho"]
        gamma = param["gamma"]
        unbalanced = param["unbalanced"]
        b = param["b"]
        c = param["c"]
        print(param)

    p = int(n * kappa)
    assert p % 2 == 0
    h = int(p * unbalanced)    
    g = p - h

        
    dvec = np.zeros(p, dtype=np.float64)
    dvec[: g] = b
    dvec[g :] = c
    
    # performative diagonal D: first half 0.2, rest 0.0 (apply with elementwise multiply)
    print(f"Hello sigma_{sigma}_kappa_{kappa}_rho_{rho}_gamma_{gamma}_u_{unbalanced}_b_{b}_c_{c}")


    sigma2 = sigma**2

    use_test = False         # True -> empirical test MSE; False -> analytic ||θ̂-θ*||^2 + σ^2
    n_test = 10**5
    test_chunk = 20000
    
    os.makedirs(outdir, exist_ok=True)
    I_n = np.eye(n)
    lams = np.arange(lam_start, lam_stop, lam_step, dtype=np.float64)

        
    def empirical_mse(theta_hat, seed):
        rng = np.random.default_rng(seed)  # same test set for every λ
        total = 0.0
        seen = 0
        for start in range(0, n_test, test_chunk):
            m = min(test_chunk, n_test - start)
            Xb = sample_X(rng, m, p, rho)                         # <-- correlated test features
            yb = Xb @ theta_hat_star + sigma * rng.standard_normal(m)  # untouched test y
            err = (Xb @ theta_hat) - yb
            total += float(err @ err)
            seen += m
        return total / seen

    def one_run(seed: int, rid: int, param):
        rng = np.random.default_rng(seed)

        # theta*
        v = rng.standard_normal(g); v /= np.linalg.norm(v)
        theta_star = np.zeros(p); theta_star[: g] = v

        # per-λ estimator carried across steps
        theta_prev = np.zeros((p, lams.size), dtype=np.float64)

        for s in range(1, steps + 1):
            # fresh correlated X, eps each step
            X = sample_X(rng, n, rho, gamma, g, h)
            eps = sigma * rng.standard_normal(n)
            y_base = X @ theta_star + eps

            G = (X @ X.T) / float(p)

            theta_new = np.empty_like(theta_prev)
            for i, lam in enumerate(lams):
                y = y_base if s == 1 else y_base + X @ (dvec * theta_prev[:, i])
                w = np.linalg.solve(G + lam * I_n, y / float(p))
                theta_new[:, i] = X.T @ w
            theta_prev = theta_new

        # final-step risks
        R = np.empty(lams.shape, dtype=np.float64)
        if use_test:
            seed_test = seed + 10**6
            # cache θ* for empirical_mse
            global theta_hat_star
            theta_hat_star = theta_star
            for i in range(lams.size):
                theta_hat = theta_prev[:, i]
                R[i] = empirical_mse(theta_hat, seed_test)
        else:
            for i in range(lams.size):
                d = theta_prev[:, i] - theta_star
                R[i] = (d @ d) + sigma2

        base = "_".join(f"{k}_{v}" for k, v in sorted(param.items()))
        filename = os.path.join(outdir, f"run_{base}_{rid}.npz")

        assert not os.path.exists(filename)
        np.savez(filename, lambdas=lams, risks=R, run=rid, sigma2=sigma2, steps=steps, n=n, p=p, param=param, **param)

        return filename


    for r in range(1, runs + 1):
        filename = one_run(seed=1234 + r, rid=r, param=param)
        print(f"Saved {filename}")




if __name__ == "__main__":
    app()

# grid_params()