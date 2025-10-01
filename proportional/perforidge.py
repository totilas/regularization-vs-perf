# perfo_runs.py
import os
import numpy as np

import typer
from typer_config import use_yaml_config

app = typer.Typer()


@app.command()
@use_yaml_config()
def main(unbalanced:bool,
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

    if run_id is not None:
        grid_parameters = {
            "sigma": [.2, .5, 1],
            "kappa": [1.1, 2],
            "rho": [.5, .8],
            "gamma": [0, .3, .6, .9],
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
    

    return None

    p = int(n * kappa)
    assert p % 2 == 0
    if unbalanced:
        h = int(p * .8)    
    else:
        h = p//2
    g = p - h
        
    dvec = np.zeros(p, dtype=np.float64)
    dvec[: g] = b
    dvec[g :] = c
    
    # performative diagonal D: first half 0.2, rest 0.0 (apply with elementwise multiply)

    print(f"Hello sigma_{sigma}_kappa_{kappa}_rho_{rho}_{gamma}_{unbalanced}")


    sigma2 = sigma**2

    use_test = False         # True -> empirical test MSE; False -> analytic ||θ̂-θ*||^2 + σ^2
    n_test = 10**5
    test_chunk = 20000
    outdir = "results_perfo"
    os.makedirs(outdir, exist_ok=True)
    I_n = np.eye(n)
    lams = np.arange(lam_start, lam_stop, lam_step, dtype=np.float64)




    def sample_X(rng, rows, p, rho, gamma, unbalanced):
        if b == 0 and c== 0 and gamma == 0:
            return rng.standard_normal((rows, p))
        else:
            # TODO
            return rng.standard_normal((rows, p))
        # h = p // 2
        # Z1 = rng.standard_normal((rows, h))
        # Z2 = rng.standard_normal((rows, h))
        # X_left  = Z1
        # X_right = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        # return np.concatenate([X_left, X_right], axis=1)
    
    # TODO
    # elif covariance == "id":
    #     def sample_X(rng, rows, p, rho):
            
        
    # elif covariance == "unbalanced":
    #     def sample_X(rng, rows, p, rho):
    #         Z1 = rng.standard_normal((rows, g))
    #         Z2 = rng.standard_normal((rows, h))
    #         Xl = Z1
    #         Xr = np.empty((rows, h))
    #         Xr[:, :g] = rho * Z1 + np.sqrt(1 - rho**2) * Z2[:, :g]
    #         Xr[:, g:] = Z2[:, g:]
    #         return np.concatenate([Xl, Xr], axis=1) 
        
    # elif covariance == "toeplitz":
    #     def gauss_cov(rng, n, S):
    #         w, V = np.linalg.eigh(S)
    #         L = V * np.sqrt(np.clip(w, 0, None))
    #         Z = rng.standard_normal((n, p))
    #         return Z @ L.T

    #     def toeplitz_cov(p, rho):
    #         i = np.arange(p)
    #         return rho ** np.abs(i[:, None] - i[None, :])

    #     def sample_X(rng, rows, p, rho):
    #         return gauss_cov(rng, rows, toeplitz_cov(p, rho))


        
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

    def one_run(seed: int, rid: int):
        rng = np.random.default_rng(seed)

        # theta*
        v = rng.standard_normal(g); v /= np.linalg.norm(v)
        theta_star = np.zeros(p); theta_star[: g] = v

        # per-λ estimator carried across steps
        theta_prev = np.zeros((p, lams.size), dtype=np.float64)

        for s in range(1, steps + 1):
            # fresh correlated X, eps each step
            X = sample_X(rng, n, p, rho)
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

        np.savez(os.path.join(outdir, f"run_{name}_{rid}.npz"),
                lambdas=lams, risks=R, run=rid, n=n, p=p,
                sigma2=sigma2, steps=steps, rho=rho, gamma=gamma)

    for r in range(1, runs + 1):
        one_run(seed=1234 + r, rid=r)
        print(f"Saved {os.path.join(outdir, f'run_sigma_{sigma}_kappa_{kappa}_rho_{rho}_{gamma}_{unbalanced}_{r}.npz')}")

import itertools


if __name__ == "__main__":
    app()

# grid_params()