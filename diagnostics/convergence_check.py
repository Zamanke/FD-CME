"""Small, independent probability-table checks for the convergence experiment.

Run from the repository root with the local virtual environment:
    python diagnostics/convergence_check.py audit
    python diagnostics/convergence_check.py parity
    python diagnostics/convergence_check.py rates
    python diagnostics/convergence_check.py continuous
    python diagnostics/convergence_check.py plot

The rates check bins the current binary DGP's continuous Y at 0.5 and 1.5.
It uses exact population probabilities, a fixed Gaussian kernel on {0,1,2},
and multinomial samples. No fitted hyperparameters or Monte Carlo truth.
"""

import argparse
import json
from math import erf, exp, sqrt
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def expit(t):
    return 1 / (1 + exp(-t))


def ndtr(t):
    return (1 + erf(t / sqrt(2))) / 2


def population(a0=1):
    """Joint P(X,A,M,Y_bin) and exact front-door intervention distribution."""
    joint = np.zeros((2, 2, 2, 3))
    truth = np.zeros(3)
    for x in range(2):
        for a in range(2):
            pa1 = expit(-(2*x-1))
            pm1 = expit(-(2*a+2*x-2))
            for m in range(2):
                py = np.diff([0, ndtr(0.5-m-x), ndtr(1.5-m-x), 1])
                joint[x, a, m] = 0.5 * (pa1 if a else 1-pa1) * (pm1 if m else 1-pm1) * py
                if a == a0:
                    truth += 0.5 * (pm1 if m else 1-pm1) * py
    return joint, truth


def table_estimates(train, test, a0=1):
    """Direct sums over probability tables; no embedding implementation used."""
    nxam = train.sum(axis=3)
    nxa = nxam.sum(axis=2)
    nx = nxa.sum(axis=1)
    if np.any(nxam == 0):
        raise ValueError('Empty training (X,A,M) cell; unsmoothed estimate undefined')
    pa = nxa / nx[:, None]
    pm = nxam / nxa[:, :, None]
    g = train / nxam[:, :, :, None]
    plugin = np.zeros(3)
    one_step = np.zeros(3)
    eye = np.eye(3)
    for x in range(2):
        h = np.einsum('a,amy->my', pa[x], g[x])
        v = np.einsum('m,amy->ay', pm[x, a0], g[x])
        eta = np.einsum('m,my->y', pm[x, a0], h)
        plugin += nx[x] / train.sum() * eta
        for a in range(2):
            for m in range(2):
                influence = (
                    pm[x, a0, m] / pm[x, a, m] * (eye - g[x, a, m])
                    + (a == a0) / pa[x, a0] * (h[m] - eta)
                    + v[a]
                )
                one_step += test[x, a, m] @ influence / test.sum()
    return plugin, one_step


def expand(counts):
    """Convert counts back into arrays accepted by the repository."""
    cells = np.indices(counts.shape).reshape(4, -1).T
    rows = np.repeat(cells, counts.ravel().astype(int), axis=0).astype(float)
    return tuple(rows[:, [i]] for i in [1, 2, 0, 3])  # A, M, X, Y


def audit():
    for name in ['binary_tabular', 'binary_tabular_v2']:
        with (ROOT / 'results' / (name + '.npy')).open('rb') as f:
            dr, pi = np.load(f), np.load(f)
        same = (dr == pi) | (np.isnan(dr) & np.isnan(pi))
        print(name, 'matching entries:', int(same.sum()), '/', same.size,
              'different indices:', np.argwhere(~same).tolist())
    for name in ['propensity_convergence', 'embedding_convergence']:
        with (ROOT / 'results' / (name + '.npy')).open('rb') as f:
            arr = np.load(f)
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        print(name, 'nonzero seeds:', np.flatnonzero(np.any(arr != 0, axis=axes)).tolist(),
              'zero fraction:', float(np.mean(arr == 0)))


def parity():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from embeddings import compute_plugin_a0, compute_EIF_a0, MMD_plugin, MMD_dr
    from KRR_utils import ard_median_heuristic

    joint, _ = population()
    rng = np.random.default_rng(452)
    # Guarantee every cell is represented in this algebra check.
    train = rng.multinomial(180, joint.ravel()).reshape(joint.shape) + 1
    test = rng.multinomial(210, joint.ravel()).reshape(joint.shape) + 1
    tr, te = expand(train), expand(test)
    params = tuple(jnp.array([v]) for v in [1e-5, 1e-5, 0.5, 1.2, 0.8])
    train_features = np.eye(3)[tr[3].astype(int).ravel()]
    test_features = np.eye(3)[te[3].astype(int).ravel()]
    for a0 in [0, 1]:
        expected_pi, expected_dr = table_estimates(train, test, a0)
        pi = compute_plugin_a0(*tr, a0=a0, x_method='table', params=params,
                               explicit_m=True, explicit_y=True)
        dr, eif = compute_EIF_a0(*tr, *te, a0=a0, x_method='table', mx_method='table',
                               params=params, explicit_m=True, explicit_y=True)
        actual_pi = np.asarray(pi).ravel() @ train_features
        actual_dr = np.asarray(eif).ravel() @ train_features + np.asarray(dr) @ test_features
        np.testing.assert_allclose(actual_pi, expected_pi, atol=1e-10)
        np.testing.assert_allclose(actual_dr, expected_dr, atol=1e-10)
        print('a0', a0, 'tabular plug-in/one-step parity max error:',
              max(np.max(np.abs(actual_pi-expected_pi)), np.max(np.abs(actual_dr-expected_dr))))

        # Verify the MMD implementation using a separate quadratic form.
        sigma = float(np.asarray(ard_median_heuristic(tr[3])).item())
        kernel = np.exp(-0.5 * (np.subtract.outer(np.arange(3), np.arange(3))/sigma)**2)
        truth_y = np.array([[0.], [1.], [1.], [2.]])
        truth_p = np.array([0.25, 0.5, 0.25])
        for actual, measured in [
            (actual_pi, MMD_plugin(tr[3], truth_y, pi)),
            (actual_dr, MMD_dr(te[3], tr[3], truth_y, eif, dr)),
        ]:
            error = actual-truth_p
            np.testing.assert_allclose(float(measured), error @ kernel @ error, atol=1e-10)

        # As ridge -> 0 on finite inputs, KRR with tabular M should agree
        # with tabular Y. Different A and M bandwidths expose a kernel mixup.
        pi_krr = compute_plugin_a0(*tr, a0=a0, x_method='table', params=params,
                                   explicit_m=True, explicit_y=False)
        dr_krr, eif_krr = compute_EIF_a0(*tr, *te, a0=a0, x_method='table',
                                       mx_method='table', params=params,
                                       explicit_m=True, explicit_y=False)
        krr_pi = np.asarray(pi_krr).ravel() @ train_features
        krr_dr = np.asarray(eif_krr).ravel() @ train_features + np.asarray(dr_krr) @ test_features
        print('a0', a0, 'small-ridge KRR vs table errors:',
              float(np.max(np.abs(krr_pi-actual_pi))), float(np.max(np.abs(krr_dr-actual_dr))),
              'plug-in total mass:', float(krr_pi.sum()))
        np.testing.assert_allclose(krr_pi, actual_pi, atol=1e-5)
        np.testing.assert_allclose(krr_dr, actual_dr, atol=1e-5)


def rates(repetitions):
    joint, truth = population()
    kernel = np.exp(-0.5 * np.subtract.outer(np.arange(3), np.arange(3))**2)
    sizes = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    rng = np.random.default_rng(82711)
    rows = []
    for n in sizes:
        squared_errors = []
        failures = 0
        for _ in range(repetitions):
            train = rng.multinomial(n, joint.ravel()).reshape(joint.shape)
            test = rng.multinomial(n, joint.ravel()).reshape(joint.shape)
            try:
                estimates = table_estimates(train, test)
            except ValueError:
                failures += 1
                continue
            squared_errors.append([(p-truth) @ kernel @ (p-truth) for p in estimates])
        rmse = np.sqrt(np.mean(squared_errors, axis=0))
        row = dict(n=n, repetitions=repetitions, empty_cell_failures=failures,
                   plugin_rmse=float(rmse[0]), one_step_rmse=float(rmse[1]),
                   plugin_scaled=float(np.sqrt(n)*rmse[0]),
                   one_step_scaled=float(np.sqrt(n)*rmse[1]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Fit only sizes >= 1000, and expose failures rather than treating them as zero.
    slopes = {key: float(np.polyfit(np.log(sizes[2:]),
                                   np.log([r[key] for r in rows[2:]]), 1)[0])
              for key in ['plugin_rmse', 'one_step_rmse']}
    print('log-log slopes:', slopes)
    output = ROOT / 'diagnostics' / 'tabular_rates.json'
    output.write_text(json.dumps(dict(rows=rows, slopes=slopes), indent=2) + '\n')


def continuous():
    """Small fresh run of the actual DGP and estimator, with analytic truth.

    For the default DGP (u_scale=0, a0=1), Y^1 is a mixture of three
    unit-variance Gaussians. Both target terms of Gaussian MMD are analytic.
    The evaluation bandwidth is fixed at 1 across runs.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from DGPs import DGP_binary
    from embeddings import compute_plugin_a0, compute_EIF_a0

    means = np.arange(3.)
    p2 = 0.5 * expit(-2)
    probs = np.array([0.25, 0.75-p2, p2])
    target_norm = probs @ (np.exp(-np.subtract.outer(means, means)**2/6)/sqrt(3)) @ probs
    params = tuple(jnp.array([v]) for v in [1., 1., 0.5, 1.2, 0.8])
    rows = []
    for n in [250, 500, 1000]:
        errors = []
        failures = 0
        for seed in range(8):
            data = DGP_binary(2*n, seed=seed)
            tr, te = tuple(z[:n] for z in data), tuple(z[n:] for z in data)
            a, m, x = (np.asarray(z).astype(int).ravel() for z in tr[:3])
            if np.any(np.bincount(4*x+2*a+m, minlength=8) == 0):
                failures += 1
                continue
            pi = compute_plugin_a0(*tr, params=params, x_method='table', explicit_y=True)
            dr, eif = compute_EIF_a0(*tr, *te, params=params, x_method='table',
                                   mx_method='table', explicit_y=True)
            y = np.asarray(data[3]).ravel()
            weights = np.column_stack([
                np.r_[np.asarray(pi).ravel(), np.zeros(n)],
                np.r_[np.asarray(eif).ravel(), np.asarray(dr).ravel()],
            ])
            kernel = np.exp(-0.5*np.subtract.outer(y, y)**2)
            target_cross = np.exp(-(y[:, None]-means)**2/4)/sqrt(2) @ probs
            squared = np.sum(weights*(kernel @ weights), axis=0) - 2*weights.T@target_cross + target_norm
            if not np.all(np.isfinite(squared)) or np.any(squared < -1e-12):
                raise ValueError(f'Invalid squared MMD at n={n}, seed={seed}: {squared}')
            errors.append(np.sqrt(np.maximum(squared, 0)))
        rmse = np.sqrt(np.mean(np.array(errors)**2, axis=0))
        row = dict(n=n, seeds=8, empty_cell_failures=failures,
                   plugin_rmse=float(rmse[0]), one_step_rmse=float(rmse[1]),
                   plugin_scaled=float(sqrt(n)*rmse[0]), one_step_scaled=float(sqrt(n)*rmse[1]))
        rows.append(row)
        print(json.dumps(row), flush=True)
        jax.clear_caches()
    (ROOT/'diagnostics'/'continuous_smoke.json').write_text(json.dumps(rows, indent=2)+'\n')


def plot():
    """Plot only sample sizes with no empty-cell failures in the rate run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT/'diagnostics'
    rows = json.loads((folder/'tabular_rates.json').read_text())['rows']
    rows = [r for r in rows if r['n'] >= 1000 and r['empty_cell_failures'] == 0]
    n = np.array([r['n'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for method, label, color in [('plugin', 'Plug-in', '#2465a9'),
                                  ('one_step', 'One-step', '#b44831')]:
        error = np.array([r[method+'_rmse'] for r in rows])
        axes[0].loglog(n, error, 'o-', color=color, label=label, lw=2)
        axes[1].semilogx(n, np.sqrt(n)*error, 'o-', color=color, label=label, lw=2)
    axes[0].loglog(n, 0.023*np.sqrt(1000/n), '--', color='#777777', label='n$^{-1/2}$ reference')
    axes[0].set_ylabel('RMSE in the RKHS norm')
    axes[1].set_ylabel(r'$\sqrt{n}$ x RMSE')
    axes[1].set_ylim(0.65, 0.8)
    for axis in axes:
        axis.set_xlabel('n per split')
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Finite-table control recovers root-n convergence', fontsize=15)
    fig.text(0.5, 0.015, f'Exact population truth | fixed Gaussian kernel on binned Y | {rows[0]["repetitions"]:,} repetitions per n',
             ha='center', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(folder/'tabular_convergence.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['audit', 'parity', 'rates', 'continuous', 'plot'])
    parser.add_argument('--repetitions', type=int, default=2000)
    args = parser.parse_args()
    if args.check == 'rates':
        rates(args.repetitions)
    else:
        globals()[args.check]()
