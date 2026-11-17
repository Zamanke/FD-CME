"""Full binary-input convergence sweep without changing production estimators.

For repeated binary inputs, kernel ridge regression reduces exactly to an
8-by-8 system. We check its observation weights against embeddings.py first.
Gaussian MMD is evaluated using deterministic Gauss-Hermite Fourier features,
checked against exact dense kernel calculations. This is not random features.
Population target terms use the known Gaussian-mixture intervention law.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DGPs import DGP_binary

FOLDER = ROOT/'diagnostics'
SIZES = [250, 500, 1000, 2000, 4000, 8000]
NAMES = ['Table Y: plug-in', 'Table Y: one-step', 'KRR Y: plug-in', 'KRR Y: one-step']
SITES = np.indices((2, 2, 2)).reshape(3, -1).T  # X, A, M
z, w = hermgauss(192)
positive = z > 0
OMEGA = np.sqrt(2)*z[positive]
SPECTRAL_WEIGHT = 2*w[positive]/np.sqrt(np.pi)
ROOT_WEIGHT = np.sqrt(SPECTRAL_WEIGHT)
MEANS = np.arange(3.)
p2 = 0.5/(1+np.exp(2))
MIXTURE = np.array([0.25, 0.75-p2, p2])


def features(y):
    phase = np.asarray(y).ravel()[:, None]*OMEGA
    return np.concatenate([np.cos(phase)*ROOT_WEIGHT,
                           np.sin(phase)*ROOT_WEIGHT], axis=1)


TARGET_FEATURE = MIXTURE @ features(MEANS) * np.tile(np.exp(-OMEGA**2/2), 2)


def cache_path(repetitions):
    return FOLDER/'.cache'/f'fixed_sweep_features_{repetitions}.npz'


def generator_hash():
    return hashlib.sha256((ROOT/'DGPs.py').read_bytes()).hexdigest()


def fit():
    """Use the unchanged hyperparameter learner on an independent sample."""
    from KRR_utils import learn_hyperparams
    start = time.time()
    values = learn_hyperparams(*DGP_binary(2000, seed=100000), subsample=False)
    result = dict(params=[float(p.item()) for p in values],
                  names=['lambda_m', 'lambda_y', 'sigma_A', 'sigma_M', 'sigma_X'],
                  n=2000, seed=100000, dtype='float64', seconds=time.time()-start)
    (FOLDER/'fixed_sweep_params.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def cell_ids(data):
    a, m, x = (np.asarray(v).astype(int).ravel() for v in data[:3])
    return 4*x+2*a+m


def cell_feature_means(ids, y):
    phi = features(y)
    out = np.zeros((8, phi.shape[1]))
    for s in range(8):
        if np.any(ids == s):
            out[s] = phi[ids == s].mean(axis=0)
    return out


def coefficients(counts_tr, counts_te, params, tabular_y, a0=1):
    """Return coefficients on the eight empirical outcome cell means.

    If K is the site Gram matrix and D the diagonal matrix of cell counts,
    KRR predictions have cell-mean weights D (K D + lambda I)^-1 K.
    These are exactly the dense observation-level KRR predictions.
    """
    if np.any(counts_tr == 0):
        raise ValueError('Empty training cell: unsmoothed one-step undefined')
    count = counts_tr.reshape(2, 2, 2)
    nxa = count.sum(axis=2)
    nx = nxa.sum(axis=1)
    pa = nxa/nx[:, None]
    pm = count/nxa[:, :, None]
    if tabular_y:
        g = np.eye(8)
    else:
        _, ridge_y, sigma_a, sigma_m, sigma_x = params
        inputs = SITES[:, [1, 2, 0]]
        delta = (inputs[:, None, :]-inputs[None, :, :])/np.array([sigma_a, sigma_m, sigma_x])
        kernel = np.exp(-0.5*np.sum(delta**2, axis=2))
        g = counts_tr[:, None]*np.linalg.solve(kernel*counts_tr[None, :]+ridge_y*np.eye(8), kernel)
    g = g.reshape(8, 2, 2, 2)  # basis, X, A, M
    plugin = np.zeros(8)
    eif = np.zeros(8)
    direct = np.zeros(8)
    for x in range(2):
        h = np.einsum('a,bam->bm', pa[x], g[:, x])
        v = np.einsum('m,bam->ba', pm[x, a0], g[:, x])
        eta = h @ pm[x, a0]
        plugin += nx[x]/counts_tr.sum()*eta
        for a in range(2):
            for m in range(2):
                s = 4*x+2*a+m
                rho = pm[x, a0, m]/pm[x, a, m]
                fraction = counts_te[s]/counts_te.sum()
                eif += fraction*(-rho*g[:, x, a, m]
                                 + (a == a0)/pa[x, a0]*(h[:, m]-eta) + v[:, a])
                direct[s] = fraction*rho
    return plugin, eif, direct


def validate():
    from embeddings import compute_plugin_a0, compute_EIF_a0
    # Different bandwidths, both treatments, table and KRR outcome embeddings.
    data = DGP_binary(400, seed=23)
    tr, te = tuple(v[:200] for v in data), tuple(v[200:] for v in data)
    ids_tr, ids_te = cell_ids(tr), cell_ids(te)
    counts_tr = np.bincount(ids_tr, minlength=8)
    counts_te = np.bincount(ids_te, minlength=8)
    params = [1., 0.7, 0.5, 1.2, 0.8]
    parameter_sets = [params]
    if (FOLDER/'fixed_sweep_params.json').exists():
        parameter_sets.append(json.loads((FOLDER/'fixed_sweep_params.json').read_text())['params'])
    max_weight_error = 0.
    max_mmd_error = 0.
    for parameters in parameter_sets:
        for a0 in [0, 1]:
            for tabular_y in [True, False]:
                pi, eif, direct = coefficients(counts_tr, counts_te, parameters, tabular_y, a0)
                p = tuple(jnp.array([v]) for v in parameters)
                actual_pi = np.asarray(compute_plugin_a0(*tr, a0=a0, params=p, x_method='table',
                                                         explicit_m=True, explicit_y=tabular_y)).ravel()
                actual_dr, actual_eif = compute_EIF_a0(*tr, *te, a0=a0, params=p, x_method='table',
                                                       mx_method='table', explicit_m=True, explicit_y=tabular_y)
                expected_pi = pi[ids_tr]/counts_tr[ids_tr]
                expected_eif = eif[ids_tr]/counts_tr[ids_tr]
                expected_dr = direct[ids_te]/counts_te[ids_te]
                for actual, expected in [(actual_pi, expected_pi), (np.asarray(actual_eif).ravel(), expected_eif),
                                         (np.asarray(actual_dr), expected_dr)]:
                    max_weight_error = max(max_weight_error, float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
                if a0 != 1:
                    continue
                y = np.asarray(data[3]).ravel()
                kernel = np.exp(-0.5*np.subtract.outer(y, y)**2)
                cross = np.exp(-(y[:, None]-MEANS)**2/4)/np.sqrt(2) @ MIXTURE
                target_norm = MIXTURE @ (np.exp(-np.subtract.outer(MEANS, MEANS)**2/6)/np.sqrt(3)) @ MIXTURE
                for weights in [np.r_[expected_pi, np.zeros(200)], np.r_[expected_eif, expected_dr]]:
                    exact = weights @ kernel @ weights - 2*weights @ cross + target_norm
                    delta = weights @ features(y)-TARGET_FEATURE
                    approx = delta @ delta
                    max_mmd_error = max(max_mmd_error, abs(exact-approx))
                    np.testing.assert_allclose(approx, exact, atol=1e-12, rtol=1e-9)
    distances = np.linspace(0, 20, 10001)
    kernel_error = float(np.max(np.abs(np.cos(distances[:, None]*OMEGA) @ SPECTRAL_WEIGHT
                                       - np.exp(-distances**2/2))))
    assert kernel_error < 1e-12
    y_grid = np.linspace(-10, 10, 2001)
    exact_cross = np.exp(-(y_grid[:, None]-MEANS)**2/4)/np.sqrt(2) @ MIXTURE
    cross_error = float(np.max(np.abs(features(y_grid) @ TARGET_FEATURE-exact_cross)))
    exact_norm = MIXTURE @ (np.exp(-np.subtract.outer(MEANS, MEANS)**2/6)/np.sqrt(3)) @ MIXTURE
    norm_error = float(abs(TARGET_FEATURE @ TARGET_FEATURE-exact_norm))
    assert max(cross_error, norm_error) < 1e-12
    result = dict(max_observation_weight_error=max_weight_error, max_squared_mmd_error=max_mmd_error,
                  kernel_grid_max_error=kernel_error, kernel_grid_max_distance=20,
                  target_cross_grid_max_error=cross_error, target_norm_error=norm_error,
                  parameter_sets_checked=len(parameter_sets))
    (FOLDER/'fixed_sweep_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def prepare(repetitions):
    """Cache cell summaries while the independent hyperparameter fit runs."""
    count_cache = np.zeros((len(SIZES), repetitions, 2, 8), dtype=int)
    feature_cache = np.zeros((len(SIZES), repetitions, 2, 8, len(TARGET_FEATURE)))
    max_outcome = 0.
    start = time.time()
    for i, n in enumerate(SIZES):
        for seed in range(repetitions):
            data = DGP_binary(2*n, seed=seed)
            max_outcome = max(max_outcome, float(np.max(np.abs(np.asarray(data[3])))))
            if max_outcome >= 10:
                raise ValueError('Observed Y exceeds the validated quadrature range')
            for fold, selection in enumerate([slice(None, n), slice(n, None)]):
                subset = tuple(v[selection] for v in data)
                ids = cell_ids(subset)
                count_cache[i, seed, fold] = np.bincount(ids, minlength=8)
                feature_cache[i, seed, fold] = cell_feature_means(ids, subset[3])
            if (seed+1) % 50 == 0:
                print(f'Prepared n={n}, {seed+1}/{repetitions}, elapsed={time.time()-start:.1f}s', flush=True)
        jax.clear_caches()
    destination = cache_path(repetitions)
    destination.parent.mkdir(exist_ok=True)
    np.savez(destination, counts=count_cache, features=feature_cache, max_abs_y=max_outcome,
             generator_hash=generator_hash(), n=SIZES)
    print(f'Cached cell summaries; max |Y|={max_outcome:.4f}', flush=True)


def run(repetitions):
    param_info = json.loads((FOLDER/'fixed_sweep_params.json').read_text())
    params = param_info['params']
    rows = []
    raw = np.full((len(SIZES), repetitions, 4), np.nan)
    max_outcome = 0.
    cached = None
    if cache_path(repetitions).exists():
        with np.load(cache_path(repetitions)) as archive:
            cached = {key: archive[key] for key in archive.files}
        assert cached['generator_hash'].item() == generator_hash(), 'DGP changed: prepare cache again'
        np.testing.assert_array_equal(cached['n'], SIZES)
        max_outcome = float(cached['max_abs_y'])
    start = time.time()
    for i, n in enumerate(SIZES):
        failures = []
        for seed in range(repetitions):
            if cached is None:
                data = DGP_binary(2*n, seed=seed)
                tr, te = tuple(v[:n] for v in data), tuple(v[n:] for v in data)
                ids_tr, ids_te = cell_ids(tr), cell_ids(te)
                counts_tr, counts_te = np.bincount(ids_tr, minlength=8), np.bincount(ids_te, minlength=8)
            else:
                counts_tr, counts_te = cached['counts'][i, seed]
            if np.any(counts_tr == 0):
                failures.append(seed)
                continue
            if cached is None:
                max_outcome = max(max_outcome, float(np.max(np.abs(np.asarray(data[3])))))
                if max_outcome >= 10:
                    raise ValueError('Observed Y exceeds the validated quadrature range')
                ftr, fte = cell_feature_means(ids_tr, tr[3]), cell_feature_means(ids_te, te[3])
            else:
                ftr, fte = cached['features'][i, seed]
            for j, tabular_y in enumerate([True, False]):
                pi, eif, direct = coefficients(counts_tr, counts_te, params, tabular_y)
                raw[i, seed, 2*j] = np.linalg.norm(pi @ ftr-TARGET_FEATURE)
                raw[i, seed, 2*j+1] = np.linalg.norm(eif @ ftr+direct @ fte-TARGET_FEATURE)
            if (seed+1) % 50 == 0:
                print(f'n={n}, {seed+1}/{repetitions}, elapsed={time.time()-start:.1f}s', flush=True)
        valid = raw[i, np.all(np.isfinite(raw[i]), axis=1)]
        row = dict(n=n, attempted=repetitions, valid=len(valid), empty_cell_seeds=failures,
                   mean=valid.mean(axis=0).tolist(), std=valid.std(axis=0, ddof=1).tolist(),
                   rmse=np.sqrt(np.mean(valid**2, axis=0)).tolist())
        rows.append(row)
        np.savez(FOLDER/'fixed_convergence_raw.npz', mmd=raw, n=np.array(SIZES))
        summary = dict(rows=rows, estimators=NAMES, params=param_info,
                       evaluation_bandwidth=1., reference='analytic population Gaussian mixture',
                       a0=1, u_scale=0, dtype='float64', quadrature_order=192,
                       max_abs_y=max_outcome, elapsed_seconds=time.time()-start)
        (FOLDER/'fixed_convergence_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        jax.clear_caches()


def plot(krr_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads((FOLDER/'fixed_convergence_summary.json').read_text())
    rows = data['rows']
    n = np.array([r['n'] for r in rows])
    rmse = np.array([r['rmse'] for r in rows])
    means = np.array([r['mean'] for r in rows])
    std = np.array([r['std'] for r in rows])
    valid = np.array([r['valid'] for r in rows])
    styles = [('#2864a0', '-', 'o'), ('#cf6b29', '-', 's'),
              ('#448f91', '--', '^'), ('#a25896', '--', 'D')]
    labels = data['estimators']
    prefix = 'fixed_convergence'
    if krr_only:
        rmse, means, std = rmse[:, 2:], means[:, 2:], std[:, 2:]
        styles = styles[2:]
        labels = ['Plug-in', 'One-step']
        prefix = 'fixed_krr_outcome'
    footer = ('Corrected binary-input DGP; continuous Y; table propensities and mediator; '
              'analytic population truth; Gaussian evaluation kernel with bandwidth 1')
    missing = ', '.join(f"n={r['n']}: {r['attempted']-r['valid']}" for r in rows if r['valid']<r['attempted']) or 'none'
    curve_group = 'both curves' if krr_only else 'all four curves'
    failure_note = f"{rows[0]['attempted']} repetitions per n. Undefined training-cell runs omitted from {curve_group}: {missing}."
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for j, (color, line, marker) in enumerate(styles):
        for ax, values in [(axes[0], rmse[:, j]), (axes[1], np.sqrt(n)*rmse[:, j])]:
            ax.plot(n, values, color=color, linestyle=line, marker=marker, markersize=5, linewidth=1.8, label=labels[j])
    axes[0].plot(n, rmse[-1, 0]*np.sqrt(n[-1]/n), ':', color='#555555', label=r'$n^{-1/2}$ reference', lw=1.6)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('RMSE of the mean embedding')
    axes[0].set_title('Error against the population target')
    axes[1].set_ylabel(r'$\sqrt{n}$ x RMSE')
    axes[1].set_title('A horizontal curve indicates root-n convergence')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(n, [str(v) for v in n])
        ax.set_xlabel('n per split')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    title = (r'KRR outcome stage: $\mathbb{E}[\phi(Y)\mid M,A,X]$' if krr_only
             else 'Convergence after fixing the simulation and estimator bugs')
    fig.suptitle(title, fontsize=16)
    fig.text(.5, .065, footer, ha='center', fontsize=9)
    fig.text(.5, .025, failure_note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    fig.savefig(FOLDER/(prefix+'_rmse.png'), dpi=180)
    fig.savefig(FOLDER/(prefix+'_rmse.svg'))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for j, (color, line, marker) in enumerate(styles):
        for ax, values in [(axes[0, 0], means[:, j]), (axes[0, 1], std[:, j]),
                           (axes[1, 0], np.sqrt(n)*means[:, j]), (axes[1, 1], np.sqrt(n)*std[:, j])]:
            ax.plot(n, values, color=color, linestyle=line, marker=marker, markersize=4, label=labels[j])
        # Monte Carlo standard error of the mean; this is not a data SD band.
        halfwidth = 1.96*std[:, j]/np.sqrt(valid)
        axes[0, 0].fill_between(n, means[:, j]-halfwidth, means[:, j]+halfwidth, color=color, alpha=.08)
        axes[1, 0].fill_between(n, np.sqrt(n)*(means[:, j]-halfwidth), np.sqrt(n)*(means[:, j]+halfwidth), color=color, alpha=.08)
    for ax, label in zip(axes.ravel(), ['Mean MMD', 'SD of MMD', r'$\sqrt{n}$ x mean MMD', r'$\sqrt{n}$ x SD of MMD']):
        ax.set_ylabel(label)
        ax.set_xlabel('n per split')
        ax.set_xscale('log', base=2)
        ax.set_xticks(n, [str(v) for v in n])
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('KRR outcome stage: mean and spread' if krr_only else
                 'Corrected convergence experiment: mean and spread', fontsize=16)
    fig.text(.5, .055, 'Shading: approximate 95% Monte Carlo intervals for mean MMD. n denotes each split size.', ha='center', fontsize=9)
    fig.text(.5, .025, failure_note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(FOLDER/(prefix+'_mean_sd.png'), dpi=180)
    fig.savefig(FOLDER/(prefix+'_mean_sd.svg'))
    plt.close(fig)
    slopes = {name: float(np.polyfit(np.log(n[n>=1000]), np.log(rmse[n>=1000, j]), 1)[0])
              for j, name in enumerate(labels)}
    print('Log-log RMSE slopes for n>=1000:', json.dumps(slopes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fit', 'validate', 'prepare', 'run', 'plot', 'plot_krr'])
    parser.add_argument('--repetitions', type=int, default=200)
    args = parser.parse_args()
    if args.mode in ['run', 'prepare']:
        globals()[args.mode](args.repetitions)
    elif args.mode == 'plot_krr':
        plot(krr_only=True)
    else:
        globals()[args.mode]()
