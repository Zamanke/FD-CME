"""Exercise the notebook loops cheaply, with distinctive fake estimator outputs."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_loop(notebook, cell, namespace):
    cells = json.loads((ROOT/notebook).read_text(encoding='utf8'))['cells']
    statements = ast.parse(''.join(cells[cell]['source'])).body
    loop = next(statement for statement in statements if isinstance(statement, ast.For))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=[loop], type_ignores=[]), notebook, 'exec'), namespace)


class NotebookLoops(unittest.TestCase):
    def test_conditional_reference_has_the_outcome_dimension(self):
        class Embedder:
            def train(self, data):
                self.n = data.shape[0]
            def predict(self, query):
                return jnp.ones((self.n, query.shape[0]))/self.n
        calls = []
        def check_mmd(outcomes, reference, weights):
            self.assertEqual(outcomes.shape[1], reference.shape[1])
            calls.append(reference.shape)
            return jnp.array(0.)
        cells = json.loads((ROOT/'nuisance_convergence.ipynb').read_text())['cells']
        ns = dict(jax=jax, jnp=jnp, Array=object, mu_y_amx=Embedder,
                  MMD_plugin=check_mmd,
                  DGP_binary=lambda n, seed: (jnp.zeros((n, 1)),)*4)
        exec(''.join(cells[7]['source']), ns)
        ns['embedding_convergence'](8)
        self.assertEqual(len(calls), 8)

    def test_plugin_results_are_preserved(self):
        shape = (2, 2, 51)
        ns = dict(jnp=jnp, n_list=[10, 20], num_seeds=51, truth_table=[True, False],
                  dr_mat=jnp.zeros(shape), pi_mat=jnp.zeros(shape), DGP_binary=None,
                  params=None, A_truth=None, M_truth=None, X_truth=None, Y_truth=None,
                  treatment=1, method='table')
        ns['run_convergence_experiment'] = lambda _, n, seed, **kw: (n+seed+1, 100+n+seed)
        run_loop('convergence_experiment.ipynb', 2, ns)
        expected = np.array([100+n+np.arange(51) for n in ns['n_list']])
        for j in range(2):
            np.testing.assert_array_equal(ns['pi_mat'][:, j], expected)

    def test_every_propensity_seed_runs(self):
        calls = []
        def estimate(_, n, seed, **kw):
            calls.append((n, seed))
            return jnp.ones((2, 2)), jnp.ones((4, 2))
        ns = dict(jnp=jnp, n_list=[10, 20], num_seeds=51, DGP_binary=None,
                  ax_mat=jnp.zeros((2, 51)), max_mat=jnp.zeros((2, 51)),
                  propensity_convergence=estimate)
        run_loop('nuisance_convergence.ipynb', 9, ns)
        self.assertEqual(len(calls), 102)
        self.assertTrue(np.all(np.asarray(ns['ax_mat']) > 0))

    def test_every_embedding_seed_runs(self):
        calls = []
        def estimate(n, seed):
            calls.append((n, seed))
            return jnp.ones(8)
        ns = dict(jnp=jnp, n_list=[10, 20], num_seeds=51,
                  mu_mat=jnp.zeros((2, 51, 8)), embedding_convergence=estimate)
        run_loop('nuisance_convergence.ipynb', 10, ns)
        self.assertEqual(len(calls), 102)
        self.assertTrue(np.all(np.asarray(ns['mu_mat']) == 1))


if __name__ == '__main__':
    unittest.main()
