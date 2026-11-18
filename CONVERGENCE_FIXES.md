# Convergence fixes: a short guide

The main cause of the binary experiment's error plateau was **random-key reuse
in the simulation**. It made supposedly independent structural noises dependent,
so the generated data did not satisfy the intended front-door assumptions.
Other errors corrupted the stored results and nuisance checks.

See the [file-by-file change inventory](diagnostics/CHANGES.md) for the review
scope, and the [full audit](diagnostics/README.md) for reproductions and all
13 findings. The fixes below are the changes to the estimator/experiment code;
the standalone diagnostics provide evidence for reviewing them.

## 1. Give each structural noise a fresh key

The old pattern sampled with `key`, then used `split(key)[0]` for the next
variable. JAX's binomial sampler had already used that child key internally.
In a minimal reproduction, the binary draw and subsequent Gaussian noise had
correlation about -0.80. For the binary DGP, the resulting dependence between
mediator and outcome noise invalidated the front-door identification argument.

All seven generators now allocate independent keys before sampling:

```python
key_x, key_u, key_a, key_m, key_y = jax.random.split(jax.random.key(seed), 5)
# Use each key only for its corresponding variable.
M = jax.random.binomial(key_m, 1, p_m)
noise_y = jax.random.normal(key_y, shape=M.shape)
```

The same assignments are used under intervention, even when sampling A is
skipped. **Prevention:** never sample with a key and then split that consumed
key for another draw. Use `jax.debug_key_reuse(True)` during tests, and check
the recovered structural noise conditional on the other variables.

## 2. Keep each experiment result in its own array

`pi_mat = dr_mat.at[...].set(pi)` copied the one-step array into the plug-in
results at every iteration. Only the final plug-in value survived. The fix is
`pi_mat = pi_mat.at[...].set(pi)`.

The nuisance loops also placed computation inside `if k % 50 == 0`; only four
of 200 seeds ran. Computation now sits outside that logging condition.
**Prevention:** test loops with cheap, distinct fake outputs; verify the number
of calls and every stored entry. Represent missing results explicitly, not as
successful zero errors. Existing corrupted arrays must be regenerated.

## 3. Match the kernel and the array shape to the variable

Four mediator feature evaluations used `A_kernel` instead of `M_kernel`.
They now use the mediator bandwidth. This matters when the bandwidths differ.
**Prevention:** compare KRR against the exact table solution at small ridge
penalty, deliberately using unequal A and M bandwidths.

The conditional-outcome reference received a `(1,3)` row instead of a `(3,)`
vector. Indexing and broadcasting silently produced three-dimensional reference
outcomes instead of one-dimensional outcomes. The caller now passes the vector.
**Prevention:** check query shapes and assert equal outcome dimensions before
evaluating kernels.

Two undefined type annotations and an unused import of a nonexistent `plot`
function were also repaired. Check imports in a fresh Python process, rather
than relying on notebook execution history.

## 4. Check convergence against a trustworthy target

The standalone rerun uses the corrected binary-input DGP, continuous Y,
`u_scale=0`, float64, a fixed Gaussian evaluation bandwidth of 1, and analytic
population truth. A fixed finite reference sample would impose an error floor.
These evaluation choices are explicit differences from the original notebook.

Across 200 repetitions at n=250 through 8000, the four RMSE slopes for n>=1000
are -0.514, -0.504, -0.498 and -0.498: consistent with root-n convergence for
plug-in/one-step and tabular/KRR outcome embeddings. Six empty-cell runs at
n=250 are explicitly excluded; all runs at larger n succeeded.

KRR still has only eight input states here, and the code's fixed penalty in
`K + c*I` corresponds to normalized regularization c/n. Root-n plug-in behaviour
is therefore expected. This is a correctness check, not evidence of an
efficiency advantage: compare equal total sample sizes and use cross-fitting
when assessing efficiency. The current one-step wrapper uses only one split
direction.

## Review and reproduce

From the repository root, using the Python environment described in the
[audit](diagnostics/README.md#reproduction):

```text
python -m unittest discover -s diagnostics -p "test_*.py" -v
python diagnostics/check_randomness.py
python diagnostics/convergence_check.py parity
python diagnostics/fixed_convergence_sweep.py validate
```

The five regression tests check all generator paths, result storage, seed
coverage and reference dimensions. The parity checks compare the estimators
with independent tables and the faster sweep with the dense implementation.
See the [full sweep instructions](diagnostics/README.md#full-corrected-binary-input-sweep)
to regenerate the plots.

**Still outstanding:** empty-cell handling in the main estimator; float32 MMD
cancellation; the older notebook's unsupported argument; treatment effects
remaining under `null=True` in three generators; a missing import in
`divergences.py`; and CV fold edge cases. These are recorded in the audit and
are not claimed fixed by this change.
