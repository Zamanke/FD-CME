# Change inventory

Review snapshot against commit `4a31f54`, prepared 17 November 2026.
Read the [short tutorial](../CONVERGENCE_FIXES.md) for the reasoning behind
the fixes, and the [audit](README.md) for evidence and the E01-E13 issue IDs.

Counts are physical file lines, including notebook JSON and newline-only
changes. New files are counted in full. Refresh this snapshot if files change.

## Existing files

**Core fixes: five files, +61 / -86.** The main README additionally gains six
lines linking the review documentation.

| File | Added | Deleted | Main changes |
|---|---:|---:|---|
| [DGPs.py](../DGPs.py) | 46 | 71 | Independent keys for X, U, A, M and Y in all seven generators; correct binary-DGP description. (E01) |
| [embeddings.py](../embeddings.py) | 5 | 5 | Use the mediator kernel in four mediator-feature calls; add a final newline. (E03) |
| [convergence_experiment.ipynb](../convergence_experiment.ipynb) | 2 | 2 | Preserve plug-in results and remove the nonexistent plot import. (E02, E07) |
| [nuisance_convergence.ipynb](../nuisance_convergence.ipynb) | 6 | 6 | Run all seeds and pass a vector to the conditional reference generator. (E04, E05) |
| [utils.py](../utils.py) | 2 | 2 | Replace undefined Integer/Float annotations with int/Scalar. (E06) |
| [README.md](../README.md) | 6 | 0 | Link the short tutorial, this inventory and the detailed audit. |

## New code and documentation

| File | Added | Deleted | Purpose |
|---|---:|---:|---|
| [CONVERGENCE_FIXES.md](../CONVERGENCE_FIXES.md) | 105 | 0 | Short tutorial: failure mechanisms, fixes, prevention and validation. |
| [diagnostics/CHANGES.md](./CHANGES.md) | 61 | 0 | This review inventory. |
| [diagnostics/README.md](./README.md) | 299 | 0 | Complete error log, evidence, remaining issues and reproduction instructions. |
| [diagnostics/check_randomness.py](./check_randomness.py) | 55 | 0 | Reproduce original key reuse and check corrected structural-noise independence. |
| [diagnostics/convergence_check.py](./convergence_check.py) | 286 | 0 | Independent table calculations, estimator parity and convergence controls. |
| [diagnostics/fixed_convergence_sweep.py](./fixed_convergence_sweep.py) | 360 | 0 | Validated fast computation of the 200-repetition sweep and plots; does not replace production estimators. |
| [diagnostics/test_generators.py](./test_generators.py) | 29 | 0 | Check key use in every generator, under observation and intervention. |
| [diagnostics/test_notebooks.py](./test_notebooks.py) | 81 | 0 | Check output storage, all-seed execution and conditional-reference dimensions. |

## Generated artifacts

These are separate from the handwritten source changes. Existing historical
result arrays and figures are retained; corrected outputs have new filenames.

| Files under diagnostics/ | Count | Added lines | Deleted | Contents |
|---|---:|---:|---:|---|
| `.json` | 6 | 461 | 0 | Parameters, validation evidence and result summaries. |
| `.svg` | 4 | 11083 | 0 | Editable vector exports of four convergence figures. |
| `.png` | 5 | Binary | 0 | Five rendered convergence figures. |
| `.npz` | 1 | Binary | 0 | Individual MMD results from the full sweep. |

The local virtual environment and intermediate feature cache are ignored by
Git. No estimator imports or depends on the new diagnostic scripts.

## Suggested review order

1. Import fixes (`utils.py` and the convergence notebook import).
2. Random-key allocation (`DGPs.py`) and the key-reuse checks.
3. Plug-in result storage and nuisance-loop/reference fixes.
4. Mediator-kernel correction and estimator/table parity checks.
5. Diagnostic scripts, documentation and generated evidence separately.

The [outstanding-issues list](../CONVERGENCE_FIXES.md#review-and-reproduce)
identifies findings that this patch does not fix.
