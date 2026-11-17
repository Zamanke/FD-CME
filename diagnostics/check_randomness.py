"""Reproduce the random-key failure and verify independent structural noise.

The baseline is pinned to the cloned commit; no files are changed by loading it.
"""
import json
from pathlib import Path
import subprocess
import sys
import types

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DGPs import DGP_binary

before = types.ModuleType('original_DGPs')
source = subprocess.check_output(
    ['git', 'show', '4a31f54:DGPs.py'], cwd=ROOT, text=True)
exec(source, before.__dict__)

results = {}
for name, generator in [('before', before.DGP_binary), ('after', DGP_binary)]:
    a, m, x, y = map(np.asarray, generator(50000, seed=7))
    noise = y-m-x  # u_scale=0, so this must be independent N(0,1).
    cells = []
    for xv in range(2):
        for av in range(2):
            for mv in range(2):
                selected = ((x == xv) & (a == av) & (m == mv)).ravel()
                cell = noise[selected]
                cells.append(dict(x=xv, a=av, m=mv, n=int(selected.sum()),
                                  noise_mean=float(cell.mean()),
                                  z=float(cell.mean()/(cell.std()/np.sqrt(cell.size)))))
    results[name] = cells
    max_z = max(abs(c['z']) for c in cells)
    print(name, 'max |conditional noise mean|', max(abs(c['noise_mean']) for c in cells),
          'max |z|', max_z)
    if name == 'after':
        assert max_z < 6, 'Outcome noise still depends on (X,A,M)'

with jax.debug_key_reuse(True):
    DGP_binary(100, seed=5)
print('Corrected generator passes JAX key-reuse checker')
try:
    with jax.debug_key_reuse(True):
        before.DGP_binary(100, seed=5)
except jax.errors.KeyReuseError:
    print('Original generator fails JAX key-reuse checker as expected')
else:
    raise AssertionError('Expected the original generator to reuse a key')

(ROOT/'diagnostics'/'randomness_check.json').write_text(json.dumps(results, indent=2)+'\n')
