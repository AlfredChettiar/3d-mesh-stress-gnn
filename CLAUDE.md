# CLAUDE.md

Guidance for Claude Code when working in this repository. See `SPEC.md` for the
full project specification.

## Project

3D FEA mesh-to-stress Graph Neural Network surrogate: parametric bracket /
tubular-joint meshes are treated as node-edge graphs, and a GATv2 model predicts
per-node Von Mises stress and displacement orders of magnitude faster than FEA.

## Tech Stack

| Concern            | Choice                                              |
| ------------------ | --------------------------------------------------- |
| Language           | Python 3.14+ in a local virtualenv at `.venv/`      |
| Deep learning      | PyTorch                                             |
| Graph learning     | PyTorch Geometric (`torch_geometric.nn.GATv2Conv`)  |
| Numerics           | NumPy, SciPy (`scipy.spatial`, `scipy.sparse`)      |
| UI                 | Streamlit                                           |
| 3D visualisation   | Plotly 3D (`plotly.graph_objects.Mesh3d`)           |
| Testing            | Pytest                                              |

## Environment Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Always run tools through `.venv` (`.venv/bin/python`, `.venv/bin/pytest`).
- Pin dependencies in `requirements.txt`; do not install packages globally.
- If a dependency lacks Python 3.14 wheels, report it rather than silently
  downgrading the Python requirement.

## Commands

```bash
python -m src.mesh_generator --n-samples 1000 --out data/dataset.pt
python -m src.train --data data/dataset.pt --out runs/latest
streamlit run app.py
pytest -q
```

## Layout

```
app.py                 # Streamlit entry point (UI only, no model/physics logic)
src/
  __init__.py
  mesh_generator.py    # parametric geometry, graph build, BCs, ground truth
  physics.py           # analytical / semi-empirical stress & displacement
  gnn_model.py         # MLPBaseline, GATv2Surrogate
  train.py             # training, evaluation, benchmarking, metric logging
  viz.py               # Plotly figure builders (shared by app.py)
tests/                 # mirrors src/ (test_mesh_generator.py, ...)
data/  runs/           # generated artefacts, git-ignored
```

Rules:
- Keep modules small and single-purpose; `app.py` only imports from `src/`.
- No business logic in notebooks or `app.py`. Put it in `src/` and test it.
- Public functions get type hints and short docstrings that state **units**.
- Configuration goes in dataclasses or CLI args, never magic numbers inline.
- All randomness flows from an explicit seed (`numpy.random.Generator`,
  `torch.manual_seed`) so datasets and runs are reproducible.

## Physical Guardrails

Use SI units throughout (m, N, Pa, kg); convert only at the UI boundary
(display in mm / MPa).

Ground truth and predictions must satisfy:

- **Non-negativity:** Von Mises stress `σ_vm >= 0` and displacement magnitude
  `|u| >= 0` everywhere. Models output via a non-negative head (e.g. softplus)
  or predict in log space.
- **Finite values:** no NaN/Inf in features, targets or predictions; assert in
  dataset generation and in training.
- **Fixed supports:** nodes with a Dirichlet boundary condition have
  `u = 0` exactly (enforce by masking, not by learning).
- **Elastic regime:** generated loads are scaled so `max σ_vm` stays below the
  material yield strength (default steel: E = 210 GPa, ν = 0.30,
  σ_y = 250 MPa). Reject or rescale samples that violate it.
- **Linearity / scaling:** for linear elasticity, doubling the load doubles
  stress and displacement; doubling E halves displacement. Keep a pytest that
  checks this on the ground-truth generator.
- **Equilibrium sanity:** net applied load matches net reaction at supports
  within a stated tolerance.
- **Geometry validity:** positive dimensions, fillet radius < smaller adjacent
  feature, no degenerate (zero-area) elements, connected graph.
- **Stress concentration:** a Kt factor at steps/fillets/joints is bounded
  (typically 1 <= Kt <= 4); never produce a concentration lower than nominal.
- Normalise targets for training, but compute and report all metrics in
  physical units (MPa, mm) after de-normalising.
- Ground truth is analytical / semi-empirical, **not** a real FEA solve. Label
  it as such in code, UI and docs; never present it as validated FEA.

## Testing

- Pytest, under `tests/`, one test module per `src/` module.
- Cover: graph connectivity and shapes, BC masking, physical invariants above,
  determinism under a fixed seed, model forward shapes and non-negativity, a
  tiny overfit smoke test (few epochs, small data), and metric functions.
- Tests must be fast (< 60 s total) and CPU-only; use small meshes/datasets.
- Run `pytest -q` before every commit and fix failures rather than skipping.

## Code Style

- Formatting/linting with `ruff` (line length 100); type-check with `mypy` where
  practical.
- Prefer vectorised NumPy / tensor operations over Python loops over nodes.
- Seed and log every experiment; write metrics to `runs/<name>/metrics.json`.
- Do not commit data, checkpoints, `.venv/`, or `runs/`.

## Git Workflow

- Develop on a feature branch; open a Pull Request, do not push to `main`.
- Small focused commits with imperative messages.
