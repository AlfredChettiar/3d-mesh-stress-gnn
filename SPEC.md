# SPEC: 3D FEA Mesh-to-Stress GNN Surrogate

## 1. Goal

Build a surrogate that predicts per-node **Von Mises stress** and **displacement**
on 3D meshes of mechanical parts from the mesh graph, geometry, material, loads
and boundary conditions, then compare it against a non-graph baseline and show
the results interactively.

Success criteria (on a held-out test split, in physical units):

| Metric                              | Target                                   |
| ----------------------------------- | ---------------------------------------- |
| GATv2 Von Mises R²                  | >= 0.90                                  |
| GATv2 beats MLP baseline            | lower MAE and higher R² on stress        |
| Inference speedup vs. ground truth  | reported in ms/sample and x-factor       |
| Physical validity of predictions    | 0 NaN/Inf, no negative stress, u=0 at fixed supports |

Ground truth is **analytical / semi-empirical**, not a real FEA solve (see §2.4).
All docs and UI must say so.

## 2. Data Generation: `src/mesh_generator.py`

### 2.1 Geometries (parametric)

Two families, ~50/50 mix, 1,000 samples total (configurable `--n-samples`, `--seed`).

1. **Stepped bracket**: two or three rectangular blocks of different cross
   sections joined in series (cantilever-like), optional step fillet.
   Parameters: lengths `L1..L3`, widths `w1..w3`, thicknesses `t1..t3`,
   fillet radius `r`.
2. **Tubular joint**: a main tube (chord) with a perpendicular branch (brace)
   tube. Parameters: chord outer diameter `D`, wall thickness `T`, chord length
   `Lc`, brace diameter `d`, brace wall `t`, brace length `Lb`, brace angle
   (90° default, 45–90° allowed).

Parameter ranges live in a `GeometryConfig` dataclass and obey the geometry
guardrails in `CLAUDE.md`.

### 2.2 Graph construction

- Structured hexahedral-style node grid per part, giving 300–1,500 nodes per sample.
- Edges: element connectivity (grid neighbours + face diagonals), made
  bidirectional, with no duplicates or self loops (`scipy.spatial` / `scipy.sparse`
  may be used).
- Node features `x` (float32):
  - position `(x, y, z)` normalised by characteristic length
  - fixed-support flag, loaded-node flag
  - applied nodal force vector `(Fx, Fy, Fz)` normalised by max load
  - local section dims (e.g. thickness / diameter) and distance to nearest
    step / fillet / joint junction
  - global scalars broadcast to nodes: `E`, `ν`, load magnitude
- Edge features `edge_attr`: relative vector `(dx, dy, dz)`, length.
- Targets `y` (float32, physical units): `[sigma_vm, u_x, u_y, u_z]` (Pa, m);
  `|u|` is derived.
- Boundary conditions: one face/end fully fixed; one or more loaded regions
  (axial, transverse bending, torsion or mixed), random magnitude and direction
  within a configured range.

### 2.3 Output

Dataset saved as a list of `torch_geometric.data.Data` objects
(`data/dataset.pt`) with a deterministic train/val/test split (70/15/15) stored
as index arrays, and normalisation statistics computed from the train split only.

### 2.4 Ground truth (analytical / semi-empirical)

Per node, from beam / thin-wall theory plus concentration factors:

- Nominal axial `σ = F/A`, bending `σ = M·c/I`, torsional shear `τ = T·r/J`
  and transverse shear, combined to Von Mises
  `σ_vm = sqrt(σ² + 3τ²)` (or the full 3D form when all components exist).
- Stress concentration at steps/fillets and tube junctions via empirical Kt
  (Peterson-style for stepped shafts/plates; Kt vs. d/D and branch/chord
  ratio for tubular joints), decaying with distance from the discontinuity.
- Displacement by Euler–Bernoulli / Timoshenko beam integration along the
  member axis (axial, bending deflection, twist), mapped to nodes; `u = 0`
  at the fixed support.
- Small bounded noise (< 2%) may be added to avoid a purely closed-form
  mapping; it must be seeded and documented.
- Loads are rescaled so `max σ_vm <= 0.9 σ_y`.

Implemented in `src/physics.py` with unit tests for each formula against hand
calculations.

## 3. Models and Training: `src/gnn_model.py`, `src/train.py`

### 3.1 Models (`gnn_model.py`)

- **`MLPBaseline`**: node-wise MLP with no message passing (inputs: node
  features only), 3–4 layers, hidden 128, ReLU/GELU, LayerNorm.
- **`GATv2Surrogate`**: linear encoder → N (default 4) `GATv2Conv` layers
  (heads=4, `edge_dim` set for edge features, residual connections, LayerNorm,
  dropout) → MLP decoder producing 4 outputs.
- Both share the same input/output interface and a non-negative stress head
  (softplus); displacement outputs are masked to 0 at fixed-support nodes.
- Parameter counts of the two models are logged and kept comparable.

### 3.2 Training (`train.py`)

- Loss: weighted Huber/MSE on normalised stress and displacement.
- AdamW, cosine or ReduceLROnPlateau schedule, early stopping on val loss.
- PyG `DataLoader`, fixed seeds, CPU or CUDA auto-select.
- CLI: `python -m src.train --data ... --model {mlp,gat,both} --epochs ... --out runs/<name>`.

### 3.3 Benchmark and logging

Evaluated on the test split in physical units (MPa, mm), for each model:

- **R²** and **MAE** for Von Mises stress and for displacement magnitude;
  also max-stress (peak) error, as peaks drive design decisions.
- **Latency**: mean ms per sample for model inference (warm-up excluded,
  batch size 1, averaged over the test set) versus ground-truth generator
  time per sample, and **speedup = t_truth / t_model**.
- Parameter count and training time.

Results written to `runs/<name>/metrics.json` and `runs/<name>/benchmark.md`
(table), plus saved checkpoints `mlp.pt`, `gat.pt`, and per-sample test
predictions for the app. Metrics are also printed to the console.

## 4. Interactive Viewer: `app.py`

Streamlit app (`streamlit run app.py`) with Plotly 3D. It loads the dataset and
checkpoints from `data/` and `runs/`, and shows a friendly message if they are missing.

- **Sidebar:** model choice (GATv2 / MLP), test-sample selector (index and
  "worst-error sample" shortcut), field choice (Von Mises stress / displacement magnitude),
  colour scale, optional deformation scale slider (visual only, labelled),
  show/hide edges, show BC markers (fixed nodes, load arrows).
- **Main view, three synchronised panels** (shared colour range so panels are comparable):
  1. Ground Truth (analytical FEA surrogate target)
  2. GNN Predicted
  3. Absolute Error map (`|pred − truth|`, own colour bar) with optionally
     percent-error toggle
  Rendered with `Mesh3d` (surface triangulation, intensity per vertex) or
  `Scatter3d` for nodes; camera linked between panels where supported.
- **Metrics strip:** per-sample MAE, max stress truth vs. predicted, peak
  error %, and global test R²/MAE, plus inference time and speedup from the benchmark.
- **Hover:** node id, position (mm), values in MPa / mm.
- A visible disclaimer that ground truth is semi-empirical, not a full FEA solve.
- Plot-building code lives in `src/viz.py` and is unit tested without Streamlit.

## 5. Repository Deliverables

```
CLAUDE.md  SPEC.md  README.md  requirements.txt  .gitignore  app.py
src/{__init__,mesh_generator,physics,gnn_model,train,viz}.py
tests/{test_mesh_generator,test_physics,test_gnn_model,test_train,test_viz}.py
```

## 6. Milestones

1. Scaffolding: `.venv`, `requirements.txt`, package layout, CI-ready pytest.
2. `physics.py` + `mesh_generator.py` with tests; generate 1,000-sample dataset.
3. `gnn_model.py` + `train.py`; baseline vs. GATv2 benchmark with logged metrics.
4. `viz.py` + `app.py`; manual check of all three panels.
5. README with results table and screenshots.

## 7. Out of Scope

- Real FEA solves (Abaqus, CalculiX, FEniCS) and mesh import from CAD/STEP.
- Nonlinear materials, contact, dynamics, plasticity.
- Production deployment, auth, and multi-user hosting.

## 8. Acceptance Checklist

- [ ] 1,000 samples generate deterministically from a seed, with all physical guardrails passing.
- [ ] GATv2 and MLP both train from the CLI; metrics (R², MAE, ms, speedup) are logged.
- [ ] GATv2 outperforms the MLP baseline on stress MAE and R².
- [ ] Streamlit app shows Ground Truth, Predicted and Error maps in 3D.
- [ ] `pytest -q` passes on CPU in under 60 s.
