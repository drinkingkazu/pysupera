# pysupera

<p align="center">
  <img src="figures/supera.png" alt="pysupera logo" width="320"/>
</p>

[![Tests](https://github.com/drinkingkazu/pysupera/actions/workflows/tests.yml/badge.svg)](https://github.com/drinkingkazu/pysupera/actions/workflows/tests.yml)

A Python library for grouping simulated LArTPC particles into physics partitions.
`pysupera` takes a list of reconstructed particles—each carrying a 3-D point cloud of detector hits—and merges spatially touching particles that satisfy configurable physical criteria into partition groups.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
   - [Python API](#python-api)
   - [Interactive Event Viewer (`pysupera-app`)](#interactive-event-viewer-pysupera-app)
5. [Workflow](#workflow)
   - [Input: the `Particle` object](#input-the-particle-object)
   - [Step 0 — Configuration](#step-0--configuration)
   - [Step 1 — Preprocessing](#step-1--preprocessing)
   - [Step 2 — Applying Conditions](#step-2--applying-conditions)
   - [Step 3 — Generating Partitions](#step-3--generating-partitions)
   - [Step 4 — Output](#step-4--output)
6. [CLI Usage](#cli-usage)
7. [Compute Backends](#compute-backends)
8. [Algorithms](#algorithms)
   - [Partition-level Incremental Algorithm](#partition-level-incremental-algorithm)
   - [Proximity Check](#proximity-check)
   - [Condition Pipeline](#condition-pipeline)
9. [Conditions Reference](#conditions-reference)
10. [Diagnostics](#diagnostics)
11. [Assumptions](#assumptions)

---

## Introduction

In liquid-argon time-projection-chamber (LArTPC) simulations, each simulated particle (track, shower fragment, low-energy scatter, etc.) is associated with a set of 3-D space points.  Reconstruction algorithms typically produce one particle per connected sub-graph of the ionisation pattern, but physics processes—electromagnetic showers, low-energy scatters, secondary vertices—systematically fragment what is logically a single object into many reconstructed pieces.

`pysupera` reassembles those fragments into *partitions* by:

1. **Preprocessing** fragmented point clouds (optional).
2. **Selecting candidate pairs** using physical metadata (PDG code, semantic type, parent–child ancestry).
3. **Testing spatial proximity** between the merged point clouds of candidate partitions.
4. **Merging** touching candidates subject to condition-specific post-filters.

The result is a list of partitions, where each partition is a list of `Particle` objects that belong together physically.

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `numpy` | ✅ | Array operations throughout |
| `scipy` | ✅ | KDTree proximity checks (CPU backends), connected-components defragmentation |
| `h5py` | ✅ | HDF5 I/O (`EventStore`, `EventWriter`) |
| `hydra-core ≥ 1.3` | ✅ | Hierarchical configuration (CLI and programmatic) |
| `omegaconf ≥ 2.3` | ✅ | Config composition (dependency of Hydra) |
| `joblib` | optional | Parallel KDTree queries (`cpu-multi` backend) |
| `cupy` | optional | GPU pairwise distances (`bulk-gpu`, `gpu` defragmentation) |
| `cuml` (RAPIDS) | optional | GPU nearest-neighbour index (`gpu` backend) |
| `cudf`, `cugraph` (RAPIDS) | optional | Fully GPU defragmentation (`rapids` preprocessor) |
| `numba` | optional | CUDA kernel proximity checker (`numba` backend) |

---

## Installation

```bash
# Clone or navigate to the pysupera repo root
cd /path/to/pysupera

# Editable install (recommended for development)
pip install -e .

# With GPU extras
pip install -e ".[gpu]"

# With Numba extras
pip install -e ".[numba]"

# With app (Dash interactive event viewer)
pip install -e ".[app]"

# With all dev tools
pip install -e ".[dev]"
```

---

## Quick Start

### Python API

```python
from pysupera import read_events
from pysupera.partitioner import ParticlePartitioner
from pysupera.conditions import PhotonDecay, TouchingEMShower, CombineLEScatters, AbsorbLEScatter

# Load particles for one event
with read_events("my_file.h5") as store:
    particles = list(store.iter_events())[0]

# Build the partitioner
partitioner = ParticlePartitioner(
    particles          = particles,
    distance_threshold = 5.0,   # mm (same units as point cloud coordinates)
    backend            = "cpu-single",
)

# Apply all conditions in one convergence loop
partitions = partitioner.partition_combined([
    PhotonDecay(),
    TouchingEMShower(),
    CombineLEScatters(),
    AbsorbLEScatter(),
])

print(f"{len(partitions)} partitions from {len(particles)} particles")
```

---

### Interactive Event Viewer (`pysupera-app`)

Install the app dependencies:

```bash
pip install -e ".[app]"
```

Launch the viewer:

```bash
pysupera-app                        # default: http://localhost:8050
pysupera-app --port 8080            # custom port
pysupera-app --host 0.0.0.0         # expose on the network
pysupera-app --debug                # enable Dash hot-reload
```

Then open the URL in a browser.  The sidebar lets you:

| Section | Controls |
|---|---|
| **INPUT** | HDF5 file path and event index |
| **PREPROCESSING** | Enable merge-duplicates and/or defragmentation; backend; semantic-type filter |
| **PARTITIONER** | Distance threshold, checker backend, `n_jobs` |
| **CONDITIONS** | Toggle each of the four conditions independently |
| **VIEW OPTIONS** | Show/hide legends; synchronise the two 3-D camera views |
| **PARTICLE FILTER** | Instantly show/hide particles by semantic type (no re-run needed) |

Hit **▶ Run** to execute the full pipeline and render two side-by-side 3-D point-cloud plots—original particles on the left, partitions on the right.  The **PARTICLE FILTER** and **show legend** checkbox take effect immediately without re-running the pipeline.

<p align="center">
  <img src="figures/dash_display.png" alt="pysupera-app event display" width="900"/>
</p>

---

## Workflow

### Input: the `Particle` object

#### Required attributes

These attributes must always be supplied at construction time and are guaranteed to be set on every `Particle` instance.

| Attribute | Type | Description |
|---|---|---|
| `id` | `int` | Unique particle ID within an event (Geant4 track ID) |
| `parent_id` | `int` | Direct parent particle ID; equals `id` for primary particles |
| `root_id` | `int` | ID of the primary ancestor at the root of the shower/track genealogy |
| `pdg` | `int` | PDG Monte Carlo particle code |
| `parent_pdg` | `int` | PDG code of the direct parent particle |
| `process_type` | `InteractionType` | Physics process that created this particle; derived from the raw int stored in `_process_type` |
| `sem_type` | `SemanticType` | High-level semantic category derived automatically at construction (see below) |
| `point_cloud` | `ndarray (N, ≥3)` | 3-D hit positions; columns 0–2 are x, y, z |

#### Optional attributes

These attributes are not required at construction time.  Unset float32 scalars hold `FLOAT_UNSET` (= `np.float32('nan')`); all other unset attributes hold `None`.

| Attribute | Type | Default | Description |
|---|---|---|---|
| `start` | `ndarray (3,) float32` | `None` | Trajectory start position (vertex).  Also accessible as `p.vertex`. |
| `end` | `ndarray (3,) float32` | `None` | Trajectory end position. |
| `momentum_start` | `ndarray (3,) float32` | `None` | 3-momentum at the start vertex (MeV/c). |
| `momentum_end` | `ndarray (3,) float32` | `None` | 3-momentum at the trajectory end (MeV/c). |
| `kinetic_energy_start` | `float32` | `NaN` | Kinetic energy at the start vertex (MeV). |
| `kinetic_energy_end` | `float32` | `NaN` | Kinetic energy at the end of the trajectory (MeV). |
| `mass` | `float32` | `NaN` | Particle rest mass (MeV/c²). |
| `root_pdg` | `int` | `None` | PDG code of the primary (root) ancestor particle. |
| `start_process_id` | `int` | `None` | Geant4/simulation process ID for this particle's creation. |
| `start_subprocess_id` | `int` | `None` | Geant4/simulation sub-process ID for this particle's creation. |
| `start_process_name` | `str` | `None` | Human-readable name of the creation process (e.g. `"eIoni"`). |
| `end_process_id` | `int` | `None` | Geant4/simulation process ID for this particle's termination. |
| `end_subprocess_id` | `int` | `None` | Geant4/simulation sub-process ID for this particle's termination. |
| `end_process_name` | `str` | `None` | Human-readable name of the termination process. |

**Checking whether an optional attribute is set:**

```python
if p.start is not None:              # array / int / str / list check
    print(p.start)

import numpy as np
if not np.isnan(p.kinetic_energy_start):  # float32 scalar check
    print(p.kinetic_energy_start)
```

**`vertex` property** — `p.vertex` is a read/write alias for `p.start`.

#### Semantic types

`sem_type` is derived automatically from `process_type`, `pdg`, `parent_pdg`, and `point_cloud` at construction time via `SetSemanticType`.  Particles with fewer than `min_pc_size` points may be reclassified as `kLEScatter`.

| Value | Meaning |
|---|---|
| `kShower` | EM shower fragment |
| `kTrack` | Charged track |
| `kDelta` | Delta-ray |
| `kMichel` | Michel electron |
| `kLEScatter` | Low-energy scatter product |
| `kUnknown` | Unclassified |

---

### Step 0 — Configuration

`pysupera` uses [Hydra](https://hydra.cc) for configuration.  The default config lives in `pysupera/conf/config.yaml`.

**Key top-level parameters:**

| Key | Default | Description |
|---|---|---|
| `distance_threshold` | `5.0` | Proximity distance *D* in point-cloud coordinate units |
| `verbose` | `true` | Print per-event progress |
| `report` | `true` | Print end-of-run summary statistics |
| `enable_diagnostics` | `false` | Record every merge decision for inspection |
| `check_particle_tree` | `false` | Validate parent–child consistency before partitioning |
| `checker` | `gpu` | Default compute backend (config group; override with `checker=cpu_single` etc.) |

**Particle preprocessing parameters** (`particle.*`):

| Key | Default | Description |
|---|---|---|
| `particle.min_pc_size` | `-1` | Min point-cloud size for semantic-type classification; `-1` disables (all sizes accepted) |
| `particle.merge_duplicates` | `true` | Collapse points sharing identical (x,y,z) coordinates before partitioning |
| `particle.defragment` | `true` | Split disconnected point-cloud fragments into new `kLEScatter` particles |
| `particle.preprocessor.name` | `scipy` | Defragmentation backend: `scipy`, `gpu`, or `rapids` |

**Programmatic loading** (notebooks, tests):

```python
from hydra import initialize, compose
from pysupera.config import build_conditions, build_checker, configure

with initialize(config_path="pysupera/conf", version_base=None):
    cfg = compose("config", overrides=[
        "checker=cpu_single",
        "distance_threshold=5.0",
        "io.input_path=my_file.h5",
        "io.output_path=out.h5",
    ])

configure(cfg)   # sets module-level defaults (min_pc_size, etc.)
```

---

### Step 1 — Preprocessing

Two optional preprocessing stages run before partitioning.

#### Merge Duplicates

Collapses points that share identical (x, y, z) voxel positions within each particle's point cloud.  Aggregation: `time = min`, `energy = sum`, `dEdx = max`.  Enable with:

```yaml
# config.yaml
particle:
  merge_duplicates: true
```

#### Defragmentation

A particle's point cloud may be *fragmented* — split into spatially disconnected clusters that should logically belong to the same particle.  The defragmenter runs a connected-components algorithm (eps-ball radius = `distance_threshold`) on each particle's point cloud independently, then detaches small disconnected fragments (size ≤ `min_pc_size`) as new `kLEScatter` particles.

Enable with:

```yaml
particle:
  defragment: true
  min_pc_size: -1       # default: -1 (disabled); set e.g. 5 to split small fragments
  preprocessor:
    name: scipy         # choices: scipy (default), gpu, rapids
```

| Backend | Requirement | Best for |
|---|---|---|
| `scipy` | `scipy` | General use; fast early-exit for non-fragmented particles |
| `gpu` | `cupy` | Large point clouds (> `min_pts_for_gpu` points) |
| `rapids` | `cupy` + `cudf` + `cugraph` | Fully GPU, no CPU round-trip |

All backends share the same fast-path early-exit checks (single point, two-point connected, bounding-box diameter ≤ eps) that skip the full CC algorithm for the majority of particles.

---

### Step 2 — Applying Conditions

A *condition* defines which pairs of particles (or partition representatives) are candidates for merging, and optionally post-filters the proximity-passing pairs.

Each condition implements:

```python
class PartitionConditionBase(ABC):
    def get_candidates(self, partitioner) -> List[Tuple[int, int]]:
        """Particle-level candidate pairs (legacy path)."""
    def post_filter(self, partitioner, touching_pairs) -> List[Tuple[int, int]]:
        """Optional filter after proximity check."""
    def get_rep_candidates(self, partitioner, rep_lookup) -> List[Tuple[int, int]]:
        """Partition-level candidate pairs (incremental path)."""
    def get_unconditional_merges(self, partitioner, rep_lookup) -> List[Tuple[int, int]]:
        """Topology-only merges, no proximity check needed."""
```

Conditions are applied in order.  The recommended order is:

```
PhotonDecay → TouchingEMShower → CombineLEScatters → AbsorbLEScatter
```

See [Conditions Reference](#conditions-reference) for details on each.

---

### Step 3 — Generating Partitions

```python
partitioner = ParticlePartitioner(
    particles          = particles,
    distance_threshold = 5.0,
    backend            = "cpu-single",   # see Compute Backends
    enable_diagnostics = False,
)

# Option A: one condition at a time
partitions = partitioner.partition(condition)

# Option B: all conditions in one converging pass
partitions = partitioner.partition_combined(conditions)
```

`partition` and `partition_combined` both return `List[List[Particle]]`.  Each inner list is one partition; singletons represent unmerged particles.

The default mode is **partition-level proximity** (`partition_level_proximity=True`): proximity is tested between the merged point clouds of whole partitions rather than individual particles, and the algorithm iterates until convergence (at most `max_passes=10` passes; typically 1–2).

---

### Step 4 — Output

```python
from pysupera import open_writer, read_events

# Write events (particles carry updated partition labels)
with open_writer("output.h5") as writer:
    writer.append_event(particles)

# Read events back
with read_events("output.h5") as store:
    for particles in store.iter_events():
        ...
```

---

## CLI Usage

After installation the `run_pysupera` command is available:

```bash
# Basic run
run_pysupera io.input_path=/data/in.h5 io.output_path=/data/out.h5

# Use CPU multi-threaded backend, 8 threads
run_pysupera checker=cpu_multi checker.n_jobs=8

# Adjust distance threshold, disable verbose output
run_pysupera distance_threshold=3.0 verbose=false

# Enable defragmentation preprocessing
run_pysupera particle.defragment=true particle.min_pc_size=5

# Hydra parameter sweep (multiple runs)
run_pysupera --multirun \
    checker=gpu,bulk_gpu \
    distance_threshold=3.0,5.0,7.0
```

The working directory is not changed by Hydra, so relative paths in `io.*` resolve against the directory where the command is invoked.

---

## Compute Backends

Select the backend via the `backend` parameter of `ParticlePartitioner` or `checker=<name>` in the Hydra config.

| Backend name | Config key | Requirement | Description |
|---|---|---|---|
| `cpu-single` | `cpu_single` | `scipy` | Single-threaded KDTree |
| `cpu-multi` | `cpu_multi` | `scipy`, `joblib` | KDTree + joblib thread pool |
| `gpu` | `gpu` | `cuml` (RAPIDS) | RAPIDS NearestNeighbors (brute-force L2); **default** |
| `bulk-gpu` | `bulk_gpu` | `cupy` | CuPy chunked brute-force; accepts `chunk_size` |
| `numba` | `numba` | `numba`, CUDA | CUDA kernel with shared-memory tiling; accepts `block_size` |
| `cell-hash-cpu-single` | `cell_hash_cpu_single` | `scipy` | Cell-hash spatial index, single thread |
| `cell-hash-cpu-multi` | `cell_hash_cpu_multi` | `scipy`, `joblib` | Cell-hash + joblib |
| `cell-hash-gpu` | `cell_hash_gpu` | `cupy` | Cell-hash on CPU, distance kernels on GPU |

For the partition-level proximity mode (the hot path), all CPU backends override `batch_check_cloud_proximity` to build the parent KDTree **once** per candidate group rather than once per pair.

---

## Algorithms

### Partition-level Incremental Algorithm

The main algorithm (`_build_partitions_incremental`) maintains one representative `Particle` per partition.  A representative carries:
- All scalar attributes (PDG, semantic type, ancestry) of the particle that "won" each merge.
- A **lazily concatenated** point cloud (`_cloud_parts`) that is the union of all constituent particles' point clouds—concatenation is deferred until a proximity query actually needs the full array.

**Initialisation:**
- A `UnionFind` structure over all *n* particles.
- One representative per particle (shallow copy, `_cloud_parts = [original_cloud]`).
- `rep_lookup`: `particle_id → representative Particle`.
- `rep_members`: `id(rep) → [list of particle IDs]` (reverse map for O(|partition|) updates).

**Pass loop** (repeats until convergence, ≤ `max_passes`):

For each condition:

1. **Unconditional merges** (`get_unconditional_merges`): topology-driven pairs requiring no proximity check (e.g. photon → e⁺e⁻ from `PhotonDecay`).
2. **Candidate generation** (`get_rep_candidates`): directed `(child_rep_id, parent_rep_id)` pairs among current partition representatives.  Pairs are resolved to UF roots, deduplicated, then grouped by parent representative.
3. **Batch proximity check** (`batch_check_cloud_proximity`): for each parent group, the parent's cloud is materialized once (lazy concat), a single KDTree is built, and all children in the group are queried against it.
4. **Post-filter** (`post_filter`): condition-specific constraint (e.g. one merge per `kLEScatter`).
5. **Merge**: for each accepted pair, the child partition is absorbed into the parent.  The parent's `_cloud_parts` list is extended with the child's chunks (no allocation).  `rep_members` allows O(|child partition|) update of `rep_lookup`.

The loop terminates when a full pass over all conditions produces zero new merges.  At the end, surviving reps materialize their deferred clouds with a single `np.concatenate`.

**Complexity (per pass):**  
- Candidate building: O(n_reps) per condition.  
- Proximity checks: O(n_candidates × n_points_per_cloud / n_unique_parents) KDTree queries; each parent tree is built once.  
- `rep_lookup` update: O(|merged partition|) per merge (instead of O(n) with a full scan).

---

### Proximity Check

Two point clouds are considered *touching* if the minimum Euclidean distance between any point in one cloud and any point in the other is ≤ *D*.

The default implementation:
1. **Bounding-box rejection**: if the minimum possible distance between the two bounding boxes exceeds *D*, return `False` immediately (no KDTree needed).
2. **KDTree query**: build a `scipy.KDTree` on cloud B, query all points of cloud A with `distance_upper_bound = D`, return `True` if any distance ≤ *D*.

GPU backends (`GPUChecker`, `BulkGPUChecker`) override this with device-resident implementations and do not construct a CPU KDTree.

---

### Condition Pipeline

Each condition uses a three-stage pipeline:

```
get_candidates / get_rep_candidates
        ↓  (candidate pairs, O(n_class_A × n_class_B) in the worst case)
batch_check_proximity / batch_check_cloud_proximity
        ↓  (touching pairs only)
post_filter
        ↓  (merge pairs)
```

Only the touching subset proceeds to `post_filter`, and only the final `merge_pairs` are applied to the Union-Find structure.

---

## Conditions Reference

### `PhotonDecay`

**Purpose:** Merge the ±11 (e⁺/e⁻) decay products of a photon (PDG 22) into the photon's partition.

**Mechanism:** Purely topology-based (`get_unconditional_merges`).  A photon typically has an empty point cloud; its spatial extent is entirely represented by its children.  No proximity check is performed.

**Merge direction:** child (e⁺/e⁻) → parent (photon).  The photon representative survives and its `point_cloud` grows to include all child points.

**When to run:** Always first, before any proximity-based condition, so that the photon's cloud is populated before it participates in further proximity checks.

---

### `TouchingEMShower`

**Purpose:** Merge PDG-11/22 parent–child pairs that share a common ancestor and whose point clouds are spatially touching.

**Candidate filter:**
- Both particles have PDG code 11 or 22.
- Direct parent–child relationship.
- Same `root_id`.

**Merge direction:** child → parent (directed by the parent–child tree).

**When to run:** After `PhotonDecay`, to consolidate EM shower fragments.

---

### `CombineLEScatters`

**Purpose:** Consolidate transitively connected `kLEScatter` particles into a single `kLEScatter` representative before absorption.

**Motivation:** `AbsorbLEScatter` makes pairwise decisions.  Without pre-consolidation, a chain *Shower A – LEScatter B – LEScatter C* may not correctly absorb B and C into A if A and C are not directly touching.

**Candidate filter:** All pairs of distinct `kLEScatter` partition representatives.

**Merge direction (tie-breaking, `_le_direction`):**
1. Larger point cloud becomes the parent.
2. On equal size: higher energy sum (column `PointFeature.energy`) wins.
3. On equal energy: stable arbitrary tie-break.

**When to run:** Before `AbsorbLEScatter`.

---

### `AbsorbLEScatter`

**Purpose:** Absorb each `kLEScatter` partition into at most one touching non-`kLEScatter` neighbour.

**Candidate filter:** All pairs between `kLEScatter` representatives and non-`kLEScatter` representatives.

**Post-filter:** One merge per `kLEScatter` — if multiple non-LE neighbours touch the same LE partition, only the first passing pair is kept.

**Merge direction:** `kLEScatter` representative → non-`kLEScatter` representative.

**When to run:** Last, after `CombineLEScatters`.

---

## Diagnostics

Enable with `enable_diagnostics=True`:

```python
partitioner = ParticlePartitioner(particles, 5.0, enable_diagnostics=True)
partitions  = partitioner.partition_combined(conditions)

# Why were particles 42 and 99 not merged?
print(partitioner.why_not_merged(42, 99))

# All recorded decisions involving particle 42
partitioner.print_all_decisions_for_particle(42)

# Aggregate summary
partitioner.print_diagnostics_summary()

# Export to JSON
partitioner.export_diagnostics("decisions.json")
```

Diagnostics add a per-pair record overhead.  With diagnostics disabled (the default), proximity checks use the fast `batch_check_proximity` path with no per-pair Python callback.

---

## Assumptions

1. **Coordinate units** — `distance_threshold` must be in the same units as the x, y, z columns of `point_cloud` (typically millimetres for LArTPC geometry).

2. **Parent–child completeness** — Each `parent_id` referenced by a particle should correspond to another particle in the same event list.  Enable `check_particle_tree: true` to validate this at runtime.  `children_map` and `ancestor_map` are built from the provided list; missing parents silently produce empty child lists.

3. **Single event at a time** — `ParticlePartitioner` is constructed fresh for each event.  The internal KDTree index (`checker`) is built during `__init__` and covers the particles as they exist at construction time.  Particles added by preprocessing (defragmenter splitting) must be passed to the constructor after preprocessing is complete.

4. **`_cloud_parts` attribute** — The incremental algorithm attaches `_cloud_parts` to each representative `Particle` object (a `list[ndarray]` of point-cloud chunks used for deferred concatenation).  This attribute is an internal implementation detail of the partitioner and is not serialised to HDF5.

5. **Point cloud columns** — Columns 0–2 are always treated as x, y, z.  Additional columns (time, energy deposit, dEdx, etc.) are carried through the pipeline unchanged but are not used by any proximity or condition logic except for the energy tie-break in `CombineLEScatters` (column index 4, `PointFeature.energy`).

6. **`kLEScatter` ordering** — `AbsorbLEScatter` takes the *first* touching non-LE neighbour in the order candidates are iterated, which depends on the ordering of `rep_lookup`.  This is deterministic within a single run but may vary across Python versions or dict iteration orders in unusual environments.

7. **Convergence** — The pass loop terminates when no new merges occur in a complete pass.  The loop is capped at `max_passes=10` as a safety guard.  In practice, 1–2 passes are sufficient for typical LArTPC events.

8. **GPU availability** — GPU backends (`gpu`, `bulk-gpu`, `numba`, `cell-hash-gpu`) raise an `ImportError` at construction time if the required libraries are not installed.  The CPU backends (`cpu-single`, `cpu-multi`, `cell-hash-cpu-single`, `cell-hash-cpu-multi`) require only `scipy` and optionally `joblib`.
