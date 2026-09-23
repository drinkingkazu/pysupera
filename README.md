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
   - [Browser viewers (`vis_force.html`, `vis_hits.html`)](#browser-viewers-vis_forcehtml-vis_hitshtml)
5. [Workflow](#workflow)
   - [Input: the `Particle` object](#input-the-particle-object)
   - [Choosing a reader](#choosing-a-reader)
   - [Step 0 — Configuration](#step-0--configuration)
   - [Step 1 — Preprocessing](#step-1--preprocessing)
   - [Step 2 — Applying Conditions](#step-2--applying-conditions)
   - [Step 3 — Generating Partitions](#step-3--generating-partitions)
   - [Step 4 — Output](#step-4--output)
6. [Output File Format](#output-file-format)
   - [Point ordering, and why it matters](#point-ordering-and-why-it-matters)
   - [Reading points](#reading-points)
   - [Which particles are stored](#which-particles-are-stored)
   - [Particle columns](#particle-columns)
   - [Interaction columns](#interaction-columns)
   - [ID conventions](#id-conventions)
   - [Wire and pixel readout](#wire-and-pixel-readout)
   - [Truth deposits or detected hits](#truth-deposits-or-detected-hits)
   - [Recovering the true x](#recovering-the-true-x--pointstrue_x_shift)
   - [Hit provenance (JAXTPC mode)](#hit-provenance-jaxtpc-mode)
     - [Seeing it](#seeing-it)
   - [Expert and debugging output](#expert-and-debugging-output)
7. [Training Data (PyTorch)](#training-data-pytorch)
   - [Voxel merging](#voxel-merging)
   - [Low-energy scatters](#low-energy-scatters--include_le)
   - [Distributed training](#distributed-training)
   - [Profiling and W&B logging](#profiling-and-wb-logging)
8. [CLI Usage](#cli-usage)
9. [Compute Backends](#compute-backends)
10. [Algorithms](#algorithms)
   - [Partition-level Incremental Algorithm](#partition-level-incremental-algorithm)
   - [Proximity Check](#proximity-check)
   - [Condition Pipeline](#condition-pipeline)
11. [Conditions Reference](#conditions-reference)
12. [Diagnostics](#diagnostics)
13. [Assumptions](#assumptions)

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
| `hdf5plugin` | ✅ | Registers the Blosc/LZ4 HDF5 filters — needed to read JAXTPC inputs and to write `io.compression=lz4` |
| `hydra-core ≥ 1.3` | ✅ | Hierarchical configuration (CLI and programmatic) |
| `omegaconf ≥ 2.3` | ✅ | Config composition (dependency of Hydra) |
| `joblib` | optional | Parallel KDTree queries (`cpu-multi` backend) |
| `cupy` | optional | GPU pairwise distances (`bulk-gpu`, `gpu` defragmentation) |
| `cuml` (RAPIDS) | optional | GPU nearest-neighbour index (`gpu` backend) |
| `cudf`, `cugraph` (RAPIDS) | optional | Fully GPU defragmentation (`rapids` preprocessor) |
| `numba` | optional | CUDA kernel proximity checker (`numba` backend) |
| `torch` | optional | Training-data pipeline (`pysupera.torchdata`) |
| `psutil` | optional | Process and worker memory in `pysupera.profiling` (falls back to `/proc`) |
| `wandb` | optional | Weights & Biases logging of the training-data pipeline |

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

# With the PyTorch training-data pipeline
pip install -e ".[torch]"

# ... and Weights & Biases logging for it
pip install -e ".[torch,wandb]"

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
| **INPUT FORMAT** | File format selector (`native` HDF5 or `edepsim_h5`); EDepSim-specific step/particle dataset keys and electron threshold |
| **PREPROCESSING** | Enable merge-duplicates and/or defragmentation; backend; semantic-type filter |
| **PARTITIONER** | Distance threshold, checker backend, `n_jobs` |
| **CONDITIONS** | Toggle each of the four conditions independently |
| **VIEW OPTIONS** | Show/hide legends; synchronise the two 3-D camera views; toggle **colour by sem type** (fast merged-trace rendering vs. per-particle instance colouring) |
| **PARTICLE FILTER** | Instantly show/hide particles by semantic type (no re-run needed); **Min points to display** hides small point clouds from the view without re-running the pipeline |

Hit **▶ Run** to execute the full pipeline and render two side-by-side 3-D point-cloud plots—original particles on the left, partitions on the right.

**Draw modes** (toggled via *colour by sem type* in VIEW OPTIONS):

| Mode | Left plot | Right plot | Speed |
|---|---|---|---|
| **by instance** (default) | One trace per particle, unique colour per ID; hover shows particle ID, PDG, parent | One trace per partition, unique colour per partition index | Slower for large events (many traces) |
| **by sem type** | One merged trace per semantic type with fixed colours | One merged trace per semantic type across all partitions | Fast — O(n\_sem\_types) traces regardless of event size |

The **PARTICLE FILTER** and **show legend** checkbox take effect immediately without re-running the pipeline.

<p align="center">
  <img src="figures/dash_display.png" alt="pysupera-app event display" width="900"/>
</p>

---

### Browser viewers

Four standalone HTML files that read an output file directly in the browser
— no server, no install, nothing to launch. Open the file and pick your
HDF5. They answer different questions:

| | `vis_force.html` | `vis_hits.html` | `vis_drift.html` | `vis_truth.html` |
|---|---|---|---|---|
| **question** | what is in this event, and how is it grouped? | which readout hits belong to which object? | what did the drift-time ambiguity do to this event? | how do the labels look on truth points versus sensor points? |
| **inputs** | the pysupera output | the output **and** the JAXTPC hits file | the output | the output **and** a truth companion |
| **views** | one 3-D cloud, plus an instance-genealogy graph | truth cloud beside the readout — six 2-D plane images for wire, a second 3-D cloud for pixel | two 3-D clouds: true x beside inferred x | two 3-D clouds: Geant4 deposits beside the pysupera points |
| **needs** | any 3.x output | JAXTPC mode, so `groups/` exists | `points/true_x_shift`, so a pixel hit-mode run | `groups/`, plus `pysupera-truth-subset` output |

Both read HDF5 through **h5wasm, which decodes gzip only**, so the inputs have
to be repacked first — see [Compression](#compression).

#### `vis_force.html` — exploring one event

Colour the cloud by energy, time, interaction id, ancestor track id, fragment
id, instance id or PDG code, with a selectable colormap and optional log
scale. LE and non-LE points toggle independently, energy and time thresholds
slide, and a bounding box clips the view. An animation plays the event by
time, and an interaction panel and a force-directed instance-genealogy graph
open alongside.

Shortcuts: `c` centre · `a` axes · `s` screenshot · `space` play/pause ·
`r` reset animation · `o` auto-rotate · `i` interaction panel · `g` graph ·
`d` debug log · `?` help.

#### `vis_hits.html` — linking 3-D to the readout

Truth voxels on the left, the readout on the right, and the same colour
meaning the same object in both. **Hover on either side and the matching hits
light up on the other.** The link is a single owner id, so the two views
cannot disagree.

What the right-hand pane is depends on the readout, which the viewer takes
from the hits file's `config/readout_type` (falling back to wire, as JAXTPC
does):

| readout | right-hand pane |
|---|---|
| **wire** | six 2-D images, one per plane per volume — wire against drift tick. A wire hit carries one spatial coordinate, so an image is the only honest way to draw it |
| **pixel** | a second 3-D cloud, axes `x = drift tick`, `y = pixel py`, `z = pixel pz`. A pixel hit is already 3-D; flattening it to an image would throw away an axis the detector actually measured |

The pixel pane is deliberately left in readout coordinates rather than
converted to mm: the left pane already answers "where in the detector", and
this one answers "where on the anode".

| control | |
|---|---|
| **colour by** | instance, fragment or interaction — the last two derived, not read |
| **others** | opacity of everything not under the cursor. The cloud uses RGBA vertex colours with depth-write off, so unselected points are genuinely transparent and never hide the selection behind them |
| **hide LE** | drops low-energy points from both views. Filtering happens at build time, so the picker cannot select something hidden |
| **point info** | x/y/z, time, dE, dX, dE/dX, LE and owner for a truth voxel; peak charge, group, LE and owner for a hit, plus wire and time (wire) or py/pz and time (pixel) |
| **c** | fit the view |

Preparing its two inputs:

```bash
pysupera-repack out.h5 out_gz.h5 -c gzip              # the pysupera output
pysupera-hits-subset sim_wire_hits_0000.h5 hits_small.h5 -n 10
```

`-c gzip` is not optional: omitting it keeps each dataset's existing filter,
leaving a file the browser cannot read. The second command exists because the
hits file is mostly per-tick CSR arrays the viewer never opens — keeping only
`group_ids`, `peak_charges` and the hit centres took a 3-event wire sample
from 1279.7 MiB to 2.73 MiB, and a 3-event pixel sample from 43.6 MiB to
0.65 MiB.

`pysupera-hits-subset` handles either readout without being told which:
it finds planes structurally (any subgroup with `group_ids`) and keeps
whichever centres they store — `center_wires`/`center_times` for wire,
`center_py`/`center_pz`/`center_times` for pixel. It copies `config`'s
attributes across too, so the subset still says which readout it is.

`vis_hits.html` refuses to draw rather than show something false: a red
banner appears for a filter it cannot decode, a missing `groups/` table, or
an event where every voxel resolves to one owner. Files written before the
group table, or before `groups/is_le`, still load — it says what is missing
and shows what it can. The mapping it displays is described under
[Hit provenance](#hit-provenance-jaxtpc-mode).

#### `vis_drift.html` — what the t0 ambiguity did

Two 3-D panes over the same points: **true x** on the left
(`x + points/true_x_shift`), **inferred x** on the right (as stored, with t0
assumed at the beam time). Cameras are synchronised and both panes are framed
on the union of the two clouds, so a displacement reads as a displacement
rather than being normalised away by two separate fits.

**Hover a point in either pane and its whole object lights up in both**,
everything else fading to an adjustable opacity. The `highlight` selector
chooses what "its object" means — instance, fragment or interaction —
independently of the colour, so you can colour by semantic type while
following one instance.

| colour by | encoding |
|---|---|
| energy | one-hue sequential ramp; **diverging** blue↔red about a grey zero when the column is signed, which it is in hit mode — the readout response is bipolar and 13% of voxels are negative. Sign-preserving log (symlog) by default: the column spans twelve decades, and linear puts three quarters of the points within 1.5% of the midpoint. Range clipped to the 2nd–98th percentile |
| instance semantic type | the six categorical slots in fixed order |
| fragment / instance / interaction id | a stable hash — thousands of values, so no legend is possible; identity comes from the hover |
| particle type (PDG) | fixed hues for the six commonest species, everything else folded into "other" |

Hover shows energy, Geant4 track id, fragment / instance / interaction id,
semantic type, PDG, parent instance and parent PDG, plus both x values and
the shift between them. Only about two thirds of points fall inside a stored
particle row, so the Geant4 track id falls back to the fragment's
representative and says when it has.

The semantic-type strip filters classes in and out. That is not only a
convenience: six categorical hues cannot clear the all-pairs colour-vision
floor — no six in one lightness band can — so identity is never left to
colour alone. A legend is always up, the hover names the class in words, and
any pair can be isolated with the filter.

Prepare its input the same way as the others, with `pysupera-repack out.h5
out_gz.h5 -c gzip`. A file without `points/true_x_shift`, or one where every
shift is zero, loads fine and says so — both panes are then the same cloud.

#### `vis_truth.html` — the same labels on truth and on sensor points

Left pane: the Geant4 energy deposits, each painted with the label of
whatever pysupera object claimed it. Right pane: the cloud the algorithms
actually ran on. Hover either side and the object lights up in both.

The bridge is the **group**. A deposit knows its group (`deposit_to_group`,
in the JAXTPC hits file) and the output's `groups/` table says which fragment
owns each group — so the label travels backwards down the same chain the
reconstruction came up. Measured on one event, a fragment's truth deposits
sit a median 2.2 mm from that same fragment's hits.

Deposits whose group was never detected have no owner and are shown as such
rather than dropped — 34% of event 0, which is the charge the readout window
never recorded. There's a `hide undetected` toggle.

Its two inputs:

```bash
pysupera-repack out.h5 out_gz.h5 -c gzip
pysupera-truth-subset step.h5 hits.h5 truth_small.h5 -n 3
```

The second exists because the JAXTPC step and hits files are Blosc, which the
browser cannot decode, and because the viewer needs only the deposit
positions, `de`, `charge` and `deposit_to_group` — 65 MB of JAXTPC becomes
about 1 MB per event.

A useful self-check: run pysupera with `reader.point_source=deposits` and the
two panes should coincide, because the right pane's cloud then *is* the
voxelized truth deposits.

#### Display thresholds

`vis_drift.html` and `vis_truth.html` both have a **show ≥** slider — two of
them in `vis_truth.html`, one per pane, since the panes hold different
quantities. The detector simulation imposes no threshold of its own, so
without one every pixel carrying any induced-field influence is on screen.

The slider position is a percentile, but what it displays is the absolute
value it lands on, together with what survives:

| sensor cut | points kept | charge kept |
|---|---|---|
| none | 100% | 100% |
| ≥ 27.7 | 50% | **99.6%** |
| ≥ 354 | 20% | 96.5% |
| ≥ 1960 | 10% | 87.7% |

Half the points cost 0.4% of the charge. The truth pane behaves quite
differently — dropping its faintest 60% of deposits costs 61% of the dE —
because truth deposits have no diffusion halo.

#### What the energy column holds

Nothing in the point columns says whether the energy column is true dE or
measured charge, so the run writes it down as root attributes
(`point_source`, `hit_energy`, `hit_x_from`, `readout_type`) and the viewers
label the column from them. Guessing from the presence of negative values
works on a busy event and fails on a quiet one.

---

## Workflow

### Input: the `Particle` object

#### Required attributes

These attributes must always be supplied at construction time and are guaranteed to be set on every `Particle` instance.

| Attribute | Type | Description |
|---|---|---|
| `id` | `int` | Particle index within an event, assigned by the reader: `0 … n-1`, usable as a direct row index. **Not** the Geant4 track ID |
| `geant4_trackid` | `int` | Original Geant4 track ID, kept for provenance and for joining back to the input |
| `parent_id` | `int` | Direct parent's particle index; equals `id` for primary particles |
| `ancestor_id` | `int` | Particle index of the primary ancestor at the root of the shower/track genealogy |
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
| `ancestor_pdg` | `int` | `None` | PDG code of the primary (root) ancestor particle. |
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

### Choosing a reader

Which kind of input a run reads is a Hydra config group, one file per
arrangement, selected with `reader=`:

| `reader=` | input | point cloud |
|---|---|---|
| `edepsim_h5` (default) | EDepSim only | every Geant4 energy-deposit step |
| `jaxtpc_wire` | EDepSim + JAXTPC wire batch | deposits the readout detected |
| `jaxtpc_pixel` | EDepSim + JAXTPC pixel batch | the same, or the detected pixel image via `reader.point_source=hits` |

The two JAXTPC configs inherit `edepsim_h5`, so the dataset keys are written
once, and each adds only what its own arrangement needs — `jaxtpc_wire` has
no `point_source` or pixel-geometry options at all, because a wire plane has
no 3-D image to convert.

Particle metadata (PDG, track IDs, interaction type) always comes from the
EDepSim file at `io.input_path`, whichever reader is selected. The JAXTPC
configs mark their two input paths as required, so selecting one without
them stops the run rather than quietly falling back to reading EDepSim:

```
This reader config is for JAXTPC input and needs reader.jaxtpc_seg_path and
reader.jaxtpc_inst_path.  Pass them on the command line, e.g.
    reader.jaxtpc_seg_path=batch/step/run/name_step_0000_00.h5
    reader.jaxtpc_inst_path=batch/hits/run/name_hits_0000_00.h5
Use reader=edepsim_h5 to read the EDepSim steps directly instead.
```

The full set of config groups is `io`, `reader`, `checker` and `conditions`;
see [Step 0](#step-0--configuration).

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

## Output File Format

`run_pysupera` writes a single HDF5 file. This section describes format
**3.1.0**, recorded in the scalar dataset `format_version`. No HDF5 attributes
are used anywhere — everything is a dataset.

3.1.0 renamed columns for a consistent `<level>_` prefix, gave instances the
same two-range point layout as fragments, and added the true Geant4 parent.
Readers accept 3.0.0 files transparently.  3.0.0 replaced the three parallel
particle tables of 2.x with **one** table and
stores each point **once**. On a two-event file that is 6.4 MiB → 1.7 MiB.

### Layout

```
format_version   scalar str   "3.1.0"
n_events         scalar int64
events/offsets   (n_events+1,) int64   particle rows per event
points/offsets   (n_events+1,) int64   point rows per event
inter/offsets    (n_events+1,) int64   interaction rows per event
points/flat      (N, 6) float32        x, y, z, time, dE, dX
particles/<col>  (M,)
inter/<col>      (K,)
```

Everything is concatenated across events with CSR fenceposts: event *i*
occupies rows `offsets[i] : offsets[i+1]`.

### Point ordering, and why it matters

Points are ordered by

```
interaction  >  instance  >  is_LE  >  fragment  >  particle
```

Three consequences follow, and between them they replace everything 2.x needed
per-point group labels for:

1. **An instance's points are one contiguous block**, split into a non-LE part
   followed by an LE part. All three useful queries are single slices.
2. **A fragment's points are two runs** — its non-LE side and its LE side —
   because other fragments' non-LE points lie between them. Each side is
   always exactly *one* run, never more, so two ranges describe a fragment
   completely.
3. **LE-ness is positional**, so no per-point label is stored. A point is LE
   precisely when its index falls in its instance's `inst_pc_le_*` range.

The ordering is chosen for the query panoptic-segmentation training issues most
often — "the non-LE voxels of instance X" — which is a plain slice rather than
a mask over the instance's points.

### Reading points

```python
from pysupera.io_v3 import read_events_v3

with read_events_v3("out.h5") as store:
    v = store[0]                                   # column dict + points
    for r in np.flatnonzero(v.is_instance):
        non_le = v.points_of(r, "inst", le=False)  # one slice
        le     = v.points_of(r, "inst", le=True)   # one slice
        both   = v.points_of(r, "inst")            # one slice
        vtx    = v.interaction_of(r)
```

Equivalently, by hand:

```python
non_le = v.points[v["inst_pc_start"][r]    : v["inst_pc_end"][r]]
le     = v.points[v["inst_pc_le_start"][r] : v["inst_pc_le_end"][r]]

frag_nle    = v.points[v["frag_pc_start"][r]    : v["frag_pc_end"][r]]
frag_le     = v.points[v["frag_pc_le_start"][r] : v["frag_pc_le_end"][r]]
```

`-1` marks a side with no points. `points_of(r, "frag")` concatenates the two
runs; every other combination is a single slice.

### Which particles are stored

About 9% of them. A particle gets a row if it

1. **represents a fragment or instance that carries points**, or
2. **is an instance representative**, or
3. **lies on the ancestry of one**, including point-less links such as a π⁰
   between its two photons.

Merged shower members get no row — an instance is one object, and its
constituents are not individually addressable. **Every point is still stored**
regardless: a group's points are its members', whether or not those members
have rows, and no point is unreachable from some stored row.

### Particle columns

| Column | dtype | Meaning |
|---|---|---|
| `id` | int32 | this particle's index **within its event** |
| `geant4_trackid` | int32 | original Geant4 track ID (provenance) |
| `geant4_parent_trackid` | int32 | track ID of the **true** direct parent; `-1` for a primary |
| `geant4_parent_pdg` | int32 | PDG of that true parent; `-1` for a primary |
| `parent_id` | int32 | nearest **stored** ancestor; `== id` marks a primary |
| `ancestor_id` | int32 | the primary this particle descends from |
| `pdg`, `parent_pdg` | int32 | PDG codes |
| `interaction_id` | int32 | row index into this event's `inter/` slice |
| `interaction_type` | int32 | `InteractionType` enum |
| `sem_type` | int8 | this particle's own classification |
| `pc_start`, `pc_end` | int64 | this particle's own points |
| `frag_id` | int32 | representative's `id`; `== id` marks a fragment, `-1` none |
| `frag_sem_type` | int8 | the fragment's classification |
| `frag_pc_start/end` | int64 | the fragment's non-LE points |
| `frag_pc_le_start/end` | int64 | the fragment's LE points |
| `frag_merge_count` | int32 | Geant4 particles merged into the fragment |
| `frag_parent_id` | int32 | nearest ancestor fragment; `== id` if none |
| `inst_id` | int32 | representative's `id`; `== id` marks an instance |
| `inst_sem_type` | int8 | the instance's classification |
| `inst_pc_start/end` | int64 | the instance's non-LE points |
| `inst_pc_le_start/end` | int64 | the instance's LE points (adjacent: `inst_pc_end == inst_pc_le_start`) |
| `inst_merge_count` | int32 | Geant4 particles merged into the instance |
| `inst_parent_id` | int32 | nearest ancestor instance; `== id` if none |

A particle carries up to three classifications — its own, its fragment's and
its instance's — because merging reclassifies. Note that LE barely survives to
the group levels: `absorb_le_scatter` folds low-energy scatter into whatever
touches it, so ~31% of *points* are LE while almost no *instance* is.

### Interaction columns

| Column | dtype | Meaning |
|---|---|---|
| `interaction_id` | int32 | this table's own dense index — the value particles carry |
| `vertex_id` | int32 | the row this interaction had in the **input** vertex list |
| `x`, `y`, `z`, `time` | float32 | vertex position and time |
| `energy_sum`, `ke_sum` | float32 | vertex totals, copied from EDepSim |
| `reaction` | str | generator reaction label, copied from EDepSim |
| `part_start`, `part_end` | int32 | its particle rows |
| `pc_start`, `pc_end` | int64 | its points |

Interactions that no stored particle references are dropped and the survivors
renumbered. `interaction_id` is the index after renumbering, which is what a
particle's `interaction_id` refers to; `vertex_id` is the link back to the
input file, and is what survives the renumbering.

### Enumerating levels

A representative names itself, so each level is one comparison:

```python
fragments = v["frag_id"] == v["id"]
instances = v["inst_id"] == v["id"]
```

`-1` means the particle heads no group at that level. `inst_id == -1` is normal:
under `instance_output=traceable` (the default) an instance that nothing
visible depends on is dropped, so its members belong to no written instance.

### ID conventions

1. **`id` is pysupera's, not Geant4's.** It is `0 … N-1` over the event's
   *full* particle list, assigned before the output subset is chosen. Geant4
   track IDs are not guaranteed contiguous — an upstream stage may drop
   particles — and are kept separately in `geant4_trackid`. **Join back to
   the input on `geant4_trackid`, never on `id`.**

   **`id` is not a row index.** Only the particles worth keeping are written —
   about 17% of them — and rows are ordered by the point layout, not by `id`.
   So in a 2,902-row event the ids run to 21,254 and are not sorted. Build a
   map; do not index with an id:

   ```python
   row_of = {int(i): k for k, i in enumerate(v["id"])}
   k = row_of[some_id]          # not v["pdg"][some_id]
   ```

   The same applies to `parent_id`, `frag_id`, `inst_id`, `frag_parent_id`
   and `inst_parent_id`: all of them are ids, and all need the map.

2. **`geant4_trackid` is one-to-many.** Defragmentation splits a particle whose
   point cloud falls into disconnected pieces into several pysupera particles,
   and they all inherit the track they came from. `id` distinguishes them.

3. **Walking `parent_id` always terminates inside the file.** Because most
   particles have no row, `parent_id` points at the nearest *stored* ancestor,
   and a particle with no stored ancestor is marked a primary. `ancestor_id` is
   derived from that same redirected chain, so the walk and the stated ancestor
   agree.

**One asymmetry:** ids are per-event, but point ranges are stored *absolute*
into `points/flat`, so a whole-file reader slices with no arithmetic.
`EventStoreV3` rebases them on read, so inside an `EventView` everything shares
one origin and `v["pc_start"]` indexes `v.points` directly. Only raw h5py
readers see the absolute form.

### Point columns

```
0:x  1:y  2:z  3:time  4:dE  5:dX
```

Both merge stages operate within a single particle, so `dE` and `dX` both sum
over a particle-voxel and **dE/dX is column 4 / column 5**, exactly. Guard
against `dX == 0`: EDepSim writes zero-length steps, a few percent of voxels.

### Wire and pixel readout

JAXTPC simulates the two LArTPC readout geometries, and pysupera reads both
through the same code path:

| | wire | pixel |
|---|---|---|
| plane subgroups per volume | `U`, `V`, `Y` | `Pixel` |
| what one hit records | a wire and a drift tick | two pixel indices and a drift tick |
| hit centres in the hits file | `center_wires`, `center_times` | `center_py`, `center_pz`, `center_times` |
| the image it forms | three 2-D projections | one natively 3-D image |

**Nothing in the reader branches on this, and that is the point.** Visibility
is a question about a *group* — JAXTPC's cluster of energy deposits — and a
group is above threshold or it is not, regardless of how the readout that saw
it was arranged. The filter reads `group_ids` and never looks at a hit
centre, so a pixel batch needs no flag and no separate reader:

```bash
run_pysupera reader=jaxtpc_pixel \
    io.input_path=edepsim.h5 io.output_path=out.h5 \
    reader.jaxtpc_seg_path=batch/step/run/name_step_0000_00.h5 \
    reader.jaxtpc_inst_path=batch/hits/run/name_hits_0000_00.h5
```

`reader=` selects the kind of input — see
[Choosing a reader](#choosing-a-reader).

The run banner prints which one it found (`[run]   readout : pixel`), read
from the hits file's `config/readout_type`. A file without that attribute
predates it and is wire — the same assumption JAXTPC's own loader makes.
Plane subgroups are discovered structurally rather than by name, so a
geometry neither project has named yet still resolves.

What *does* differ is anything that draws hits: see
[`vis_hits.html`](#vis_hitshtml--linking-3-d-to-the-readout) for the two
readout panes, and `pysupera-hits-subset` for the centres each keeps.

```python
from pysupera.readers import read_readout_type
import h5py
with h5py.File("hits.h5") as f:
    print(read_readout_type(f))          # 'wire' or 'pixel'
```

### Truth deposits or detected hits

`reader.point_source` decides what a *point* is. The rest of the pipeline —
defragmentation, proximity, conditions, partitioning — is identical either
way; only the cloud changes.

| | `deposits` (default) | `hits` |
|---|---|---|
| a point is | a Geant4 energy deposit the readout detected | a fired pixel |
| geometry | truth geometry; the readout only selects | the detected image |
| carries | thresholding | pixelation, diffusion, threshold, drift-time ambiguity |
| points/event | 50k–143k | 1.4M–4.2M |
| cost/event | ~1.6 s | ~7 s |
| readout | wire or pixel | pixel only — a wire plane has no 3-D image |

Each hit still has exactly one Geant4 particle: a CSR entry belongs to one
group, and `group_to_track` gives each group one track. No nearest-neighbour
matching is involved.

```bash
run_pysupera preset=cubic_pixel_hits \
    io.input_path=edepsim.h5 io.output_path=out_hits.h5 \
    reader.jaxtpc_seg_path=batch/step/run/name_step_0000_00.h5 \
    reader.jaxtpc_inst_path=batch/hits/run/name_hits_0000_00.h5 \
    reader.jaxtpc_sensor_path=batch/sensor/run/name_sensor_0000_00.h5
```

`preset=cubic_pixel_hits` is a bundle of the values that are only right
together and that cut across config groups — the pixel geometry, the charge
threshold, `particle.min_pc_size`, `distance_threshold` and
`check_group_ownership`. Only the file paths stay on the command line, and
any flag still overrides it.

#### The charge threshold and `min_pc_size` are a matched pair

JAXTPC applies no threshold worth the name — 1 electron — so without one
every pixel carrying any induced-field influence becomes a point.
`reader.hit_charge_threshold` cuts on the magnitude of the induced charge in
electrons, matching JAXTPC's own encoder (the response is bipolar and ~16%
of hits are negative signal).

Cutting the halo also **thins the tracks**, which lowers the extent
threshold, so the two move together:

| `hit_charge_threshold` | track width | `min_pc_size` | agrees with truth@5 |
|---|---|---|---|
| 0 e⁻ | 13.1 mm | 85 | 99.0% |
| **500 e⁻** *(preset)* | **6.5 mm** | **13** | **95.5%** |
| 500 e⁻ | 6.5 mm | 85 *(mixed)* | 85.2% |

Mixing them is worse than either. The uncut pairing keeps every electron and
4.18M points per event; the preset's keeps 73.5% of the charge and 190k
points, and runs 4× faster.

#### The drift coordinate — `reader.hit_x_from`

`y` and `z` are pure geometry and invert exactly, to half a pixel. `x` is
drift time, and a tick counts from the start of the readout window — so it
measures `t_drift + t0`, and `t0` is precisely what a real detector has to
determine separately. Writing `T` for the time a hit's tick stands for,

```
T = (tick − reference_tick) · time_step
```

| | `nominal` (default) | `true_t0` |
|---|---|---|
| x | `x_anode ∓ T · v` | `x_anode ∓ (T − t0) · v` |
| assumes | the interaction happened at the beam time | perfect knowledge of `t0` |
| median distance to the truth cloud | 268 mm | **5.8 mm** (≈ one pixel) |
| isolates | nothing — the full effect | pixelation, diffusion, threshold |

`nominal` is what a detector can actually do. Its x is offset from the true x
by exactly `drift_direction · v · t0` — the interaction's own Geant4 time
turned into a displacement. Measured on the 10-event batch this was developed
against: `t0` is constant within an interaction (median spread 0.000 μs, max
0.531 μs → 0.85 mm) and differs *across* the interactions of one event by up
to 1094 μs → **175 cm**. So `nominal` is a rigid translation per interaction:
each keeps its own shape to sub-mm while whole interactions slide past one
another. `true_t0` puts the real `t0` back and reproduces truth `x` to
~0.3 mm, keeping every other detector effect.

#### Recovering the true x — `points/true_x_shift`

A wire-mode output stores truth geometry, so `x` is the true x. A pixel
hit-mode output under `nominal` does not: every point is displaced along the
drift axis by its interaction's own `t0`. The true position is kept anyway,
as a displacement rather than a second coordinate:

```python
with read_events_v3("out_hits.h5") as store:
    view  = store[0]
    shift = store.true_x_shift(0)          # None for files written before this
    true_x = view.points[:, 0] + shift
```

Zero throughout under `true_t0` and in deposit mode, so the relation holds
for every file and a consumer never has to ask which convention produced it.

It costs almost nothing because it is `drift_direction · t0 · v`, which takes
one value per *(interaction, volume)* pair — 28 in a test event, not 4.2M.
Measured on one event: 1.31 MB raw, **0.007 MB on disk under LZ4, 0.14% of
the output**; 7 ms to compute over 4.18M hits; +16.7 MB transient memory.

Stored per point rather than per particle because of a small but heavy
exception. The shift is exactly constant within a particle for 99.97% of
them — but the two that are not are cathode-crossing tracks with hits in
both volumes, where `drift_direction` flips, and they carry **15% of all
points** with internal spreads up to 204 mm.

The stored rows are voxels, so a voxel could in principle mix two shifts;
each takes the value of a hit that made it. Measured, 0.09% of voxels mix
anything at all and the worst disagreement inside one is **0.05 mm** — the
sub-microsecond `t0` jitter within an interaction. The 200 mm case never
shares a voxel, because the nominal x is precisely what slides those two
halves apart.

#### The reference tick — `reader.hit_reference_tick`

The tick at which a deposit sitting on the anode at Geant4 `t = 0` is
recorded: the anchor the whole drift coordinate hangs from. It is kept
separate from the anode because **a fit cannot tell the two apart** — the
fitted intercept is `x_anode + drift_direction · mm_per_tick ·
reference_tick`. So the anode is taken from the volume's stated extent
(`config/volume_ranges`, which is geometry and not in doubt) and the
reference tick is read off as what remains. On a run whose window opens at
Geant4 t=0 it comes out at zero, which is the cross-check; on a run with a
pre-window it comes out at the pre-window.

Setting it explicitly overrides the derivation. The calibration residual
still tests the *geometry* — pitch, velocity, drift direction, all
measurements — but deliberately does **not** test the stated tick: that is a
choice about where t=0 sits, and asking "what if the trigger were 100 ticks
later" must not be reported as a broken detector. How far the choice sits
from the data is reported as `reference_tick_offset_mm` instead.

#### What the readout window hides

The reference tick also decides whether points can leave the detector. On
this batch `num_time_steps = 2701` and one tick is 0.8 mm, so the recordable
ticks 0…2700 span exactly 2160 mm — one full drift — and map precisely onto
`[anode, cathode]`. A nominal x therefore **cannot** fall outside the volume
(measured: 0.04% do, by a few mm at the edges).

Instead the t0 displacement pushes charge off the *end of the window*, where
it is never recorded at all. And that turns out to be almost the whole story
of what "visible" means in this data:

| event / volume | deposits | masked | past the window | of the masked, past the window |
|---|---|---|---|---|
| 0 / 0 | 83,997 | 26.6% | 25.8% | **97.0%** |
| 0 / 1 | 132,430 | 38.2% | 37.6% | **98.1%** |
| 1 / 0 | 173,903 | 86.4% | 86.2% | **99.7%** |
| 1 / 1 | 99,822 | 73.5% | 73.0% | **99.4%** |

So a deposit is invisible here because the readout had closed, not because it
was below threshold. Both point sources lose the same charge for the same
reason, which is what makes them comparable — but it also means this batch
says very little about *threshold* effects. Move the reference tick, or
lengthen the window past one full drift, and the displacement becomes visible
as out-of-volume points instead. Both are counted in the run report and
neither is clipped: a nominal x is an inference, and clipping it would invent
a wall the reconstruction does not have.

#### The energy column — `reader.hit_energy`

`charge` (default) is what the readout measured. The response is bipolar, so
about 16% of hits carry negative charge; that is signal, not error.
`true_de` instead shares each group's true deposited energy over its hits in
proportion to `|charge|` — comparable with deposit mode's column, at the cost
of putting a truth quantity into a detected cloud. **`dx` is 0 in both
cases**: a hit is a pixel and a tick, and has no path length, so anything
reading dE/dx sees zero.

#### Geometry — configured, then audited

The pixel geometry is **stated in configuration and checked against the
data**, never measured from it. The order matters: a constant derived from
the same truth it is later compared against cannot fail for a geometry that
is wrong but linear — the fit simply absorbs the error, and a detector
effect this mode exists to measure is calibrated away instead of measured.

| | source |
|---|---|
| anode face | `config/volume_ranges` in the hits file — read for you |
| drift velocity, sampling period | the JAXTPC **sensor** file, via `reader.jaxtpc_sensor_path` |
| pixel pitch, drift direction | `reader.pixel_pitch_mm`, `reader.pixel_drift_direction` — no output file records these |
| reference tick | `reader.hit_reference_tick`, default 0 |

The reader applies them as given, converts each group's hit centre, and
measures the distance to the deposits behind it. A disagreement raises,
naming the offending number *and* what the data says it should be:

```
[run] Pixel geometry  (2 volume(s))
  volume 0  [stated]  pitch 4.3200 mm  v 1.6000 mm/us  dt 0.5000 us  drift -1  anode -2160.0 mm  ref tick 0
             data says  pitch 4.3200 mm  v 1.5999 mm/us  dt 0.5000 us  drift -1
             residual   x 0.33   y 1.07   z 1.06 mm   over 14,165 groups
```

Run hit mode without the constants once and the error lists what the data is
consistent with, so the config can be written from it and verified on the
next run. `reader.pixel_geometry_from_fit=true` measures them instead — off
by default and deliberately awkward, because it makes the check circular.

The reference tick is the one constant **not** audited: it is a choice about
where t=0 sits, not a measurement, so asking "what if the trigger were 100
ticks later" must not be reported as a broken detector. Its distance from
the data is reported as `reference_tick_offset_mm` instead.

`reader.pixel_geometry_report` returns all of this per volume after the
first read.

#### `min_pc_size` is a voxel count, and needs re-deriving per readout

`min_pc_size` decides whether a particle is a low-energy scatter. It counts
**occupied voxels**, not rows — row count is a property of the input
sampling rather than of the particle, and Geant4 deposits arrive every
0.3 mm where pixel sensor hits arrive on a 4.32 mm grid. The same particle
is 1 row in one and 40 in the other, so a threshold in rows silently
re-tunes itself when `point_source` changes.

The value follows from a physical rule: a chunk of ionisation is worth
calling a trajectory only when its direction can be read off it, which needs
it to be longer than it is wide. So the threshold is the cell count of a
blob one track-width across, `(width / voxel_size)³`.

| input | track width | implied threshold |
|---|---|---|
| truth deposits | 4.9 mm | (4.9/3)³ ≈ **5** — the default |
| pixel sensor hits | 12.8 mm | (12.8/3)³ ≈ 77, best fit **85** |

A track is 3.2× thicker in the pixel image, which is diffusion plus the
field response over about three pixel pitches. Measured over ten events, the
best-agreeing cut per event ranged 81–90, and at a fixed 85 the sensor-hit
labelling matches the truth-deposit labelling on **99.0%** of particles
(worst event 98.5%) — not merely the same fraction, the same particles. So
with `reader.point_source=hits`, set `particle.min_pc_size=85`; a run that
forgets warns.

Note this also changed truth-deposit labelling, from 85.8% to 95.3% LE,
because particles spread over a handful of rows inside one or two voxels are
now correctly measured as non-directional.

#### Two things that change in hit mode

**`distance_threshold` must clear the pixel diagonal.** At a 4.32 mm pitch,
a threshold of 5.2 mm is barely one pixel, so a single missing pixel severs a
track and defragmentation ends up measuring the grid rather than the physics.
Event 0, one event:

| `distance_threshold` | deposits: particles w/ points → partitions | hits: particles w/ points → partitions |
|---|---|---|
| 5.2 mm | 6,900 → 2,519 | 13,308 → 7,181 |
| 6.2 mm (> diagonal 6.11) | 6,900 → 2,495 | 8,845 → 3,924 |
| 8.0 mm | 6,900 → 2,426 | 8,128 → 3,197 |
| 10.0 mm | 6,900 → 2,349 | 7,372 → 2,555 |

Deposit mode never splits at all — 0.3 mm deposit spacing keeps a track
connected at any of these thresholds.

**A group can straddle a fragment split.** The `groups/` table rests on the
invariant that it cannot, which holds for deposits (0 of 354,959) and does
*not* hold for hits: the readout leaves gaps the deposit cloud does not have,
defragmentation cuts there, and 1.84% of groups end up on both sides.
`check_group_ownership=false` is therefore required in hit mode, and the run
resolves each straddled group by majority and reports how many there were.

### Hit provenance (JAXTPC mode)

Which readout hits belong to which reconstructed object. A hit is the
projection of a JAXTPC *group*, and `groups/` records the fragment that owns
each group, plus one bit saying whether that group is low-energy:

```python
with read_events_v3("out.h5") as store:
    frag_owner, is_le, vol_offsets = store.group_owners(ev)

groups = np.flatnonzero(frag_owner == my_fragment_id)
plane  = "Pixel"      # or "U" / "V" / "Y" for wire readout
gid    = inst_file[f"event_{ev:03d}/volume_0/{plane}/group_ids"][:] + vol_offsets[0]
hits   = np.flatnonzero(np.isin(gid, groups))
```

The join is the same either way — a hit names its group, and `groups/` names
the group's owner — so only the plane name changes between readouts. See
[Wire and pixel readout](#wire-and-pixel-readout).

Both arrays are indexed by event-global group number, `-1` where nothing
claims the group; `vol_offsets` shifts a plane's local group number into that
space. `pysupera.provenance` has `hits_of_groups` and `groups_of_particles`
for the same joins.

For instances, derive rather than read — `instance_of_groups(view,
frag_owner)` returns the owning instance per group, and the interaction
follows from the instance's `interaction_id`:

```python
from pysupera.provenance import instance_of_groups
inst_owner = instance_of_groups(store[ev], frag_owner)
```

#### What is stored, and what is not

| | stored? | why |
|---|---|---|
| fragment | yes | the only usable key — see below |
| instance | **no** | a fragment's points lie inside exactly one instance's block, so it follows by containment (verified exact on 29,201 groups) |
| interaction | **no** | follows from the instance's `interaction_id` |
| LE flag | yes | irreducible — see below |

**Why the fragment and not the particle.** The particle is the natural key and
does not work: a group's owner is usually an ordinary member particle, and
only about a sixth of particles get a row — 6,605 of 6,777 owners were
unresolvable in one event. Every fragment *with points* is stored by
construction (0 missing of 29,201), so the fragment is the finest key that
always resolves.

Note that `inst_id` and `frag_id` are **self-markers**, not membership: each
equals `id` on a representative row and is `-1` everywhere else. Membership is
positional, via the point ranges, which is why deriving the instance is a
containment search rather than a column join.

**Why the LE flag cannot be derived.** LE-ness is an attribute of the pysupera
particle, and a fragment mixes LE and non-LE members by design — that is what
its two point ranges are for. Measured, 26–30% of fragments contain both LE
and non-LE groups, so the fragment cannot answer it. Deriving it would mean
knowing which *point rows* a group produced, which is the opt-in deposit map
below. The bit costs 0.17 bytes/group; making it derivable instead, by storing
the owning particles as rows, costs about 20× more.

#### Why groups and not voxels

`group_to_track` gives each group one Geant4 track, but defragmentation splits
a track across several pysupera particles, so `track → particle` is
one-to-many and cannot regroup hits. What rescues it is that a group never
straddles a split — a group is a tight spatial cluster and defragmentation
clusters at `distance_threshold`. Over eight events, none of 354,959 groups
spanned two particles. So `group → particle` *is* a function, and
`hit → group → particle → fragment` is a chain of functions.

Going via voxels would be worse as well as larger: a group's deposits land in
2–4 different voxels 38% of the time, so `hit ↔ voxel` is many-to-many and
would need charge apportionment (`qs_fractions` in the inst file).

The invariant is checked per event, not assumed — `check_group_ownership`
(default true) raises naming the group and the two particles. `groups/` costs
about 21% of the output under LZ4, less under gzip.

#### Seeing it

`vis_hits.html` shows this mapping directly — 3-D voxels beside the 2-D
readout planes, hover either side to light up the other. See
[Browser viewers](#browser-viewers-vis_forcehtml-vis_hitshtml) for its
controls and for preparing its two inputs.

#### Exact per-voxel provenance — `particle.voxelize.store_mapping=true`

Off by default. Writes `<output>_voxmap.h5`, a CSR giving the input deposits
behind every output voxel. This is the exact record the group table
compresses, so it is the thing to enable if `check_group_ownership` ever
fires — and the only way to answer "which deposits made *this* voxel". It is
roughly ten times larger.

### Compression

Output is LZ4 by default (`io.compression`). The browser viewer uses h5wasm,
which supports **gzip only**:

```bash
pysupera-repack out.h5 out_vis.h5 --compression gzip --rechunk
```

`pysupera-repack` also resizes chunks to the final dataset lengths, which is
its other job — see the `repack` block in the config for why that needs a
second pass. Two further options:

```bash
# flip lz4 <-> gzip without having to remember which way round it is
pysupera-repack out.h5 flipped.h5 --compression auto

# compress every chunked dataset, not just those already compressed
pysupera-repack out.h5 small.h5 --compression gzip --scope all
```

`--compression auto` reads the file's dominant filter and swaps lz4 for gzip
or back; it errors rather than guess if the file is uncompressed. `--scope`
defaults to `source`, which leaves an uncompressed dataset uncompressed —
`all` overrides that. On a 100-event file, gzip is 65.4 MiB against LZ4's
89.4 MiB, so gzip is worth it for anything but the fastest reads.

### Expert and debugging output

Off by default, regenerable, and not intended for analysis.

#### The voxel mapping — `particle.voxelize.store_mapping=true`

Writes `<output>_voxmap.h5`: for every output voxel, which input deposits were
merged into it and with what energy. Two nested CSR levels:

```
particles/id, particles/vox_offsets   ->  a particle's voxels
voxels/input_offsets                  ->  a voxel's input points
flat/input_ids, flat/input_energies   ->  original point index + its energy
```

It carries one row per *input* deposit rather than per voxel, so it is
typically larger than the physics output itself. Nothing is computed when it is
off.

#### Seeing every instance — `instance_output=all`

Writes every instance the merger produced, including those that deposited
nothing and that nothing visible descends from. Use it to inspect the merger's
raw output; `traceable` is what analyses should read.

#### Per-merge records — `enable_diagnostics=true`

Records every merge decision in memory. Not written to the output file.


---

## Training Data (PyTorch)

`pysupera.torchdata` turns a 3.x file into batches for panoptic-segmentation
training. Install the extra first:

```bash
pip install -e ".[torch]"
```

```python
from pysupera.torchdata import make_dataloader

loader = make_dataloader("out.h5", batch_size=4, distributed=True,
                         num_workers=4)

for batch in loader:                      # batch_size counts *events*
    for ev in batch["events"]:
        x = ev["points"]                  # (V, 4) float32 — x, y, z, E
        s = ev["voxel_sem"]               # (V,) semantic label
        i = ev["voxel_instance"]          # (V,) instance id
        k = ev["voxel_interaction"]       # (V,) interaction id
```

Point clouds differ in length between events, so a batch stays a **list** of
events rather than a padded tensor; `batch_size` therefore means exactly a
number of events. `to_torch(ev, device)` converts one event's arrays to
tensors when the model needs them.

Alongside the three per-voxel label arrays, each event carries the object-level
tables `ev["instances"]` and `ev["interactions"]` — the same columns described
under [Particle columns](#particle-columns) and
[Interaction columns](#interaction-columns), restricted to instance rows. An
interaction's id is its row index, which is what `voxel_interaction` holds.

### Voxel merging

Model input is one row per voxel, but a voxel can receive energy from several
instances. Overlaps are resolved as follows:

- **Energy is summed** over every contribution, so no charge is lost.
- **Labels come from the winning contribution**, chosen by semantic priority
  `kTrack > kShower > kDelta > kMichel > kLEScatter`, then by earliest time
  within a class.

A track crossing a shower therefore keeps the voxel, and among two showers the
earlier one does. On a sample event ~10% of raw points share a cell with
another instance's.

### Low-energy scatters — `include_le`

LE deposits are roughly a third of all points. They are **excluded by
default**:

```python
loader = make_dataloader("out.h5", include_le=True)   # keep them
```

The flag governs the voxel data as a whole — input coordinates, energy, and all
three label arrays — so the model never sees a point that the labels do not
describe, and an excluded point contributes no energy either. On a sample
event, dropping LE removes 30% of the voxels and 13% of the energy.

A point counts as LE when its label is `kLEScatter`, which covers both an LE
deposit absorbed into some other instance (LE-ness is positional — see
[Point ordering](#point-ordering-and-why-it-matters)) and a point of an
instance that is itself LE. The instance and interaction labels of an absorbed
LE point remain those of its host, so it stays attributed to the shower or
track that absorbed it.

### Distributed training

`distributed=True` attaches a `DistributedSampler` so each rank sees a disjoint
shard of events. Call `loader.sampler.set_epoch(epoch)` at the top of every
epoch, or each rank replays the same order. Note that
`DistributedDataParallel` itself wraps the *model*; this module covers the
input side.

The HDF5 handle is opened lazily per worker, so `num_workers > 0` is safe.


### Profiling and W&B logging

`StreamMonitor` times the pipeline and samples memory around a loader:

```python
import wandb
from pysupera.torchdata import make_dataloader, StreamMonitor

wandb.init(project="lartpc-panoptic")

loader = make_dataloader("out.h5", batch_size=4, num_workers=4,
                         distributed=True)
mon = StreamMonitor(device="cuda", prefix="data")

for epoch in range(n_epochs):
    for batch in mon.iterate(loader, epoch=epoch):
        train_step(batch)               # already staged on the GPU

print(mon.finish())                     # also writes the W&B run summary
```

`iterate` stages each batch to *device*, forwards *epoch* to the
`DistributedSampler`, and logs per step. The metrics come in two kinds, and
mixing them up is the easy mistake.

**Main-process wall time.** These partition one iteration and sum to `step_s`:

| Metric | Meaning |
|---|---|
| `wait_s` | how long the loop **blocked** in `next(loader)` — workers not yet done, plus shipping the finished batch back and rebuilding it here |
| `stage_s` | host-to-device transfer |
| `consume_s` | the loop body — your training step, measured across the `yield` |
| `step_s` | `wait_s + stage_s + consume_s`, i.e. the real per-iteration wall time |
| `stall_fraction` | `wait_s / step_s` — **the headline number** |

**Worker-side cost.** Summed over the batch's events, measured wherever they
were produced:

| Metric | Meaning |
|---|---|
| `read_s` | h5py decompress + slice |
| `preprocess_s` | `build_event` (voxel merge, labels) plus any `transform` |
| `collate_s` | assembling the batch |
| `payload_mb` | array bytes the batch ships to the parent |
| `payload_mb_per_s` | `payload_mb / wait_s` — the rate batches actually arrive |

With `num_workers > 0` these run in other processes, concurrently with the
training step and with each other, so **their sum routinely exceeds `step_s`**.
They say what the work costs, not what it delays. Only `stall_fraction`
answers "is input the bottleneck".

Throughput and memory are logged alongside: `events_per_s`, `points_per_s`,
`n_points`, `batch_size`, and `cpu_rss_mb`, `cpu_rss_total_mb`,
`cpu_available_mb`, `gpu_alloc_mb`, `gpu_reserved_mb`, `gpu_peak_mb`,
`gpu_free_mb`, `gpu_total_mb`.

**`cpu_rss_total_mb` includes the worker processes**, which is where reading
and pre-processing actually allocate; the parent's own `cpu_rss_mb` says
nothing about them. On a 100-event file each worker added ~210 MiB over the
parent's 630 MiB, so eight workers cost ~4 GiB of host memory. Workers share
copy-on-write pages with the parent, so the total is an upper bound.

Enumerating children scans `/proc` and costs ~4.6 ms against ~0.2 ms for the
rest of the snapshot, so the handles are cached and re-enumerated only when
one dies or `profiling.CHILDREN_REFRESH_S` elapses — except that an *empty*
cache expires after `EMPTY_REFRESH_S`, since nothing in it could signal that a
worker pool has since appeared.

`summary()` adds totals and means, plus `wall_s` — measured by one clock
spanning the whole loop rather than summed — and `unaccounted_s`, the
difference between the two. That covers whatever falls outside a step — the
monitor's own overhead, and gaps between epochs such as tearing a worker pool
down — and should stay near zero; if it does not, something outside the
measured path is eating the clock. `first_wait_s` and `max_wait_s` isolate worker start-up, which
normally dominates every other wait.

Reading the output — two events per epoch, five epochs, a toy MLP, with one
warm-up epoch discarded:

```
                                   num_workers=0   nw=4,persist   nw=4,respawn
  wall                                   0.171 s        0.197 s        1.501 s
  wait_s      (blocked)        per step  13.87 ms       14.63 ms     120.18 ms
  stage_s     (host->device)   per step   0.41 ms        0.64 ms       1.13 ms
  consume_s   (training step)  per step   2.38 ms        3.41 ms       5.68 ms
  read_s      (worker)         per step   4.22 ms        9.46 ms      21.63 ms
  preprocess_s(worker)         per step   9.52 ms       12.86 ms      13.14 ms
  stall_fraction                           0.833          0.783          0.946
  first_wait_s                            9.8 ms        28.6 ms       230.1 ms
```

Three things worth reading off it:

- **`stall_fraction` is ~0.8 everywhere**, because the toy model is far too
  small to hide the input pipeline. That is the number to watch: with a real
  model `consume_s` grows and the fraction falls.
- **Workers do not help at this scale.** Two events per epoch is not enough
  work to amortise shipping batches between processes, so `num_workers=4` is
  marginally *slower* than loading in-process.
- **`persistent_workers=True` matters far more than the worker count.**
  Without it the pool is torn down and respawned every epoch, `first_wait_s`
  goes from 29 ms to 230 ms, and wall time is 8.8× worse. Pass it through
  `make_dataloader` to the DataLoader.

Beware one trap when reading your own numbers: the first steps of a process
pay for CUDA, cuBLAS and autograd initialisation, and that lands in
`consume_s`. Before the warm-up epoch above was added, `consume_s` read
116 ms against a true 2.4 ms. **Discard the first epoch when benchmarking.**

`StreamMonitor` warms the CUDA transfer path on construction, because the
first host-to-device copy in a process otherwise pays ~35 ms of lazy
initialisation against the ~1 ms a warm copy costs. Some first-step cost
remains while the caching allocator grows, so **discard step 1 when
benchmarking**.

The pieces are usable on their own: `pysupera.profiling.memory_snapshot()`
returns the memory dict, `WandbLogger` is the rank-aware no-op-safe sink, and
`stage_batch(batch, device)` returns `(batch, seconds)`.

Overhead is small enough to leave on: the per-event timing is two
`perf_counter` calls, and a memory snapshot is ~0.2 ms. Both can be turned off
anyway — `profile=False` on `make_dataloader`, `memory=False` on the monitor —
and `unaccounted_s` tells you whether it is worth it.


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
- Same `ancestor_id`.

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
