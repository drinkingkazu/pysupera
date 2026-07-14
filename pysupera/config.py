"""
Helpers for building workflow objects from a Hydra / OmegaConf config.

Option B — programmatic config loading (notebook / test friendly):

    from hydra import initialize, compose
    from pysupera.config import build_checker, build_conditions

    # Load config without changing the working directory
    with initialize(config_path="pysupera/conf", version_base=None):
        cfg = compose("config", overrides=[
            "checker=bulk_gpu",
            "checker.chunk_size=256",
            "distance_threshold=4.0",
            "io.input_path=/data/in.h5",
            "io.output_path=/data/out.h5",
        ])

    checker    = build_checker(cfg)
    conditions = build_conditions(cfg)

This module has no dependency on ``@hydra.main`` and works in any Python
context (notebooks, pytest, library code).
"""

from __future__ import annotations

from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Override-key auditing infrastructure
# ---------------------------------------------------------------------------

# Complete set of dotted config paths that build_pipeline consumes.
# Any user override whose key is NOT in this set is flagged as unused.
_PIPELINE_KEYS: frozenset[str] = frozenset({
    # I/O
    "io.input_path",
    "io.output_path",
    "io.compression",
    # Reader (config-group switch + individual keys)
    "reader",
    "reader.format",
    "reader.particle_key",
    "reader.step_key",
    "reader.ass_key",
    "reader.electron_energy_threshold",
    # Particle construction
    "particle.min_pc_size",
    # Merge-duplicates preprocessor
    "particle.merge_duplicates",
    # Voxelizer
    "particle.voxelize.enabled",
    "particle.voxelize.voxel_size",
    "particle.voxelize.origin",
    # Defragmentation preprocessor
    "particle.defragment",
    "particle.preprocessor.name",
    "particle.preprocessor.min_pts_for_gpu",
    "particle.preprocessor.sem_types",
    # Proximity checker / partitioner (config-group switch + per-backend keys)
    "checker",
    "checker.name",
    "checker.n_jobs",
    "checker.chunk_size",
    "checker.block_size",
    "distance_threshold",
    # Partitioner control
    "verbose",
    "report",
    "enable_diagnostics",
    "check_particle_tree",
    # Conditions (config-group switch + individual toggles)
    "conditions",
    "conditions.photon_decay",
    "conditions.touching_em_shower",
    "conditions.combine_le_scatters",
    "conditions.absorb_le_scatter",
})

# Registry: id(cfg) → frozenset of override key strings.
# Populated by load_cfg; consumed by build_pipeline.
_cfg_override_keys: dict[int, frozenset[str]] = {}


def _parse_override_keys(overrides: list[str]) -> frozenset[str]:
    """
    Extract the dotted key portion from a list of Hydra override strings.

    Strips leading sigils (``+``, ``++``, ``~``) and returns only the key
    part (everything before the first ``=``).

    Examples
    --------
    ``"distance_threshold=0.8"``  →  ``"distance_threshold"``
    ``"checker=cpu_single"``      →  ``"checker"``
    ``"+extra.key=1"``            →  ``"extra.key"``
    ``"~remove.key"``             →  ``"remove.key"``
    """
    keys: set[str] = set()
    for s in overrides:
        s = s.lstrip("+~")          # remove Hydra append / force / delete sigils
        key = s.split("=")[0]       # key=value  or  key  (group switch)
        if key:
            keys.add(key)
    return frozenset(keys)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def configure(cfg: DictConfig) -> None:
    """
    Apply top-level config values to module-level defaults.

    Call this once after loading the config — before constructing any
    :class:`~pysupera.data.Particle` objects — so that defaults like
    ``min_pc_size`` are respected by code that constructs particles without
    passing the parameter explicitly (e.g. :class:`~pysupera.io.EventStore`).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.particle.min_pc_size``).
    """
    import pysupera.data as _data
    _data._DEFAULT_MIN_PC_SIZE = int(cfg.particle.min_pc_size)


def build_preprocessor(cfg: DictConfig, verbose: bool = False):
    """
    Instantiate the defragmentation preprocessor described by
    *cfg.particle.preprocessor*.

    Returns ``None`` when ``cfg.particle.defragment`` is ``false``.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.particle``,
        ``cfg.distance_threshold``).
    verbose : bool, optional
        Forward to the backend constructor so that
        :meth:`~pysupera.preproc.DefragmentBase.process` prints
        per-particle details for any particle that reaches the full
        connected-components algorithm.  Default ``False``.

    Returns
    -------
    DefragmentBase or None

    Raises
    ------
    ValueError
        If ``cfg.particle.preprocessor.name`` is not a recognised backend.
    """
    from pysupera.preproc import (
        ScipyDefragmenter,
        GPUDefragmenter,
        RAPIDSDefragmenter,
    )

    if not cfg.particle.get('defragment', False):
        return None

    from pysupera.utils import SemanticType

    D            = float(cfg.distance_threshold)
    min_pc_size  = int(cfg.particle.min_pc_size)
    preproc_cfg  = cfg.particle.get('preprocessor', {})
    name         = preproc_cfg.get('name', 'scipy') if preproc_cfg else 'scipy'

    # Parse optional sem_types filter.  Config value is a list of SemanticType
    # name strings, e.g. ["kShower", "kTrack"].  Empty list → apply to all.
    raw_sem_types = list(preproc_cfg.get('sem_types', [])) if preproc_cfg else []
    valid_names = {st.name for st in SemanticType}
    invalid = [s for s in raw_sem_types if s not in valid_names]
    if invalid:
        raise ValueError(
            f"Invalid sem_types value(s) in preprocessor config: {invalid}.\n"
            f"Valid names are: {sorted(valid_names)}"
        )
    sem_types = [SemanticType[s] for s in raw_sem_types] if raw_sem_types else None

    # n_jobs: prefer preprocessor-specific override, fall back to checker.n_jobs
    n_jobs = int(
        preproc_cfg.get('n_jobs', cfg.checker.get('n_jobs', 1))
        if preproc_cfg else cfg.checker.get('n_jobs', 1)
    )

    if name == 'scipy':
        return ScipyDefragmenter(D, min_pc_size, verbose=verbose,
                                 sem_types=sem_types, n_jobs=n_jobs)

    if name == 'gpu':
        min_pts = int(preproc_cfg.get('min_pts_for_gpu', 64))
        return GPUDefragmenter(D, min_pc_size, min_pts_for_gpu=min_pts,
                               verbose=verbose, sem_types=sem_types,
                               n_jobs=n_jobs)

    if name == 'rapids':
        min_pts = int(preproc_cfg.get('min_pts_for_gpu', 64))
        return RAPIDSDefragmenter(D, min_pc_size, min_pts_for_gpu=min_pts,
                                  verbose=verbose, sem_types=sem_types,
                                  n_jobs=n_jobs)

    raise ValueError(
        f"Unknown preprocessor name {name!r}.  "
        f"Valid options: scipy, gpu, rapids."
    )


def build_voxelizer(cfg: DictConfig, verbose: bool = False):
    """
    Instantiate a :class:`~pysupera.preproc.VoxelizeProcessor` when
    ``cfg.particle.voxelize.enabled`` is ``true``.

    Returns ``None`` when voxelization is disabled or the block is absent.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.particle``).
    verbose : bool, optional
        Forward to the :class:`~pysupera.preproc.VoxelizeProcessor`
        constructor.  Default ``False``.

    Returns
    -------
    VoxelizeProcessor or None
    """
    vox_cfg = cfg.particle.get("voxelize", None)
    if not vox_cfg or not vox_cfg.get("enabled", False):
        return None

    from pysupera.preproc import VoxelizeProcessor

    raw_vs = vox_cfg.get("voxel_size", 1.0)
    # OmegaConf ListConfig → plain Python list so numpy can consume it.
    voxel_size = list(raw_vs) if hasattr(raw_vs, "__iter__") else float(raw_vs)

    raw_origin = vox_cfg.get("origin", None)
    origin = list(raw_origin) if raw_origin is not None else None

    return VoxelizeProcessor(
        voxel_size       = voxel_size,
        origin           = origin,
        verbose          = verbose,
        merge_duplicates = bool(vox_cfg.get('merge_duplicates', False)),
        store_mapping    = bool(vox_cfg.get('store_mapping',    False)),
    )


def build_merge_processor(cfg: DictConfig, verbose: bool = False):
    """
    Instantiate a :class:`~pysupera.preproc.MergeDuplicatesProcessor`
    when ``cfg.particle.merge_duplicates`` is ``true``.

    Returns ``None`` when ``cfg.particle.merge_duplicates`` is ``false``
    (or absent).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.particle``).
    verbose : bool, optional
        Forward to :class:`~pysupera.preproc.MergeDuplicatesProcessor`
        so that
        :meth:`~pysupera.preproc.MergeDuplicatesProcessor.process`
        prints per-particle details for any particle that had duplicate
        coordinates.  Default ``False``.

    Returns
    -------
    MergeDuplicatesProcessor or None
    """
    if not cfg.particle.get('merge_duplicates', False):
        return None
    from pysupera.preproc import MergeDuplicatesProcessor
    return MergeDuplicatesProcessor(verbose=verbose)


def build_checker(cfg: DictConfig):
    """
    Instantiate the :class:`~pysupera.proxck.ProximityChecker` described
    by *cfg.checker*.

    The checker is returned **uninitialised** — call
    ``checker.initialize(particles)`` before using it.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.checker`` and
        ``cfg.distance_threshold``).

    Returns
    -------
    ProximityChecker

    Raises
    ------
    ValueError
        If ``cfg.checker.name`` is not a recognised backend.
    """
    D    = float(cfg.distance_threshold)
    name = cfg.checker.name

    if name == "cpu-single":
        from pysupera.proxck import CPUSingleThreadChecker
        return CPUSingleThreadChecker(D)

    if name == "cpu-multi":
        from pysupera.proxck import CPUMultiThreadChecker
        return CPUMultiThreadChecker(D, n_jobs=int(cfg.checker.get("n_jobs", -1)))

    if name == "gpu":
        from pysupera.proxck import GPUChecker
        return GPUChecker(D, chunk_size=int(cfg.checker.get("chunk_size", 512)))

    if name == "bulk-gpu":
        from pysupera.proxck import BulkGPUChecker
        return BulkGPUChecker(D, chunk_size=int(cfg.checker.get("chunk_size", 512)))

    if name == "numba":
        from pysupera.proxck import NumbaKernelChecker
        return NumbaKernelChecker(D, block_size=int(cfg.checker.get("block_size", 128)))

    if name == "cell-hash-cpu-single":
        from pysupera.proxck import CellHashCPUSingleThreadChecker
        return CellHashCPUSingleThreadChecker(D)

    if name == "cell-hash-cpu-multi":
        from pysupera.proxck import CellHashCPUMultiThreadChecker
        return CellHashCPUMultiThreadChecker(D, n_jobs=int(cfg.checker.get("n_jobs", -1)))

    if name == "cell-hash-gpu":
        from pysupera.proxck import CellHashGPUChecker
        return CellHashGPUChecker(D)

    raise ValueError(
        f"Unknown checker name {name!r}.  "
        f"Valid options: cpu-single, cpu-multi, gpu, bulk-gpu, numba, "
        f"cell-hash-cpu-single, cell-hash-cpu-multi, cell-hash-gpu."
    )


def check_particle_list(particles) -> None:
    """
    Verify that the particle list is self-consistent.

    For every particle whose ``parent_id`` differs from its own ``id``
    (i.e. every non-root particle), check that a particle with that
    ``parent_id`` exists in *particles*.  Raises :exc:`ValueError` if
    any parent IDs are unresolved, listing all missing IDs.

    Parameters
    ----------
    particles : list of Particle
        The particle collection to validate.

    Raises
    ------
    ValueError
        If one or more ``parent_id`` values do not correspond to any
        particle ``id`` in the list.
    """
    id_set = {p.id for p in particles}
    missing: dict[int, list[int]] = {}   # missing_parent_id → [child_ids]
    for p in particles:
        if p.parent_id != p.id and p.parent_id not in id_set:
            missing.setdefault(p.parent_id, []).append(p.id)
    if missing:
        lines = [
            f"  parent_id={pid} missing (referenced by child ids: {cids})"
            for pid, cids in sorted(missing.items())
        ]
        raise ValueError(
            f"Particle list is incomplete — {len(missing)} unresolved parent ID(s):\n"
            + "\n".join(lines)
        )


def build_conditions(cfg: DictConfig) -> list:
    """
    Instantiate the merging conditions that are enabled in *cfg.conditions*.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.conditions``).

    Returns
    -------
    list of PartitionConditionBase
        Conditions in the canonical order they should be applied.
    """
    from pysupera.conditions.touching_em_shower import TouchingEMShower
    from pysupera.conditions.absorb_le_scatter import AbsorbLEScatter
    from pysupera.conditions.combine_le_scatters import CombineLEScatters
    from pysupera.conditions.photon_decay import PhotonDecay

    conditions = []

    # 1. Photon decay: merge e+/e- children into photon rep unconditionally
    #    (topology-only, no proximity); must run early so photon reps have
    #    real point clouds before any spatial condition inspects them.
    if cfg.conditions.get("photon_decay", True):
        conditions.append(PhotonDecay())

    # 2. Consolidate transitively-touching kLEScatter chains into one blob
    #    before absorption so chains are absorbed as a unit.
    if cfg.conditions.get("combine_le_scatters", True):
        conditions.append(CombineLEScatters())

    # 3. Absorb every kLEScatter rep into a touching non-LE rep.
    #    After this step no standalone kLEScatter representative remains,
    #    so TouchingEMShower (step 4) cannot accidentally promote a
    #    kLEScatter rep to a shower parent.
    if cfg.conditions.get("absorb_le_scatter", True):
        conditions.append(AbsorbLEScatter())

    # 4. Merge touching PDG-11/22 parent-child pairs into EM shower fragments.
    #    Runs last in stage-1 so all kLEScatter reps are already absorbed.
    if cfg.conditions.get("touching_em_shower", True):
        conditions.append(TouchingEMShower())

    return conditions


def build_reader(cfg: DictConfig):
    """
    Instantiate an :class:`~pysupera.readers.EventReaderBase` described by
    *cfg.reader*, opening the file at *cfg.io.input_path*.

    The reader format is selected by ``cfg.reader.format``.  Currently only
    ``"edepsim_h5"`` is supported.

    JAXTPC visibility filtering
    ---------------------------
    When ``cfg.reader.jaxtpc_seg_path`` and ``cfg.reader.jaxtpc_inst_path``
    are both non-null/non-empty, the reader automatically switches to
    :class:`~pysupera.readers.JaxtpcHDF5Reader`, which restricts each
    particle's point cloud to energy-deposit segments that were visible in the
    JAXTPC readout simulation.  All particle-level metadata still comes from
    the EDepSim file at *cfg.io.input_path*.

    With JAXTPC masking disabled (default)::

        run_pysupera io.input_path=edepsim.h5 io.output_path=out.h5

    With JAXTPC visibility filtering::

        run_pysupera io.input_path=edepsim.h5 io.output_path=out.h5 \\
            reader.jaxtpc_seg_path=sim_seg_0000.h5 \\
            reader.jaxtpc_inst_path=sim_inst_0000.h5

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.  Must contain ``cfg.io.input_path`` and
        ``cfg.reader`` with at least a ``format`` key.

    Returns
    -------
    EventReaderBase
        An open reader positioned at the start of the file.  Close or use
        as a context manager when done.

    Raises
    ------
    ValueError
        If ``cfg.reader.format`` is not a recognised format string.

    Examples
    --------
    ::

        cfg = load_cfg(["io.input_path=/data/sim.h5"])
        with build_reader(cfg) as reader:
            for particles in reader:
                ...
    """
    fmt  = cfg.reader.get("format", "edepsim_h5")
    path = str(cfg.io.input_path)

    if fmt == "edepsim_h5":
        # Check whether JAXTPC visibility filtering has been requested.
        seg_path  = cfg.reader.get("jaxtpc_seg_path",  None)
        inst_path = cfg.reader.get("jaxtpc_inst_path", None)

        jaxtpc_active = bool(seg_path) and bool(inst_path)

        if jaxtpc_active:
            # Both JAXTPC paths are set → use visibility-filtered reader.
            from pysupera.readers import JaxtpcHDF5Reader
            return JaxtpcHDF5Reader(
                edepsim_path              = path,
                seg_path                  = str(seg_path),
                inst_path                 = str(inst_path),
                vertex_key                = str(cfg.reader.get("vertex_key",
                                                "vertex/geant4")),
                particle_key              = str(cfg.reader.get("particle_key",
                                                "particle/geant4")),
                electron_energy_threshold = float(cfg.reader.get(
                                                "electron_energy_threshold", 0.05)),
                min_pc_size               = int(cfg.particle.get("min_pc_size", -1)),
            )

        # Default: full EDepSim reader (no JAXTPC masking).
        from pysupera.readers import EDepSimHDF5Reader
        return EDepSimHDF5Reader(
            path,
            vertex_key                = str(cfg.reader.get("vertex_key",
                                             "vertex/geant4")),
            particle_key              = str(cfg.reader.get("particle_key",
                                             "particle/geant4")),
            step_key                  = str(cfg.reader.get("step_key",
                                             "pstep/lar_vol")),
            ass_key                   = str(cfg.reader.get("ass_key",
                                             "ass/particle_pstep_lar_vol")),
            electron_energy_threshold = float(cfg.reader.get(
                                             "electron_energy_threshold", 0.05)),
            min_pc_size               = int(cfg.particle.get("min_pc_size", -1)),
        )

    raise ValueError(
        f"Unknown reader format {fmt!r}.  "
        f"Valid options: edepsim_h5."
    )


def load_cfg(overrides: list[str] | None = None,
             config_path: str = "conf",
             config_name: str = "config") -> DictConfig:
    """
    Load the Hydra config programmatically without starting the Hydra runtime.

    Convenience wrapper so notebooks and scripts do not need to import Hydra
    APIs directly.

    Parameters
    ----------
    overrides : list of str, optional
        Hydra override strings, e.g.
        ``["checker=bulk_gpu", "distance_threshold=4.0"]``.
        Checker names may use hyphens or underscores interchangeably
        (``"checker=bulk-gpu"`` and ``"checker=bulk_gpu"`` are equivalent).
    config_path : str, optional
        Path to the ``conf/`` directory **relative to this file's location**.
        Default is ``"conf"`` (the ``conf/`` directory that lives inside the
        ``pysupera/`` package directory alongside this file).
    config_name : str, optional
        Base name of the root config file (without ``.yaml``).

    Returns
    -------
    DictConfig

    Examples
    --------
    >>> from pysupera.config import load_cfg
    >>> cfg = load_cfg(["checker=cpu_single", "distance_threshold=3.0",
    ...                 "io.input_path=/tmp/in.h5",
    ...                 "io.output_path=/tmp/out.h5"])
    >>> cfg.checker.name
    'cpu-single'
    """
    from hydra import initialize_config_dir, compose
    import os

    abs_conf = os.path.normpath(
        os.path.join(os.path.dirname(__file__), config_path)
    )

    # --- Normalise overrides ------------------------------------------------
    # 1. checker=<name>  : convert hyphens → underscores in the value so that
    #                      e.g. "bulk-gpu" resolves to bulk_gpu.yaml.
    # 2. checker.<key>=  : pull these OUT of the Hydra compose call and apply
    #                      them directly via OmegaConf after composition.
    #                      Hydra's struct-mode handling of config-group sub-keys
    #                      is brittle across versions (MissingConfigException /
    #                      ConfigCompositionException depending on whether the
    #                      key already exists and which sigil is used).
    #                      Setting them on the live DictConfig is simpler and
    #                      completely version-agnostic.
    _hydra_overrides:   list[str]        = []
    _checker_subkeys:   dict[str, str]   = {}   # key → raw string value

    for ov in (overrides or []):
        bare = ov.lstrip("+~")
        if bare.startswith("checker="):
            prefix = ov[: len(ov) - len(bare)]
            val    = bare[len("checker="):]
            _hydra_overrides.append(f"{prefix}checker={val.replace('-', '_')}")
        elif bare.startswith("checker.") and "=" in bare:
            rest = bare[len("checker."):]
            key, _, val = rest.partition("=")
            _checker_subkeys[key] = val
        else:
            _hydra_overrides.append(ov)

    # Hydra keeps a process-wide singleton; clear it so repeated calls inside
    # a notebook kernel always get a fresh composition.
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=abs_conf, version_base=None):
        cfg = compose(config_name, overrides=_hydra_overrides)

    # Apply checker sub-key overrides directly, bypassing struct-mode checks.
    if _checker_subkeys:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg.checker, False)
        for k, raw in _checker_subkeys.items():
            # Coerce to int or float when possible so callers get numeric types.
            try:
                v: object = int(raw)
            except ValueError:
                try:
                    v = float(raw)
                except ValueError:
                    v = raw
            OmegaConf.update(cfg.checker, k, v, merge=True)
        OmegaConf.set_struct(cfg.checker, True)

    configure(cfg)
    _cfg_override_keys[id(cfg)] = _parse_override_keys(overrides or [])
    return cfg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    A fully-configured pysupera processing pipeline.

    Holds pre-built reader, preprocessors, and conditions so that per-event
    work is reduced to a single :meth:`partition` call.  Construct via
    :func:`build_pipeline` rather than directly.

    Parameters
    ----------
    reader : EventReaderBase
        Open file reader.  Owned by this object; closed on :meth:`close`.
    merger : MergeDuplicatesProcessor or None
        Merge-duplicates step, or ``None`` if disabled.
    voxelizer : VoxelizeProcessor or None
        Voxelization step, or ``None`` if disabled.
    preprocessor : DefragmentBase or None
        Defragmentation step, or ``None`` if disabled.
    conditions : list of PartitionConditionBase
        Merging conditions in canonical application order.
    cfg : DictConfig
        Full Hydra config retained for partitioner construction.

    Examples
    --------
    ::

        pipeline = build_pipeline(cfg)
        with pipeline:
            for event_idx, particles, partitions in pipeline:
                print(event_idx, len(partitions))
    """

    def __init__(self, reader, merger, voxelizer, preprocessor, conditions, cfg):
        self.reader       = reader
        self.merger       = merger
        self.voxelizer    = voxelizer
        self.preprocessor = preprocessor
        self.conditions   = conditions
        self.cfg          = cfg

    # ------------------------------------------------------------------ #

    def preprocess(self, particles: list) -> list:
        """
        Apply all enabled preprocessing steps to *particles* in order:
        merge-duplicates → voxelize → defragment.

        Parameters
        ----------
        particles : list of Particle

        Returns
        -------
        list of Particle
            The (possibly modified / extended) particle list.
        """
        if self.merger and not (
            self.voxelizer and self.voxelizer.merge_duplicates
        ):
            particles = self.merger.process(particles)
        if self.voxelizer:
            particles = self.voxelizer.process(particles)
        if self.preprocessor:
            particles = self.preprocessor.process(particles)
        return particles

    def partition(self, particles: list, verbose: bool = True) -> list:
        """
        Preprocess *particles* and run the full partitioning algorithm.

        Timing for each stage is recorded and accessible via
        :meth:`print_timing` after this call returns.  The underlying
        :class:`~pysupera.partitioner.ParticlePartitioner` is kept as
        :attr:`last_partitioner` for post-hoc diagnostics.

        Parameters
        ----------
        particles : list of Particle
            Raw particle list (typically one event from the reader).
        verbose : bool, optional
            Forwarded to
            :meth:`~pysupera.partitioner.ParticlePartitioner.partition_combined`.
            When ``True`` (default) the partitioner prints per-pass progress
            and a partition-size histogram to stdout.  Set ``False`` to
            suppress all partitioner output and inspect results solely via
            :meth:`print_timing`.

        Returns
        -------
        list of list of Particle
            Each inner list is one partition.
        """
        import time
        from pysupera.partitioner import ParticlePartitioner

        n_in = len(particles)

        # Time each preprocessing step individually for print_timing detail.
        _preproc_t: dict = {}
        t0 = time.perf_counter()
        if self.merger and not (
            self.voxelizer and self.voxelizer.merge_duplicates
        ):
            _ts = time.perf_counter()
            particles = self.merger.process(particles)
            _preproc_t['merge_duplicates'] = time.perf_counter() - _ts
        if self.voxelizer:
            _ts = time.perf_counter()
            particles = self.voxelizer.process(particles)
            label = 'voxelize (+merge_duplicates)' if self.voxelizer.merge_duplicates else 'voxelize'
            _preproc_t[label] = time.perf_counter() - _ts
        if self.preprocessor:
            _ts = time.perf_counter()
            particles = self.preprocessor.process(particles)
            _preproc_t['defragment'] = time.perf_counter() - _ts
        self._t_preprocess = time.perf_counter() - t0
        self._t_preprocess_steps = _preproc_t
        self._n_after_preprocess = len(particles)

        partitioner = ParticlePartitioner(
            particles,
            distance_threshold = float(self.cfg.distance_threshold),
            backend            = self.cfg.checker.name,
            n_jobs             = int(self.cfg.checker.get("n_jobs", -1)),
            enable_diagnostics = bool(self.cfg.enable_diagnostics),
            check_completeness = bool(self.cfg.check_particle_tree),
        )
        self.last_partitioner = partitioner

        t0 = time.perf_counter()
        partitions = partitioner.partition_combined(self.conditions, verbose=verbose)
        self._t_partition = time.perf_counter() - t0

        self._last_n_in         = n_in
        self._last_n_partitions = len(partitions)
        return partitions

    def print_timing(self) -> None:
        """
        Print a summary table of the most recent :meth:`partition` call.

        Shows wall-clock time for preprocessing and partitioning, particle
        counts before/after preprocessing, and the resulting partition count.

        Raises
        ------
        RuntimeError
            If :meth:`partition` has not yet been called.
        """
        if not hasattr(self, "_t_partition"):
            raise RuntimeError("No timing data available — call partition() first.")

        total = self._t_preprocess + self._t_partition
        W = 60  # table width

        print("=" * W)
        print(f"{'Pipeline timing summary':^{W}}")
        print("=" * W)
        print(f"  {'Stage':<36}  {'Time (s)':>8}  {'%':>5}")
        print("-" * W)

        # ---- Preprocessing section ----
        preproc_steps = getattr(self, "_t_preprocess_steps", {})
        if preproc_steps:
            print(f"  {'Preprocessing':<36}  {self._t_preprocess:>8.4f}"
                  f"  {100*self._t_preprocess/total:>4.1f}%")
            for step_name, step_t in preproc_steps.items():
                pct = 100 * step_t / total
                label = f"    {step_name}"
                print(f"  {label:<36}  {step_t:>8.4f}  {pct:>4.1f}%")
                # Sub-row: per-step statistics from last_stats
                if step_name == 'merge_duplicates':
                    proc = self.merger
                elif step_name.startswith('voxelize'):
                    proc = self.voxelizer
                elif step_name == 'defragment':
                    proc = self.preprocessor
                else:
                    proc = None
                stats = getattr(proc, 'last_stats', {}) if proc else {}
                if stats:
                    if step_name == 'defragment':
                        parts = [
                            f"sem_skip={stats['n_skipped']}",
                            f"early_exit={stats['n_early_exit']}",
                            f"cc_checked={stats['n_cc_checked']}",
                            f"fragmented={stats['n_fragmented']}",
                            f"spawned={stats['n_spawned']}",
                            f"n_jobs={stats.get('n_jobs', '?')}",
                            f"cc_pts(med/max/tot)={stats.get('cc_pts_median','?')}/{stats.get('cc_pts_max','?')}/{stats.get('cc_pts_total','?')}",
                        ]
                    else:
                        removed = stats['pts_before'] - stats['pts_after']
                        parts = [
                            f"affected={stats['n_affected']}",
                            f"pts {stats['pts_before']}→{stats['pts_after']}",
                            f"removed={removed}",
                        ]
                    print(f"  {'':36}    ({', '.join(parts)})")
        else:
            preproc_label = "Preprocessing (none)"
            print(f"  {preproc_label:<36}  {self._t_preprocess:>8.4f}"
                  f"  {100*self._t_preprocess/total:>4.1f}%")

        # ---- Partitioning section ----
        print(f"  {'Partitioning':<36}  {self._t_partition:>8.4f}"
              f"  {100*self._t_partition/total:>4.1f}%")

        # Per-condition breakdown (available when incremental path was used)
        cond_timing = getattr(self.last_partitioner, "_condition_timing", None)
        if cond_timing:
            for cname, ct in cond_timing.items():
                pct = 100 * ct['time_s'] / total
                label = f"    {cname}"
                print(f"  {label:<36}  {ct['time_s']:>8.4f}  {pct:>4.1f}%")
                # Sub-row: candidate → resolved → touching → merged
                parts = []
                if ct['n_unconditional']:
                    parts.append(f"unconditional={ct['n_unconditional']}")
                parts.append(f"candidates={ct['n_candidates']}")
                parts.append(f"resolved={ct['n_resolved']}")
                parts.append(f"touching={ct['n_touching']}")
                parts.append(f"merged={ct['n_merged']}")
                print(f"  {'':36}    ({', '.join(parts)})")

        print("-" * W)
        print(f"  {'Total':<36}  {total:>8.4f}  100.0%")
        print("=" * W)
        print(f"  Particles in           : {self._last_n_in}")
        print(f"  Particles after preproc: {self._n_after_preprocess}"
              + (f"  (+{self._n_after_preprocess - self._last_n_in} from defrag)"
                 if self._n_after_preprocess != self._last_n_in else ""))
        print(f"  Partitions out         : {self._last_n_partitions}")
        print("=" * W)

    def __len__(self) -> int:
        """Number of events in the attached reader."""
        return len(self.reader)

    def __iter__(self):
        """
        Iterate over all events, yielding ``(event_idx, particles, partitions)``
        tuples.  Preprocessing and partitioning are applied to each event.
        The ``verbose`` setting from the config (``cfg.verbose``) is forwarded
        to the partitioner.
        """
        verbose = bool(self.cfg.get("verbose", True))
        for idx, particles in enumerate(self.reader):
            partitions = self.partition(particles, verbose=verbose)
            yield idx, particles, partitions

    def close(self) -> None:
        """Close the underlying file reader."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        n = len(self.reader) if self.reader is not None else 0
        steps = []
        if self.merger:       steps.append("merge_duplicates")
        if self.voxelizer:    steps.append("voxelize")
        if self.preprocessor: steps.append("defragment")
        steps.append(f"{len(self.conditions)} conditions")
        return (
            f"Pipeline(events={n}, backend={self.cfg.checker.name!r}, "
            f"D={self.cfg.distance_threshold}, steps=[{', '.join(steps)}])"
        )


def build_pipeline(cfg: DictConfig, strict: bool = False) -> "Pipeline":
    """
    Build a fully-configured :class:`Pipeline` from a Hydra config.

    This is the **recommended entry point** for running the pysupera
    workflow.  It constructs every pipeline component in the correct order
    (reader, merge-duplicates, voxelizer, defragmenter, conditions) and
    audits the config for user overrides that were not consumed by any
    component — catching typos and accidentally ignored settings before
    data processing begins.

    .. note::
        Only overrides that the user explicitly passed to :func:`load_cfg`
        are checked.  Default values baked into ``conf/*.yaml`` are never
        flagged.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config as returned by :func:`load_cfg`.  If *cfg* was
        not produced by :func:`load_cfg` (e.g. loaded via ``hydra.compose``
        directly), no unused-override check is performed.
    strict : bool, optional
        When ``False`` (default), unused overrides are reported as a
        :class:`UserWarning`.  When ``True``, a :exc:`ValueError` is
        raised instead — useful in automated pipelines or CI where a
        misconfigured run should hard-fail.

    Returns
    -------
    Pipeline

    Raises
    ------
    ValueError
        If *strict* is ``True`` and unused override keys are detected.

    Examples
    --------
    ::

        from pysupera.config import load_cfg, build_pipeline

        cfg = load_cfg([
            "io.input_path=/data/sim.h5",
            "checker=cpu_single",
            "distance_threshold=0.8",
            "particle.defragment=true",
        ])

        with build_pipeline(cfg) as pipeline:
            print(pipeline)          # Pipeline(events=100, backend='cpu-single', ...)
            for idx, particles, partitions in pipeline:
                print(idx, len(partitions))

        # Or process a single pre-loaded particle list:
        pipeline = build_pipeline(cfg)
        partitions = pipeline.partition(my_particles)
    """
    import warnings

    # ── build all components ──────────────────────────────────────────────
    reader       = build_reader(cfg)
    merger       = build_merge_processor(cfg)
    voxelizer    = build_voxelizer(cfg)
    preprocessor = build_preprocessor(cfg)
    conditions   = build_conditions(cfg)

    # ── audit unused user overrides ───────────────────────────────────────
    override_keys = _cfg_override_keys.get(id(cfg))
    if override_keys is not None:          # None = cfg not from load_cfg; skip
        unused = override_keys - _PIPELINE_KEYS
        if unused:
            msg = (
                "build_pipeline: the following override key(s) were not "
                "consumed by any pipeline step and will have no effect:\n"
                + "\n".join(f"  {k}" for k in sorted(unused))
            )
            if strict:
                reader.close()
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    return Pipeline(
        reader       = reader,
        merger       = merger,
        voxelizer    = voxelizer,
        preprocessor = preprocessor,
        conditions   = conditions,
        cfg          = cfg,
    )
