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

    if name == 'scipy':
        return ScipyDefragmenter(D, min_pc_size, verbose=verbose,
                                 sem_types=sem_types)

    if name == 'gpu':
        min_pts = int(preproc_cfg.get('min_pts_for_gpu', 64))
        return GPUDefragmenter(D, min_pc_size, min_pts_for_gpu=min_pts,
                               verbose=verbose, sem_types=sem_types)

    if name == 'rapids':
        min_pts = int(preproc_cfg.get('min_pts_for_gpu', 64))
        return RAPIDSDefragmenter(D, min_pc_size, min_pts_for_gpu=min_pts,
                                  verbose=verbose, sem_types=sem_types)

    raise ValueError(
        f"Unknown preprocessor name {name!r}.  "
        f"Valid options: scipy, gpu, rapids."
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
    from pysupera.proxck import (
        CPUSingleThreadChecker,
        CPUMultiThreadChecker,
        GPUChecker,
        BulkGPUChecker,
        NumbaKernelChecker,
        CellHashCPUSingleThreadChecker,
        CellHashCPUMultiThreadChecker,
        CellHashGPUChecker,
    )

    D    = float(cfg.distance_threshold)
    name = cfg.checker.name

    if name == "cpu-single":
        return CPUSingleThreadChecker(D)

    if name == "cpu-multi":
        return CPUMultiThreadChecker(D, n_jobs=int(cfg.checker.get("n_jobs", -1)))

    if name == "gpu":
        return GPUChecker(D)

    if name == "bulk-gpu":
        return BulkGPUChecker(D, chunk_size=int(cfg.checker.get("chunk_size", 512)))

    if name == "numba":
        return NumbaKernelChecker(D, block_size=int(cfg.checker.get("block_size", 128)))

    if name == "cell-hash-cpu-single":
        return CellHashCPUSingleThreadChecker(D)

    if name == "cell-hash-cpu-multi":
        return CellHashCPUMultiThreadChecker(D, n_jobs=int(cfg.checker.get("n_jobs", -1)))

    if name == "cell-hash-gpu":
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

    if cfg.conditions.get("photon_decay", True):
        conditions.append(PhotonDecay())

    if cfg.conditions.get("touching_em_shower", True):
        conditions.append(TouchingEMShower())

    # Must come before absorb_le_scatter so transitive LEScatter chains are
    # consolidated into one particle before shower absorption.
    if cfg.conditions.get("combine_le_scatters", True):
        conditions.append(CombineLEScatters())

    if cfg.conditions.get("absorb_le_scatter", True):
        conditions.append(AbsorbLEScatter())

    return conditions


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

    with initialize_config_dir(config_dir=abs_conf, version_base=None):
        cfg = compose(config_name, overrides=overrides or [])

    configure(cfg)
    return cfg
