from .utils import InteractionType, SemanticType, SetSemanticType
import numpy as np

# ---------------------------------------------------------------------------
# Module-level defaults.
# Set these via pysupera.config.configure(cfg) after loading Hydra config.
# ---------------------------------------------------------------------------
_DEFAULT_MIN_PC_SIZE: int = -1

# ---------------------------------------------------------------------------
# Sentinel values for unset optional Particle attributes
# ---------------------------------------------------------------------------
# Design rationale
# ~~~~~~~~~~~~~~~~
# Python does not have typed "invalid" values the way C++ has INT32_MAX etc.
# Two conventions are used here depending on type:
#
#   float32 scalars → np.float32('nan')
#       NaN is the standard IEEE 754 "not a number".  It propagates visibly
#       through all numpy arithmetic (nan + x = nan, nan < x = False), so
#       unintentional use is conspicuous rather than silent.  It also round-
#       trips through HDF5 storage without special handling.
#
#   int scalars, arrays, strings → None
#       None immediately raises TypeError / AttributeError if used in
#       arithmetic or array indexing, making accidental use obvious.  It also
#       maps cleanly to h5py optional datasets (omitted = None on read).
#
# Check whether an attribute is set with the standard Python idiom:
#
#       if p.start is not None:         # array / int test
#       if not np.isnan(p.mass):        # float32 scalar test
#
# Module-level aliases exposed so callers can write:
#
#       from pysupera.data import FLOAT_UNSET
#       p.mass = FLOAT_UNSET           # reset to "not set"

FLOAT_UNSET: np.float32 = np.float32("nan")
"""Sentinel value for unset ``float32`` scalar attributes on :class:`Particle`."""


# ============================================================================
# Particle Data Class
# ============================================================================

class Particle:
    """
    Simulation particle with physical metadata and a 3-D point cloud.

    Required attributes (always set)
    ---------------------------------
    id : int
        Unique integer identifier within an event.  For Geant4 simulation
        this is the track ID.
    parent_id : int
        ID of the direct parent particle.  Equals ``id`` for primary
        (root) particles that have no parent; use ``p.parent_id == p.id``
        to test whether a particle is primary.
    root_id : int
        ID of the primary ancestor particle from which this particle
        descends (root of the shower/track genealogy).
    pdg : int
        PDG Monte Carlo particle code.
        See https://pdg.lbl.gov/2023/mcdata/mc_particle_id_contents.html.
    parent_pdg : int
        PDG code of the direct parent particle.
    sem_type : SemanticType
        High-level semantic category derived from *interaction_type*, *pdg*,
        *parent_pdg*, and *point_cloud* by
        :func:`~pysupera.utils.SetSemanticType`.
    interaction_type : InteractionType
        Physics process that created this particle, expressed as an
        :class:`~pysupera.utils.InteractionType` member.  Derived from
        the raw integer stored in ``_interaction_type``.
    point_cloud : numpy.ndarray, shape (N, ≥3)
        3-D spatial hits; columns 0–2 are x, y, z.  Additional columns
        (time, energy, dE/dx, …) may be present.

    Optional attributes (``None`` or ``FLOAT_UNSET`` when not set)
    --------------------------------------------------------------
    start : numpy.ndarray of float32, shape (3,) or None
        Trajectory start position (vertex).  Also accessible as
        ``p.vertex``.
    end : numpy.ndarray of float32, shape (3,) or None
        Trajectory end position.
    momentum_start : numpy.ndarray of float32, shape (3,) or None
        3-momentum at the start vertex (MeV/c).
    momentum_end : numpy.ndarray of float32, shape (3,) or None
        3-momentum at the trajectory end (MeV/c).
    kinetic_energy_start : float32 or FLOAT_UNSET
        Kinetic energy at the start vertex (MeV).
    kinetic_energy_end : float32 or FLOAT_UNSET
        Kinetic energy at the end of the trajectory (MeV).
    mass : float32 or FLOAT_UNSET
        Particle rest mass (MeV/c²).
    root_pdg : int or None
        PDG code of the primary (root) ancestor particle.
    start_process_id : int or None
        Geant4/simulation process ID for this particle's creation.
    start_subprocess_id : int or None
        Geant4/simulation sub-process ID for this particle's creation.
    start_process_name : str or None
        Human-readable name of the creation process (e.g. ``"eIoni"``).
    end_process_id : int or None
        Geant4/simulation process ID for this particle's termination.
    end_subprocess_id : int or None
        Geant4/simulation sub-process ID for this particle's termination.
    end_process_name : str or None
        Human-readable name of the termination process.
    """

    def __init__(
        self,
        id,
        parent_id,
        root_id,
        pdg,
        parent_pdg,
        interaction_type,
        point_cloud,
        min_pc_size=None,
        # ------------------------------------------------------------------
        # Optional physical attributes — default to None / FLOAT_UNSET.
        # See class docstring for sentinel-value conventions.
        # ------------------------------------------------------------------
        *,
        start=None,
        end=None,
        momentum_start=None,
        momentum_end=None,
        kinetic_energy_start=FLOAT_UNSET,
        kinetic_energy_end=FLOAT_UNSET,
        mass=FLOAT_UNSET,
        root_pdg=None,
        start_process_id=None,
        start_subprocess_id=None,
        start_process_name=None,
        end_process_id=None,
        end_subprocess_id=None,
        end_process_name=None,
    ):
        """
        Parameters
        ----------
        id : int
            Unique particle identifier within an event.
        parent_id : int
            ID of the direct parent particle.
        root_id : int
            ID of the root ancestor particle.
        pdg : int
            PDG Monte Carlo particle code.
        parent_pdg : int
            PDG code of the parent particle.
        interaction_type : int
            Raw integer encoding of the interaction process (stored as
            ``_interaction_type``).  The public ``interaction_type`` property
            returns the corresponding :class:`~pysupera.utils.InteractionType`
            member.
        point_cloud : numpy.ndarray, shape (N, ≥3)
            Spatial hit array.
        min_pc_size : int or None, optional
            Forwarded to :func:`~pysupera.utils.SetSemanticType`.  ``None``
            reads :data:`_DEFAULT_MIN_PC_SIZE`.

        All remaining parameters are keyword-only and optional; they
        default to ``None`` (arrays / ints / strings) or
        :data:`FLOAT_UNSET` (float32 scalars).  See the class docstring
        for full descriptions.
        """
        # --- required fields -----------------------------------------------
        self.id         = id
        self.parent_id  = parent_id
        self.root_id    = root_id
        self.pdg        = pdg
        self.parent_pdg = parent_pdg
        self._interaction_type = (interaction_type.value
                                    if isinstance(interaction_type, InteractionType)
                                    else int(interaction_type))  # raw int; retained for serialisation

        if min_pc_size is None:
            min_pc_size = _DEFAULT_MIN_PC_SIZE

        self.sem_type   = SetSemanticType(
            interaction_type, pdg, parent_pdg, point_cloud,
            point_cloud_size=min_pc_size,
        )
        self.point_cloud = point_cloud

        # --- optional physical attributes -----------------------------------
        self.start                = (np.asarray(start,            dtype=np.float32)
                                     if start            is not None else None)
        self.end                  = (np.asarray(end,              dtype=np.float32)
                                     if end              is not None else None)
        self.momentum_start       = (np.asarray(momentum_start,   dtype=np.float32)
                                     if momentum_start   is not None else None)
        self.momentum_end         = (np.asarray(momentum_end,     dtype=np.float32)
                                     if momentum_end     is not None else None)
        self.kinetic_energy_start = np.float32(kinetic_energy_start)
        self.kinetic_energy_end   = np.float32(kinetic_energy_end)
        self.mass                 = np.float32(mass)
        self.root_pdg             = root_pdg
        self.start_process_id     = start_process_id
        self.start_subprocess_id  = start_subprocess_id
        self.start_process_name   = start_process_name
        self.end_process_id       = end_process_id
        self.end_subprocess_id    = end_subprocess_id
        self.end_process_name     = end_process_name

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def vertex(self):
        """Alias for :attr:`start` (trajectory start / vertex position)."""
        return self.start

    @vertex.setter
    def vertex(self, value):
        self.start = (np.asarray(value, dtype=np.float32)
                      if value is not None else None)

    @property
    def interaction_type(self) -> "InteractionType":
        """
        :class:`~pysupera.utils.InteractionType` member corresponding to
        the raw integer stored in ``_interaction_type``.

        Read-only; modify by setting ``_interaction_type`` directly if needed.
        """
        try:
            return InteractionType(int(self._interaction_type))
        except ValueError:
            return InteractionType.kInvalidProcess

    # ------------------------------------------------------------------ #
    # Factory methods                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_arrays(
        cls,
        ids,
        parent_ids,
        root_ids,
        pdgs,
        parent_pdgs,
        interaction_types,
        point_clouds,
        min_pc_size=None,
    ):
        """
        Construct a list of :class:`Particle` objects from columnar arrays.

        This is a convenience factory for bulk construction (e.g. inside a
        file reader) where each scalar attribute is already stored as a
        flat array and the point clouds are available as a sequence of
        arrays.

        Parameters
        ----------
        ids : array-like of int, shape (N,)
            Unique particle identifiers.
        parent_ids : array-like of int, shape (N,)
            Direct parent particle IDs.
        root_ids : array-like of int, shape (N,)
            Root ancestor IDs.
        pdgs : array-like of int, shape (N,)
            PDG Monte Carlo codes.
        parent_pdgs : array-like of int, shape (N,)
            PDG codes of direct parents.
        interaction_types : array-like of int, shape (N,)
            Raw interaction process codes forwarded to
            :func:`~pysupera.utils.SetSemanticType`.
        point_clouds : sequence of numpy.ndarray, length N
            One array per particle; each has shape ``(M_i, ≥3)`` where
            *M_i* is the number of points for particle *i*.
        min_pc_size : int or None, optional
            Applied uniformly to all particles (forwarded to
            :class:`Particle.__init__`).  ``None`` (default) reads the
            module-level :data:`_DEFAULT_MIN_PC_SIZE`.

        Returns
        -------
        list of Particle
            Length-*N* list of fully constructed :class:`Particle` objects.

        Raises
        ------
        ValueError
            If the lengths of the scalar arrays and *point_clouds* are not
            all equal.

        Examples
        --------
        >>> import numpy as np
        >>> pcs = [np.zeros((10, 4)), np.zeros((5, 4))]
        >>> particles = Particle.from_arrays(
        ...     ids=[0, 1],
        ...     parent_ids=[0, 0],
        ...     root_ids=[0, 0],
        ...     pdgs=[11, 22],
        ...     parent_pdgs=[0, 11],
        ...     interaction_types=[0, 7],
        ...     point_clouds=pcs,
        ... )
        >>> len(particles)
        2
        """
        ids               = list(ids)
        parent_ids        = list(parent_ids)
        root_ids          = list(root_ids)
        pdgs              = list(pdgs)
        parent_pdgs       = list(parent_pdgs)
        interaction_types = list(interaction_types)
        point_clouds      = list(point_clouds)

        n = len(ids)
        if not all(len(a) == n for a in (parent_ids, root_ids, pdgs,
                                          parent_pdgs, interaction_types,
                                          point_clouds)):
            raise ValueError(
                "All input arrays must have the same length. "
                f"Got lengths: ids={len(ids)}, parent_ids={len(parent_ids)}, "
                f"root_ids={len(root_ids)}, pdgs={len(pdgs)}, "
                f"parent_pdgs={len(parent_pdgs)}, "
                f"interaction_types={len(interaction_types)}, "
                f"point_clouds={len(point_clouds)}."
            )

        return [
            cls(
                id=ids[i],
                parent_id=parent_ids[i],
                root_id=root_ids[i],
                pdg=pdgs[i],
                parent_pdg=parent_pdgs[i],
                interaction_type=interaction_types[i],
                point_cloud=point_clouds[i],
                min_pc_size=min_pc_size,
            )
            for i in range(n)
        ]

    @classmethod
    def from_flat_arrays(
        cls,
        ids,
        parent_ids,
        root_ids,
        pdgs,
        parent_pdgs,
        interaction_types,
        point_cloud_flat,
        point_cloud_offsets,
        min_pc_size=None,
    ):
        """
        Construct a list of :class:`Particle` objects from columnar arrays
        where all point clouds are stored as a single flat array together
        with a per-particle offset array.

        The point cloud for particle *i* is recovered as::

            point_cloud_flat[point_cloud_offsets[i, 0] : point_cloud_offsets[i, 1]]

        Parameters
        ----------
        ids : array-like of int, shape (N,)
            Unique particle identifiers.
        parent_ids : array-like of int, shape (N,)
            Direct parent particle IDs.
        root_ids : array-like of int, shape (N,)
            Root ancestor IDs.
        pdgs : array-like of int, shape (N,)
            PDG Monte Carlo codes.
        parent_pdgs : array-like of int, shape (N,)
            PDG codes of direct parents.
        interaction_types : array-like of int, shape (N,)
            Raw interaction process codes forwarded to
            :func:`~pysupera.utils.SetSemanticType`.
        point_cloud_flat : numpy.ndarray, shape (M, ≥3)
            Concatenated point clouds for all *N* particles.  Must be a
            plain (non-structured) 2-D ``float32`` array.  Columns 0–2
            are x, y, z; additional columns (time, energy, dEdx, …)
            are carried through unchanged.

            .. note::
                File-format-specific readers (e.g.
                :class:`~pysupera.readers.EDepSimHDF5Reader`) are
                responsible for converting their native structured arrays
                into this layout before calling :meth:`from_flat_arrays`.
        point_cloud_offsets : array-like of int, shape (N, 2)
            Row slices into *point_cloud_flat*.
            ``point_cloud_offsets[i] = [start, end]`` such that
            ``point_cloud_flat[start:end]`` is the point cloud of
            particle *i*.
        min_pc_size : int or None, optional
            Applied uniformly to all particles.  ``None`` (default)
            reads the module-level :data:`_DEFAULT_MIN_PC_SIZE`.

        Returns
        -------
        list of Particle

        Raises
        ------
        ValueError
            If the lengths of the scalar arrays and the first dimension of
            *point_cloud_offsets* are not all equal, or if
            *point_cloud_offsets* does not have shape ``(N, 2)``.

        Examples
        --------
        >>> import numpy as np
        >>> flat = np.zeros((15, 4))        # 15 points total
        >>> offsets = np.array([[0, 10],    # particle 0: 10 points
        ...                     [10, 15]])  # particle 1:  5 points
        >>> particles = Particle.from_flat_arrays(
        ...     ids=[0, 1],
        ...     parent_ids=[0, 0],
        ...     root_ids=[0, 0],
        ...     pdgs=[11, 22],
        ...     parent_pdgs=[0, 11],
        ...     interaction_types=[0, 7],
        ...     point_cloud_flat=flat,
        ...     point_cloud_offsets=offsets,
        ... )
        >>> len(particles)
        2
        >>> particles[0].point_cloud.shape
        (10, 4)
        """
        ids               = list(ids)
        parent_ids        = list(parent_ids)
        root_ids          = list(root_ids)
        pdgs              = list(pdgs)
        parent_pdgs       = list(parent_pdgs)
        interaction_types = list(interaction_types)
        offsets           = np.asarray(point_cloud_offsets)

        n = len(ids)

        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError(
                f"point_cloud_offsets must have shape (N, 2), "
                f"got {offsets.shape}."
            )
        if not all(len(a) == n for a in (parent_ids, root_ids, pdgs,
                                          parent_pdgs, interaction_types)):
            raise ValueError(
                "All scalar arrays must have the same length. "
                f"Got lengths: ids={len(ids)}, parent_ids={len(parent_ids)}, "
                f"root_ids={len(root_ids)}, pdgs={len(pdgs)}, "
                f"parent_pdgs={len(parent_pdgs)}, "
                f"interaction_types={len(interaction_types)}."
            )
        if offsets.shape[0] != n:
            raise ValueError(
                f"point_cloud_offsets has {offsets.shape[0]} rows but "
                f"scalar arrays have length {n}."
            )

        flat = np.asarray(point_cloud_flat)
        if flat.dtype.names is not None:
            raise TypeError(
                "point_cloud_flat must be a plain 2-D array, not a structured "
                "array.  Convert named fields to columns before calling "
                "from_flat_arrays (e.g. in the file-format reader)."
            )

        return [
            cls(
                id=ids[i],
                parent_id=parent_ids[i],
                root_id=root_ids[i],
                pdg=pdgs[i],
                parent_pdg=parent_pdgs[i],
                interaction_type=interaction_types[i],
                point_cloud=flat[offsets[i, 0] : offsets[i, 1]],
                min_pc_size=min_pc_size,
            )
            for i in range(n)
        ]

    
    def __str__(self):
        """Return a human-readable summary of this particle."""
        lines = [
            f"Particle(id={self.id}, parent_id={self.parent_id}, "
            f"root_id={self.root_id}",
            f"  pdg={self.pdg}, parent_pdg={self.parent_pdg}, "
            f"root_pdg={self.root_pdg}",
            f"  sem_type={self.sem_type}, interaction_type={self.interaction_type}",
            f"  point_cloud: shape={self.point_cloud.shape}",
        ]
        if self.start is not None:
            lines.append(f"  start={self.start}, end={self.end}")
        if not np.isnan(self.kinetic_energy_start):
            lines.append(
                f"  ke_start={self.kinetic_energy_start:.4g} MeV, "
                f"ke_end={self.kinetic_energy_end:.4g} MeV, "
                f"mass={self.mass:.4g} MeV/c²"
            )
        if self.start_process_name is not None:
            lines.append(
                f"  creation: {self.start_process_name} "
                f"(id={self.start_process_id}, sub={self.start_subprocess_id})"
            )
        return "\n".join(lines) + "\n"

    def __repr__(self):
        """
        Return the canonical string representation (same as ``__str__``).

        Returns
        -------
        str
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Equality is defined by particle *id* alone.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id == other.id

    def __hash__(self):
        """
        Hash based on particle *id* so particles can be stored in sets/dicts.

        Returns
        -------
        int
        """
        return hash(self.id)

    def __lt__(self, other):
        """
        Less-than comparison by particle *id*.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id < other.id

    def __gt__(self, other):
        """
        Greater-than comparison by particle *id*.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id > other.id

    def __le__(self, other):
        """
        Less-than-or-equal comparison by particle *id*.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id <= other.id

    def __ge__(self, other):
        """
        Greater-than-or-equal comparison by particle *id*.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id >= other.id

    def __ne__(self, other):
        """
        Inequality comparison by particle *id*.

        Parameters
        ----------
        other : Particle

        Returns
        -------
        bool
        """
        return self.id != other.id

    def __len__(self):
        """
        Return the number of points in the point cloud.

        Returns
        -------
        int
            ``point_cloud.shape[0]``.
        """
        return len(self.point_cloud)


# ============================================================================
# Point-cloud defragmentation — convenience wrapper
# ============================================================================

def defragment_particles(particles, distance_threshold: float, min_pc_size: int):
    """
    Split fragmented particle point clouds into separate particles.

    Thin convenience wrapper around
    :class:`~pysupera.preproc.ScipyDefragmenter` for call-sites that do
    not need a configurable backend.  See
    :mod:`pysupera.preproc` for full documentation and alternative GPU
    backends (``GPUDefragmenter``, ``RAPIDSDefragmenter``).

    Parameters
    ----------
    particles : list of Particle
        Input particle collection for a single event.
    distance_threshold : float
        Neighbourhood radius (same units as point-cloud coordinates).
    min_pc_size : int
        Fragments with ``size <= min_pc_size`` are split off as new
        kLEScatter particles.

    Returns
    -------
    list of Particle
    """
    from .preproc import ScipyDefragmenter
    return ScipyDefragmenter(distance_threshold, min_pc_size).process(particles)
