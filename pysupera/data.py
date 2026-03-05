from .utils import InteractionType, SemanticType, SetSemanticType
import numpy as np

# ---------------------------------------------------------------------------
# Module-level defaults.
# Set these via pysupera.config.configure(cfg) after loading Hydra config.
# ---------------------------------------------------------------------------
_DEFAULT_MIN_PC_SIZE: int = -1


# ============================================================================
# Particle Data Class
# ============================================================================

class Particle:
    """
    Simulation particle with physical metadata and a 3-D point cloud.

    Each particle stores the scalar identifiers produced by the upstream
    simulation (PDG code, ancestry chain, interaction process) together with
    the spatial hit collection associated with that particle.  The semantic
    type is derived automatically from the stored fields via
    :func:`~pysupera.utils.SetSemanticType` at construction time.

    Attributes
    ----------
    id : int
        Unique integer identifier for this particle within an event.
    parent_id : int
        ID of the direct parent particle.  Equals ``id`` for primary
        particles that have no parent.
    ancestor_id : int
        ID of the root ancestor particle in the shower/track genealogy.
    pdg : int
        PDG Monte Carlo numbering scheme code.
        See https://pdg.lbl.gov/2023/mcdata/mc_particle_id_contents.html.
    parent_pdg : int
        PDG code of the direct parent particle.
    sem_type : SemanticType
        Semantic category derived from *process_type*, *pdg*, *parent_pdg*,
        and *point_cloud* by :func:`~pysupera.utils.SetSemanticType`.
        One of ``kShower``, ``kTrack``, ``kDelta``, ``kMichel``,
        ``kLEScatter``, or ``kUnknown``.
    point_cloud : numpy.ndarray
        Array of shape ``(N, ≥3)`` containing the 3-D spatial hits
        (columns 0–2 are x, y, z; additional columns such as time and
        energy deposit may be present).
    """

    def __init__(self, id, parent_id, ancestor_id, pdg, parent_pdg,
                 process_type, point_cloud, min_pc_size=None):
        """
        Parameters
        ----------
        id : int
            Unique particle identifier within an event.
        parent_id : int
            ID of the direct parent particle.
        ancestor_id : int
            ID of the root ancestor particle.
        pdg : int
            PDG Monte Carlo particle code for this particle.
        parent_pdg : int
            PDG code of the parent particle.
        process_type : int
            Raw integer encoding of the interaction process that produced
            this particle (0-based index into :class:`~pysupera.utils.InteractionType`).
            Stored as ``_process_type`` for lossless serialisation; the
            derived ``sem_type`` is computed from it.
        point_cloud : numpy.ndarray
            Spatial hit array of shape ``(N, ≥3)``.
        min_pc_size : int or None, optional
            Minimum point-cloud size threshold forwarded to
            :func:`~pysupera.utils.SetSemanticType` as
            ``point_cloud_size``.  Particles with fewer than
            *min_pc_size* points may receive a different semantic type
            than larger ones (e.g. ``kLEScatter`` instead of ``kDelta``
            or ``kShower``).  ``None`` (default) reads the module-level
            :data:`_DEFAULT_MIN_PC_SIZE`, which is ``-1`` unless
            overridden by :func:`~pysupera.config.configure`.
        """
        self.id = id
        self.parent_id = parent_id
        self.ancestor_id = ancestor_id
        self.pdg = pdg
        self.parent_pdg = parent_pdg
        self._process_type = process_type  # raw int; retained for serialisation
        if min_pc_size is None:
            min_pc_size = _DEFAULT_MIN_PC_SIZE
        self.sem_type = SetSemanticType(process_type, pdg, parent_pdg, point_cloud, point_cloud_size=min_pc_size)
        self.point_cloud = point_cloud  # numpy array of shape (N, 4)

    def __str__(self):
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            ``Particle(id=..., parent_id=..., ancestor_id=..., pdg=...,
            parent_pdg=..., sem_type=..., point_cloud=...)``.
        """
        return (f"Particle(id={self.id}, parent_id={self.parent_id}, "
                f"ancestor_id={self.ancestor_id}\npdg={self.pdg}, "
                f"parent_pdg={self.parent_pdg}, sem_type={self.sem_type}\n"
                f"point_cloud: shape={self.point_cloud.shape}\n")
                #f"point_cloud: shape={self.point_cloud.shape}, energy sum {self.point_cloud[:,3].sum()} average {self.point_cloud[:,3].mean()}\n" 
                #f"first point: {self.point_cloud[0][:3]}, last point: {self.point_cloud[-1][:3]}\n")

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
