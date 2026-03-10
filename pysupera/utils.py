from enum import Enum, auto


class InteractionType(Enum):
    """
    Enumeration of the physics processes that can produce a simulation particle.

    The integer value of each member is assigned automatically (1-based).
    :func:`SetSemanticType` converts a raw integer process code to an
    ``InteractionType`` via ``InteractionType(process_type)``.

    Members
    -------
    kTrack
        Charged-particle tracking process (ionising track).
    kNeutron
        Neutron scatter / capture.
    kNucleus
        Nuclear recoil.
    kPhoton
        Photon interaction (used with kConversion/kCompton sub-types).
    kPrimary
        Primary particle from the generator vertex.
    kCompton
        Compton scattering of a photon.
    kDelta
        Delta-ray (secondary electron from ionisation).
    kConversion
        Pair conversion of a photon.
    kIonization
        Ionisation deposit that is not classified as a delta ray.
    kPhotoElectron
        Photo-electric effect.
    kDecay
        Particle decay (e.g. muon → Michel electron).
    kOtherShower
        Any remaining electromagnetic shower process.
    kInvalidProcess
        Sentinel / unrecognised process code.
    """
    kTrack   = auto()
    kNeutron = auto()
    kNucleus = auto()
    kPhoton  = auto()
    kPrimary = auto()
    kCompton = auto()
    kDelta   = auto()
    kConversion     = auto()
    kIonization     = auto()
    kPhotoElectron  = auto()
    kDecay          = auto()
    kOtherShower    = auto()
    kInvalidProcess = auto()


class SemanticType(Enum):
    """
    High-level semantic category assigned to each simulation particle.

    Used by the partitioner to decide which particles should be merged
    and serves as the primary label for downstream reconstruction tasks.

    Members
    -------
    kShower
        Electromagnetic shower (electron or photon origin).
    kTrack
        Charged-particle track (muon, pion, proton, etc.).
    kDelta
        Delta ray — a short secondary electron from ionisation whose
        point cloud exceeds the *min_pc_size* threshold.
    kMichel
        Michel electron from a muon decay.
    kLEScatter
        Low-energy scatter deposit — small delta ray, neutron capture,
        photo-electron, or ionisation fragment below *min_pc_size*.
    kUnknown
        Unclassified particle (e.g. ``kInvalidProcess`` input).
    """
    kShower    = auto()
    kTrack     = auto()
    kDelta     = auto()
    kMichel    = auto()
    kLEScatter = auto()
    kUnknown   = auto()


class PointFeature(int):
    """
    Named column indices for point-cloud feature arrays.

    Each :class:`~pysupera.data.Particle` stores its hits as a 2-D
    NumPy array of shape ``(N, F)`` where ``F \u2265 3``.  This class maps
    column positions to human-readable names so that downstream code can
    write ``pc[:, PointFeature.energy]`` instead of ``pc[:, 4]``.

    Because ``PointFeature`` subclasses ``int``, instances are valid NumPy
    integer indices and can be used directly in array subscripts without
    calling ``.value``.

    Attributes
    ----------
    x : int = 0
        x-coordinate of the 3-D hit position.
    y : int = 1
        y-coordinate.
    z : int = 2
        z-coordinate.
    time : int = 3
        Hit time.
    energy : int = 4
        Energy deposited at the hit.
    dedx : int = 5
        Ionisation energy loss (dE/dx) at the hit.
    """
    x      = 0
    y      = 1
    z      = 2
    time   = 3
    energy = 4
    dedx   = 5

    def __new__(cls, value):
        return int.__new__(cls, value)

    @classmethod
    def name_of(cls, index: int) -> str:
        """
        Return the attribute name for a column index, or ``'col<N>'`` if
        the index is beyond the defined range.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
        """
        for attr, val in cls.__dict__.items():
            if isinstance(val, int) and not attr.startswith('_') and val == index:
                return attr
        return f'col{index}'


def trace_ancestry(particle_id: int, particles, print_result: bool = True):
    """
    Trace the full ancestry chain for a given particle ID.

    Walks the ``parent_id`` links from the requested particle up to the
    primary root (the particle whose ``parent_id == id``), then collects
    all direct children of the target.  Optionally prints a formatted
    summary table.

    Parameters
    ----------
    particle_id : int
        The ID of the particle to trace.
    particles : list of Particle
        All particles in the event.
    print_result : bool, optional
        If ``True`` (default), print the ancestry chain and children to
        stdout.

    Returns
    -------
    chain : list of Particle
        Ancestry chain ordered from root to target (inclusive).
    children : list of Particle
        Direct children of the target particle (may be empty).

    Raises
    ------
    KeyError
        If *particle_id* is not found in *particles*.
    """
    lookup = {p.id: p for p in particles}

    if particle_id not in lookup:
        raise KeyError(f"Particle id={particle_id} not found in particle list "
                       f"({len(particles)} particles).")

    # ── Walk parent_id chain upward until we reach the root ──────────────
    chain = []
    visited = set()
    current = lookup[particle_id]
    while True:
        chain.append(current)
        if current.id in visited:
            # Cycle guard — shouldn't happen in valid MC, but be safe.
            break
        visited.add(current.id)
        if current.parent_id == current.id:
            break   # primary root: parent points to itself
        if current.parent_id not in lookup:
            break   # orphan — parent missing from this particle list
        current = lookup[current.parent_id]

    chain.reverse()   # root → ... → target

    # ── Direct children of the target ────────────────────────────────────
    target = lookup[particle_id]
    children = [p for p in particles if p.parent_id == particle_id
                                     and p.id != particle_id]

    if print_result:
        _fmt_particle = (
            "id={id:>6d}  pdg={pdg:>7d}  "
            "sem={sem:<12s}  parent={parent:>6d}  anc={anc:>6d}  "
            "pc={pc:>6d}"
        )

        def _row(p, tag=""):
            return _fmt_particle.format(
                id=p.id, pdg=p.pdg,
                sem=p.sem_type.name,
                parent=p.parent_id, anc=p.root_id,
                pc=len(p.point_cloud),
            ) + (f"  [{tag}]" if tag else "")

        missing_parent = (
            chain[0].parent_id not in lookup
            and chain[0].parent_id != chain[0].id
        )
        root_tag  = "orphan root" if missing_parent else "root"
        n_gen     = len(chain)

        print(f"\nAncestry chain for particle {particle_id}  "
              f"({n_gen} generation(s))")
        print(f"  {'id':>8}  {'pdg':>9}  {'sem':<14}  {'parent':>12}  "
              f"{'anc':>9}  {'pc':>8}")
        print("  " + "-" * 74)

        for idx, p in enumerate(chain):
            if idx == 0:
                tag = root_tag
            elif idx == n_gen - 1:
                tag = "query"
            else:
                tag = ""
            label = f"[{tag}]" if tag else ""
            print(f"  {_row(p)}  {label}")

        if children:
            print(f"\n  Direct children of {particle_id}:")
            print("  " + "-" * 74)
            for ch in sorted(children, key=lambda p: p.id):
                print(f"  {_row(ch)}")
        else:
            print(f"\n  (no direct children of {particle_id} in this particle list)")

    return chain, children


def SetSemanticType(process_type, pdg, parent_pdg, point_cloud, point_cloud_size=-1):
    """
    Derive the :class:`SemanticType` of a particle from its physics properties.

    The classification follows a rule-based decision tree that mirrors the
    standard LArTPC reconstruction labelling convention:

    * ``kInvalidProcess`` → ``kUnknown``
    * ``kTrack`` → ``kTrack``
    * ``kPrimary`` : electrons / photons → ``kShower``; others → ``kTrack``
    * ``kDelta`` : small cloud (< *point_cloud_size*) → ``kLEScatter``;
      large → ``kDelta``
    * ``kDecay`` : e⁻ from μ → ``kMichel``; e / γ → ``kShower``;
      others → ``kTrack``
    * ``kNeutron`` / ``kIonization`` / ``kPhotoElectron`` → ``kLEScatter``
    * ``kPhoton`` / ``kConversion`` / ``kCompton`` / ``kOtherShower`` :
      e / γ and large cloud → ``kShower``; small cloud → ``kLEScatter``
    * ``kNucleus`` : large cloud → ``kTrack``; small → ``kLEScatter``

    Parameters
    ----------
    process_type : int
        Raw integer process code (0-based).  Internally converted via
        ``InteractionType(process_type + 1)``.
    pdg : int
        PDG Monte Carlo particle code for this particle.
    parent_pdg : int
        PDG code of the parent particle.
    point_cloud : numpy.ndarray
        Spatial hit array of shape ``(N, ≥1)``.  Only ``point_cloud.shape[0]``
        (the number of rows) is used.
    point_cloud_size : int, optional
        Threshold for the small / large cloud split.  A point cloud is
        considered *small* if ``point_cloud.shape[0] < point_cloud_size``
        and *large* if ``point_cloud.shape[0] > point_cloud_size``.
        Pass ``-1`` (default) to make the threshold inactive so that all
        clouds are treated as large.

    Returns
    -------
    SemanticType
        The derived semantic category.

    Raises
    ------
    Exception
        If *process_type* maps to
        ``kPhoton`` / ``kConversion`` / ``kCompton`` / ``kOtherShower``
        but *pdg* is not ±11 or ±22, or if an entirely unrecognised
        ``InteractionType`` is encountered.
    """
 
    process_type = InteractionType(process_type)
    if process_type == InteractionType.kInvalidProcess:
        return SemanticType.kUnknown
        raise Exception("'kInvalidProcess' particle process encountered\n")
    elif process_type == InteractionType.kTrack:
        if point_cloud.shape[0] < point_cloud_size:
            return SemanticType.kLEScatter
        else:
            return SemanticType.kTrack
        
    elif process_type == InteractionType.kPrimary:
        if abs(pdg) != 11 and abs(pdg) != 22:
            return SemanticType.kTrack
        else:
            return SemanticType.kShower
        
    elif process_type == InteractionType.kDelta:
        if point_cloud.shape[0] < point_cloud_size:
            return SemanticType.kLEScatter
        else:
            return SemanticType.kDelta
        
    elif process_type == InteractionType.kDecay:
        if abs(pdg) == 11 and abs(parent_pdg) == 13:
            return SemanticType.kMichel
        elif abs(pdg) in [11,22]:
            return SemanticType.kShower
        else:
            return SemanticType.kTrack
        
    elif process_type in [InteractionType.kNeutron, InteractionType.kIonization, InteractionType.kPhotoElectron]:
        return SemanticType.kLEScatter
        
    elif process_type in [InteractionType.kPhoton, InteractionType.kConversion, InteractionType.kCompton, InteractionType.kOtherShower]:
        if abs(pdg) in [11,22]:
            if point_cloud.shape[0] > point_cloud_size:
                return SemanticType.kShower
            else:
                return SemanticType.kLEScatter
        else:
            raise Exception("kPhoton/kConversion/kCompton/kOtherShower encountered but PDG ("+str(pdg)+") not 11/22, InteractionType ("+str(process_type)+")\n")
    elif process_type == InteractionType.kNucleus:
        if point_cloud.shape[0] > point_cloud_size:
            return SemanticType.kTrack
        else:
            return SemanticType.kLEScatter
        
    else:
        raise Exception("Unexpected interaction type ("+str(process_type)+") encountered")
