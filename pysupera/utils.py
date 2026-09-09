from enum import Enum, auto


class InteractionType(Enum):
    """
    Enumeration of the physics processes that can produce a simulation particle.

    The integer value of each member is assigned automatically (1-based).
    :func:`SetSemanticType` converts a raw integer process code to an
    ``InteractionType`` via ``InteractionType(interaction_type)``.

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
    dx : int = 5
        Additional scalar feature (e.g. a second energy-like quantity) summed
        over all hits that merge into the same output point.
    id : int = 6
        (optional) integer hit ID, e.g. for matching to truth-level information.
    """
    x      = 0
    y      = 1
    z      = 2
    time   = 3
    energy = 4
    dx     = 5
    id     = 6

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


def resolve_orphans(particles, verbose: bool = False):
    """
    Repair dangling ``parent_id`` and ``root_id`` references in a particle list
    so that every genealogy walk terminates at a particle that is actually
    present in the event.

    An *orphan* is a particle whose ``parent_id`` (or ``root_id``) refers to a
    particle ID that does not appear in *particles*.  Such dangling references
    cause :meth:`~pysupera.partitioner.ParticlePartitioner._build_relationships`
    to build an incomplete ``children_map`` / ``ancestor_map``, and they also
    break the ``find_initiator`` walk inside
    :func:`~pysupera.merge.merge_em_showers`.

    The repair rule is conservative:

    * If ``p.parent_id`` is not in the known-ID set, reset it to ``p.id``
      (i.e. make the particle a primary/root with respect to this event list).
    * If ``p.root_id`` is not in the known-ID set, reset it to ``p.id``
      as well.

    This function modifies *particles* **in place** and returns the same list.

    Parameters
    ----------
    particles : list of Particle
        The full particle collection for one event.  Modified in place.
    verbose : bool, optional
        If ``True``, print a summary of how many references were repaired.

    Returns
    -------
    list of Particle
        The same list (modified in place) for convenient chaining.

    Examples
    --------
    >>> particles = resolve_orphans(particles)
    >>> partitioner = ParticlePartitioner(particles=particles, ...)
    """
    known_ids = {p.id for p in particles}
    n_parent_fixed = 0
    n_root_fixed   = 0
    for p in particles:
        if p.parent_id not in known_ids:
            p.parent_id = p.id
            n_parent_fixed += 1
        if p.root_id not in known_ids:
            p.root_id = p.id
            n_root_fixed += 1
    if verbose and (n_parent_fixed or n_root_fixed):
        print(f"[resolve_orphans] repaired {n_parent_fixed} parent_id(s) "
              f"and {n_root_fixed} root_id(s) out of {len(particles)} particles")
    return particles


def SetSemanticType(interaction_type, pdg, parent_pdg, point_cloud, point_cloud_size=-1):
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
    interaction_type : int or InteractionType
        Raw integer process code or ``InteractionType`` enum member.
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
        If *interaction_type* maps to
        ``kPhoton`` / ``kConversion`` / ``kCompton`` / ``kOtherShower``
        but *pdg* is not ±11 or ±22, or if an entirely unrecognised
        ``InteractionType`` is encountered.
    """
    if not isinstance(interaction_type, InteractionType):
        interaction_type = InteractionType(interaction_type)
    if interaction_type == InteractionType.kInvalidProcess:
        return SemanticType.kUnknown
        raise Exception("'kInvalidProcess' particle process encountered\n")
    elif interaction_type == InteractionType.kTrack:
        if point_cloud.shape[0] < point_cloud_size:
            return SemanticType.kLEScatter
        else:
            return SemanticType.kTrack
        
    elif interaction_type == InteractionType.kPrimary:
        if abs(pdg) != 11 and abs(pdg) != 22:
            return SemanticType.kTrack
        else:
            return SemanticType.kShower
        
    elif interaction_type == InteractionType.kDelta:
        if point_cloud.shape[0] < point_cloud_size:
            return SemanticType.kLEScatter
        else:
            return SemanticType.kDelta
        
    elif interaction_type == InteractionType.kDecay:
        if abs(pdg) == 11 and abs(parent_pdg) == 13:
            return SemanticType.kMichel
        elif abs(pdg) in [11,22]:
            return SemanticType.kShower
        else:
            return SemanticType.kTrack
        
    elif interaction_type in [InteractionType.kNeutron, InteractionType.kIonization, InteractionType.kPhotoElectron]:
        return SemanticType.kLEScatter
        
    elif interaction_type in [InteractionType.kPhoton, InteractionType.kConversion, InteractionType.kCompton, InteractionType.kOtherShower]:
        if abs(pdg) in [11,22]:
            if point_cloud.shape[0] > point_cloud_size:
                return SemanticType.kShower
            else:
                return SemanticType.kLEScatter
        else:
            raise Exception("kPhoton/kConversion/kCompton/kOtherShower encountered but PDG ("+str(pdg)+") not 11/22, InteractionType ("+str(interaction_type)+")\n")
    elif interaction_type == InteractionType.kNucleus:
        if point_cloud.shape[0] > point_cloud_size:
            return SemanticType.kTrack
        else:
            return SemanticType.kLEScatter
        
    else:
        raise Exception("Unexpected interaction type ("+str(interaction_type)+") encountered")


def validate_interaction_ids(particles,
                             raise_on_missing: bool = True,
                             max_report: int = 10,
                             verbose: bool = False) -> list:
    """
    Check that every particle in *particles* carries an interaction ID.

    A particle with ``_interaction_id < 0`` was never associated with any
    vertex.  That is almost always a genealogy problem rather than physics:
    :func:`~pysupera.readers.format_edepsim_h5._get_interaction_id` seeds
    interaction IDs from particles whose start position coincides with a
    vertex and then propagates them down the tree, so a broken or
    self-referential ancestor link strands a particle *and every descendant of
    it* at -1.  Because the strand is silent, the loss is easy to miss until
    something downstream indexes by interaction.

    For each offender the diagnostic walks ``parent_id`` upwards and reports
    the chain with each ancestor's PDG and interaction ID, which is normally
    enough to see where the assignment stopped.

    Parameters
    ----------
    particles : list of Particle
        Particle collection for one event.  Not modified.
    raise_on_missing : bool, optional
        ``True`` (default) raises :class:`ValueError` when any particle lacks
        an interaction ID.  ``False`` issues a :class:`UserWarning` instead and
        lets the caller decide.
    max_report : int, optional
        Number of offending particles described in detail.  Default ``10``.
    verbose : bool, optional
        Print a one-line confirmation when every particle is assigned.

    Returns
    -------
    list of Particle
        The offending particles, empty when all are assigned.

    Raises
    ------
    ValueError
        If *raise_on_missing* and at least one particle lacks an ID.
    """
    missing = [p for p in particles if int(getattr(p, '_interaction_id', -1)) < 0]

    if not missing:
        if verbose:
            print(f"[validate] all {len(particles)} particle(s) carry an "
                  f"interaction ID")
        return []

    by_id = {int(p.id): p for p in particles}

    def _chain(p, limit: int = 12) -> str:
        """Render the parent chain of *p* as id(pdg)=interaction_id links."""
        out, seen, cur = [], set(), p
        while cur is not None and int(cur.id) not in seen and len(out) < limit:
            seen.add(int(cur.id))
            out.append(f"{int(cur.id)}(pdg={int(cur.pdg)})"
                       f"=int{int(getattr(cur, '_interaction_id', -1))}")
            nxt = int(cur.parent_id)
            cur = None if nxt == int(cur.id) else by_id.get(nxt)
        return " <- ".join(out) + (" <- ..." if len(out) >= limit else "")

    lines = [_chain(p) for p in missing[:max_report]]
    msg = (
        f"{len(missing)} of {len(particles)} particle(s) have no interaction ID "
        f"(_interaction_id < 0).  Every particle must be associated with an "
        f"interaction.  Parent chains of the first {len(lines)}:\n  "
        + "\n  ".join(lines)
    )
    if len(missing) > len(lines):
        msg += f"\n  ... and {len(missing) - len(lines)} more"

    if raise_on_missing:
        raise ValueError(msg)

    import warnings
    warnings.warn(msg, UserWarning, stacklevel=2)
    return missing


def validate_root_ids(particles,
                      raise_on_missing: bool = True,
                      max_report: int = 10,
                      verbose: bool = False) -> list:
    """
    Check that every particle's ``root_id`` is the primary it descends from.

    The invariant checked is that walking ``parent_id`` upwards from a particle
    terminates at exactly ``root_id``.  It is worth checking because the
    ancestor field in EDepSim files is not always consistent with the parent
    links: a secondary written with ``ancestor_track_id == its own track_id``
    becomes a false root, and every descendant inherits it, so a handful of bad
    entries can put a third of an event on the wrong root.  See
    :func:`~pysupera.readers.format_edepsim_h5._get_root_id`, which derives
    ``root_id`` from the parent chain for this reason.

    Two failure modes are reported: a ``root_id`` naming a particle absent from
    the event, and a ``root_id`` that disagrees with the primary the parent
    chain actually leads to.

    Run this **after** :func:`resolve_orphans`, which legitimately re-roots
    particles whose references dangle; before that, its repairs look like
    violations.

    Parameters
    ----------
    particles : list of Particle
        Particle collection for one event.  Not modified.
    raise_on_missing : bool, optional
        ``True`` (default) raises :class:`ValueError`; ``False`` warns instead.
    max_report : int, optional
        Number of offenders described in detail.  Default ``10``.
    verbose : bool, optional
        Print a one-line confirmation when every root is consistent.

    Returns
    -------
    list of Particle
        The offending particles, empty when all roots are consistent.

    Raises
    ------
    ValueError
        If *raise_on_missing* and at least one root is inconsistent.
    """
    by_id = {int(p.id): p for p in particles}

    def _walk_primary(p):
        """Primary reached from *p* via parent_id; own id on a cycle."""
        seen = set()
        cur = p
        while True:
            cid = int(cur.id)
            if cid in seen:
                return cid                      # cycle: no primary reachable
            seen.add(cid)
            pid = int(cur.parent_id)
            if pid == cid or pid not in by_id:
                return cid
            cur = by_id[pid]

    offenders, reasons = [], []
    for p in particles:
        rid = int(p.root_id)
        if rid not in by_id:
            offenders.append(p)
            reasons.append(f"{int(p.id)}(pdg={int(p.pdg)}): root_id={rid} "
                           f"is not a particle in this event")
            continue
        primary = _walk_primary(p)
        if primary != rid:
            offenders.append(p)
            reasons.append(
                f"{int(p.id)}(pdg={int(p.pdg)}): root_id={rid} but the "
                f"parent chain leads to {primary}"
                f"(pdg={int(by_id[primary].pdg)})"
            )

    if not offenders:
        if verbose:
            print(f"[validate] all {len(particles)} particle(s) have a "
                  f"consistent root_id")
        return []

    shown = reasons[:max_report]
    msg = (f"{len(offenders)} of {len(particles)} particle(s) have an "
           f"inconsistent root_id.  First {len(shown)}:\n  "
           + "\n  ".join(shown))
    if len(reasons) > len(shown):
        msg += f"\n  ... and {len(reasons) - len(shown)} more"

    if raise_on_missing:
        raise ValueError(msg)

    import warnings
    warnings.warn(msg, UserWarning, stacklevel=2)
    return offenders


def select_traceable_instances(instances, particles, verbose: bool = False):
    """
    Reduce an instance list to those needed to describe the visible event.

    An event's instance list contains many representatives that deposited
    nothing: neutral particles, absorbed shower members, recoil nuclei.  Most
    are of no interest, but a few are indispensable -- a pi0 deposits no energy
    yet is the only link between its two visible photons, so dropping it breaks
    the production history of particles that *are* visible.

    Three rules decide what stays:

    1. Every instance with a non-empty point cloud (*visible*).
    2. Every ancestor of a visible instance, so each kept row's parent is also
       kept and the genealogy is closed.  These are the zero-point links.
    3. The primaries of every interaction that has at least one visible
       instance, even without visible descendants of their own, so the primary
       list of a detected interaction stays complete.  An interaction with no
       visible instance at all is dropped entirely -- it left no trace, so
       nothing represents it.

    Ancestry is resolved through the particle ``parent_id`` chain rather than
    the instance list, skipping ancestors that belong to the instance's *own*
    group: when a rep's parent was merged into the rep's own shower, the naive
    lookup answers with the rep itself and the walk dead-ends.

    Parameters
    ----------
    instances : list of Particle
        Instance representatives, e.g. from
        :func:`~pysupera.merge.merge_em_showers`.  Not modified.
    particles : list of Particle
        The event's full particle list, used for the ``parent_id`` chains.
    verbose : bool, optional
        Print a one-line summary of what was kept.

    Returns
    -------
    list of Particle
        The retained instances, in their original relative order.
    """
    if not instances:
        return list(instances)

    by_id = {int(p.id): p for p in particles}

    # member particle id -> index in *instances*
    inst_of: dict = {}
    for k, inst in enumerate(instances):
        members = inst.member_ids if inst.member_ids is not None else [inst.id]
        for pid in members:
            inst_of[int(pid)] = k

    def _parent_index(k):
        """Nearest ancestor instance of *k*, or None when *k* is a primary."""
        inst = instances[k]
        pid = int(inst.parent_id)
        seen = set()
        while pid not in seen:
            seen.add(pid)
            hit = inst_of.get(pid)
            if hit is not None and hit != k:
                return hit
            p = by_id.get(pid)
            if p is None or int(p.parent_id) == int(p.id):
                return None
            pid = int(p.parent_id)
        return None

    parent_idx = [_parent_index(k) for k in range(len(instances))]

    visible = {k for k, i in enumerate(instances) if len(i.point_cloud) > 0}
    visible_interactions = {int(instances[k]._interaction_id) for k in visible}

    keep = set(visible)
    for k in visible:                       # rule 2: walk to the root
        cur = parent_idx[k]
        while cur is not None and cur not in keep:
            keep.add(cur)
            cur = parent_idx[cur]

    for k in range(len(instances)):         # rule 3: primaries of live interactions
        if parent_idx[k] is None and \
                int(instances[k]._interaction_id) in visible_interactions:
            keep.add(k)

    out = [instances[k] for k in range(len(instances)) if k in keep]
    if verbose:
        print(f"[instances] kept {len(out)} of {len(instances)} "
              f"({len(visible)} visible, {len(out) - len(visible)} genealogy/primary), "
              f"{len(visible_interactions)} interaction(s) with signal")
    return out
