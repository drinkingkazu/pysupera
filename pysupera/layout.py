"""
Layout computation for the 3.0.0 on-disk format.

Format 3.0.0 replaces the three parallel particle tables of 2.x with a single
``particles`` table plus one flat point array.  A particle row carries its own
identity and, when it represents a group, that group's columns as well:

    frag_id / inst_id    the *particle id of the group's representative*.
                         A representative therefore satisfies ``frag_id == id``,
                         which is how the levels are enumerated.  ``-1`` means
                         the particle heads no group at that level.
    *_pc_start/end       the group's slice of the point array
    *_merge_count        Geant4 particles merged into the group

This module does the part that has to be right before any bytes are written:
deciding which particles to store, what order the points go in, and what those
ranges are.  It is deliberately free of HDF5 so it can be tested directly.

Two invariants it establishes
-----------------------------
*Nesting.*  Points are ordered by ``(interaction, instance, fragment,
particle)``.  Because instances partition into fragments and fragments into
particles, every level's points are then a single contiguous slice, so one
``(start, end)`` pair per level suffices and no per-point group labels are
needed.  That is checked on every event rather than assumed: each range must
hold exactly its members' own rows, or the layout raises
:class:`LayoutError`.  It once failed silently -- an empty fragment of an
unwritten instance, whose members sorted apart, claimed the rows of the
particles between them -- and it would again for a group whose members span
two interactions.

*Walkable genealogy.*  Only a fraction of Geant4 particles are stored, so a
stored particle's true parent may be absent.  ``parent_id`` is therefore
redirected to the nearest *stored* ancestor, and a particle with no stored
ancestor is marked a primary (``parent_id == id``).  Walking ``parent_id``
always terminates on a stored row rather than dangling.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .utils import SemanticType as _ST

#: Low-energy scatter.  Sorting on this splits each instance into a non-LE
#: block followed by an LE block; see build_layout.
_LE_SEM = _ST.kLEScatter


#: A group's identity captured at a point in time: which particle represents
#: it, which particles it contains, and how it was classified.
Group = namedtuple("Group", "rep_id member_ids sem_type")


def snapshot_groups(reps):
    """
    Capture the membership and classification of *reps* right now.

    This must be called at the correct moment, and the reason is worth
    spelling out: ``merge_em_showers`` mutates the surviving representatives
    **in place** and returns those same objects as the instances -- every
    instance object is also a fragment object.  So after step 2 a
    representative's ``member_ids`` and ``sem_type`` describe its *instance*,
    and the fragment values it used to hold are gone.  Reading fragment
    membership after the merge silently attributes most particles to the wrong
    fragment.

    Snapshot fragments before :func:`~pysupera.merge.merge_em_showers`, and
    instances after it.
    """
    return [Group(int(r.id),
                  [int(m) for m in (r.member_ids
                                    if r.member_ids is not None else [r.id])],
                  int(r.sem_type.value))
            for r in reps]


class EventLayout:
    """Result of :func:`build_layout` for one event."""

    __slots__ = ("order", "rows", "pc_offsets", "columns", "n_points",
                 "interactions")

    def __init__(self, order, rows, pc_offsets, columns, n_points,
                 interactions=None):
        #: every particle, in point order -- defines the point array
        self.order = order
        #: indices into *order* of the particles actually written
        self.rows = rows
        #: fencepost into the point array, one entry per particle in *order*
        self.pc_offsets = pc_offsets
        #: per-written-row arrays, keyed by column name
        self.columns = columns
        self.n_points = n_points
        #: per-interaction arrays; ``particles.interaction_id`` indexes these
        self.interactions = interactions or {}

    def __repr__(self):
        return (f"EventLayout(order={len(self.order)}, rows={len(self.rows)}, "
                f"points={self.n_points})")


def _group_of(groups):
    """Map every member particle id to its group's representative id."""
    out = {}
    for g in groups:
        for m in g.member_ids:
            out[m] = g.rep_id
    return out


def _offsets(order, attr):
    """Fencepost over the per-particle arrays named *attr*, in *order*."""
    n = np.fromiter((len(getattr(p, attr, None) if getattr(p, attr, None)
                         is not None else ()) for p in order),
                    dtype=np.int64, count=len(order))
    out = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(n, out=out[1:])
    return out


class LayoutError(ValueError):
    """A group range that would not describe exactly its own members' rows."""


def _group_ranges(offsets, pos, frag_groups, inst_groups, is_le):
    """
    Every group's slice of one cloud, given its per-particle *offsets*.

    A group's members *with rows* are contiguous under the storage order, so
    min(start) / max(end) over them describes the block exactly.  Members
    without rows are left out: they still occupy a position, and one of an
    unwritten instance's fragments sorts under its own id, so counting it
    would stretch the range over whatever particles sit in between -- rows
    the group does not own.  A group with no rows at all gets (-1, -1).
    Returns ``(frag_rng, inst_rng, frag_nle, frag_le, inst_split)``.
    """
    def _span(members):
        """Range covering *members*' rows, or (-1, -1) when they have none."""
        idx = [pos[m] for m in members
               if m in pos and offsets[pos[m] + 1] > offsets[pos[m]]]
        if not idx:
            return (-1, -1)
        lo, hi = min(idx), max(idx)
        return (int(offsets[lo]), int(offsets[hi + 1]))

    def whole(groups):
        rng = {}
        for g in groups:
            r = _span(g.member_ids)
            rng[g.rep_id] = (0, 0) if r == (-1, -1) else r
        return rng

    frag_rng = whole(frag_groups)
    inst_rng = whole(inst_groups)

    # Each side of a group is a single run under this ordering, so two pairs
    # describe a fragment completely -- never three.
    frag_nle = {g.rep_id: _span([m for m in g.member_ids if not is_le[m]])
                for g in frag_groups}
    frag_le  = {g.rep_id: _span([m for m in g.member_ids if is_le[m]])
                for g in frag_groups}
    # Where the LE block starts inside an instance; == its end when the
    # instance has no LE rows at all.
    inst_split = {}
    for g in inst_groups:
        le_span = _span([m for m in g.member_ids if is_le[m]])
        inst_split[g.rep_id] = (inst_rng[g.rep_id][1] if le_span[0] < 0
                                else le_span[0])
    _check_ranges(offsets, pos, frag_groups, inst_groups, is_le,
                  frag_nle, frag_le, inst_rng, inst_split)
    return frag_rng, inst_rng, frag_nle, frag_le, inst_split


def _check_ranges(offsets, pos, frag_groups, inst_groups, is_le,
                  frag_nle, frag_le, inst_rng, inst_split):
    """
    Raise unless every range holds exactly its own members' rows.

    Contiguity is what makes a (start, end) pair enough, and it rests on the
    storage order keeping each group together -- an assumption, not a law:
    a group whose members sort apart (members of two interactions, say)
    would get a range spanning other particles' rows.  Comparing each width
    with the members' own counts catches every such case, and costs one
    pass over data already in hand.
    """
    def own(members, le):
        return sum(int(offsets[pos[m] + 1] - offsets[pos[m]])
                   for m in members if m in pos and is_le[m] is le)

    def width(r):
        return r[1] - r[0] if r[0] >= 0 else 0

    for g in frag_groups:
        for le, r in ((False, frag_nle[g.rep_id]), (True, frag_le[g.rep_id])):
            if width(r) != own(g.member_ids, le):
                raise LayoutError(
                    f"fragment {g.rep_id}: {'LE' if le else 'non-LE'} range "
                    f"{r} holds {width(r)} rows but its members own "
                    f"{own(g.member_ids, le)} -- they are not contiguous in "
                    f"the storage order.")
    for g in inst_groups:
        (s, e), sp = inst_rng[g.rep_id], inst_split[g.rep_id]
        got = ((sp - s, e - sp) if e > s else (0, 0))
        want = (own(g.member_ids, False), own(g.member_ids, True))
        if got != want:
            raise LayoutError(
                f"instance {g.rep_id}: ranges hold {got} (non-LE, LE) rows "
                f"but its members own {want} -- they are not contiguous in "
                f"the storage order.")


def build_layout(particles, frag_groups, inst_groups, vertices=None,
                 keep=()):
    """
    Decide storage order, row selection and group ranges for one event.

    Parameters
    ----------
    particles : list of Particle
        Every particle in the event.  All of their points are stored, whether
        or not the particle itself gets a row -- the charge belongs to the
        event regardless of which representative owns it.
    frag_groups, inst_groups : list of Group
        Step-1 and step-2 groups from :func:`snapshot_groups`, taken before
        and after ``merge_em_showers`` respectively -- see that function for
        why the timing matters.  *inst_groups* should already be filtered to
        the instances being written; ordering still uses every fragment, so
        the nesting stays complete even when rows are dropped.
    vertices : structured ndarray or None
        The event's vertices, as the reader read them.  Interactions that no
        stored particle references are dropped and the survivors renumbered,
        so ``interaction_id`` is a direct row index into the interactions
        table rather than a value that has to be searched for.
    keep : iterable of int
        Particle ids that must get a row whatever their points -- the
        fragments and instances something else refers to (hit_labels).

    Returns
    -------
    EventLayout
    """
    by_id = {int(p.id): p for p in particles}
    frag_of = _group_of(frag_groups)
    inst_of = _group_of(inst_groups)
    kept_inst_ids = {g.rep_id for g in inst_groups}

    # ---- ordering: interaction > instance > fragment > particle ----------
    # LE-ness sorts *between* instance and fragment, so each instance becomes
    # a non-LE block followed by an LE block.  That makes "the non-LE points of
    # this instance" -- the query panoptic training issues most -- a single
    # contiguous slice instead of a mask, and removes the need for any
    # per-point label: LE-ness is implied by position relative to the split.
    # The cost is that a fragment's points become two runs rather than one.
    is_le = {int(p.id): (p.sem_type is _LE_SEM) for p in particles}

    # Points belong to the interaction of the group that owns them, not of
    # the particle that deposited them.  The two differ only when a
    # proximity merge (AbsorbLEScatter, CombineLEScatters) took a particle
    # from one interaction into a representative from another: its points
    # are then the representative's, and are stored with them.  The
    # particle itself keeps its own interaction -- see the row order below.
    def group_interaction(pid):
        rep = inst_of[pid] if pid in inst_of else frag_of.get(pid, pid)
        return int(by_id[rep if rep in by_id else pid]._interaction_id)

    point_inter = {int(p.id): group_interaction(int(p.id)) for p in particles}

    def sort_key(p):
        pid = int(p.id)
        return (point_inter[pid],
                inst_of.get(pid, pid),
                is_le[pid],
                frag_of.get(pid, pid),
                pid)

    order = sorted(particles, key=sort_key)
    pos = {int(p.id): k for k, p in enumerate(order)}

    pc_offsets = _offsets(order, "point_cloud")
    frag_rng, inst_rng, frag_nle, frag_le, inst_split = _group_ranges(
        pc_offsets, pos, frag_groups, inst_groups, is_le)

    frag_count = {g.rep_id: len(g.member_ids) for g in frag_groups}
    inst_count = {g.rep_id: len(g.member_ids) for g in inst_groups}

    # ---- row selection ----------------------------------------------------
    # Representatives that carry points, every kept instance, and the full
    # ancestry of those instances so their history is walkable.
    frag_ids = {g.rep_id for g in frag_groups}
    visible = {g.rep_id for g in frag_groups
               if frag_rng[g.rep_id][1] > frag_rng[g.rep_id][0]}
    visible |= {i for i in kept_inst_ids
                if inst_rng[i][1] > inst_rng[i][0]}

    ancestry = set()
    for iid in kept_inst_ids:
        cur, seen = iid, set()
        while cur in by_id and cur not in seen:
            seen.add(cur)
            ancestry.add(cur)
            nxt = int(by_id[cur].parent_id)
            if nxt == cur:
                break
            cur = nxt

    keep = (visible | kept_inst_ids | ancestry
            | {int(i) for i in keep}) & set(by_id)
    # Rows go by each particle's *own* interaction, so an interaction's rows
    # stay contiguous (part_start/part_end) even when some of its particles'
    # points were absorbed into another interaction's group.  Within an
    # interaction, point order.
    rows = sorted((pos[i] for i in keep),
                  key=lambda k: (int(order[k]._interaction_id), k))
    row_ids = [int(order[k].id) for k in rows]
    row_set = set(row_ids)

    # ---- parent redirection ------------------------------------------------
    def nearest_stored_parent(pid):
        cur, seen = int(by_id[pid].parent_id), set()
        while cur in by_id and cur not in seen:
            seen.add(cur)
            if cur in row_set and cur != pid:
                return cur
            nxt = int(by_id[cur].parent_id)
            if nxt == cur:
                break
            cur = nxt
        return pid                      # no stored ancestor -> a primary here

    def g4_parent_trackid(pid):
        """Track ID of the real direct parent, -1 when it is the particle itself."""
        q = by_id[pid]
        par = int(q.parent_id)
        if par == pid or par not in by_id:
            return -1
        return int(getattr(by_id[par], "geant4_id", par))

    def g4_parent_pdg(pid):
        q = by_id[pid]
        par = int(q.parent_id)
        if par == pid or par not in by_id:
            return -1
        return int(by_id[par].pdg)

    def col(fn, dtype=np.int32):
        return np.fromiter((fn(int(order[k].id)) for k in rows),
                           dtype=dtype, count=len(rows))

    frag_sem = {g.rep_id: g.sem_type for g in frag_groups}
    inst_sem = {g.rep_id: g.sem_type for g in inst_groups}
    sem = lambda gid, table: table.get(gid, -1)

    columns = {
        "id":               col(lambda i: i),
        "geant4_trackid":   col(lambda i: int(getattr(by_id[i], "geant4_id", i))),
        "parent_id":        col(nearest_stored_parent),
        "pdg":              col(lambda i: int(by_id[i].pdg)),
        # parent_pdg describes whatever parent_id points at, so the two always
        # name the same particle.  The *true* Geant4 parent -- usually not
        # stored, since only ~17% of particles are -- is kept alongside.
        "parent_pdg":       col(lambda i: int(by_id[nearest_stored_parent(i)].pdg)),
        "geant4_parent_trackid": col(g4_parent_trackid),
        "geant4_parent_pdg":     col(g4_parent_pdg),
        "interaction_id":   col(lambda i: int(by_id[i]._interaction_id)),
        "interaction_type": col(lambda i: int(by_id[i]._interaction_type)),
        "sem_type":         col(lambda i: int(by_id[i].sem_type.value), np.int8),
        "pc_start":         col(lambda i: int(pc_offsets[pos[i]])),
        "pc_end":           col(lambda i: int(pc_offsets[pos[i] + 1])),
        # group columns: valid where *_id == id
        "frag_id":          col(lambda i: i if i in frag_ids else -1),
        "frag_sem_type":    col(lambda i: sem(i, frag_sem), np.int8),
        "frag_pc_start":    col(lambda i: frag_nle.get(i, (-1, -1))[0], np.int64),
        "frag_pc_end":      col(lambda i: frag_nle.get(i, (-1, -1))[1], np.int64),
        "frag_pc_le_start": col(lambda i: frag_le.get(i, (-1, -1))[0], np.int64),
        "frag_pc_le_end":   col(lambda i: frag_le.get(i, (-1, -1))[1], np.int64),
        "frag_merge_count": col(lambda i: frag_count.get(i, -1)),
        "inst_id":          col(lambda i: i if i in kept_inst_ids else -1),
        "inst_sem_type":    col(lambda i: sem(i, inst_sem), np.int8),
        # Non-LE run then LE run.  They are adjacent here, so
        # inst_pc_end == inst_pc_le_start; the pair is spelled out anyway so
        # instance and fragment level read identically.
        "inst_pc_start":    col(lambda i: inst_rng.get(i, (-1, -1))[0]),
        "inst_pc_end":      col(lambda i: inst_split.get(i, -1), np.int64),
        "inst_pc_le_start": col(lambda i: inst_split.get(i, -1), np.int64),
        "inst_pc_le_end":   col(lambda i: inst_rng.get(i, (-1, -1))[1]),
        "inst_merge_count": col(lambda i: inst_count.get(i, -1)),
    }
    # The instance a fragment belongs to, stated rather than left to be
    # inferred from where its points sit: hit_labels names fragments and
    # instances, and a reader should not need the point ranges to join them.
    columns["frag_inst_id"] = col(
        lambda i: inst_of.get(i, -1) if i in frag_ids else -1)
    # ancestor_id follows the *redirected* parents, so the chain a reader walks and
    # the root it is told to expect agree.  Memoised along each chain: O(n).
    row_index = {pid: k for k, pid in enumerate(row_ids)}
    parents = columns["parent_id"]
    root: dict = {}

    def resolve_ancestor(pid):
        chain, cur, seen = [], pid, set()
        while True:
            if cur in root:
                answer = root[cur]
                break
            if cur in seen:
                answer = cur                  # cycle: treat as its own root
                break
            seen.add(cur)
            chain.append(cur)
            nxt = int(parents[row_index[cur]])
            if nxt == cur:
                answer = cur
                break
            cur = nxt
        for c in chain:
            root[c] = answer
        return answer

    columns["ancestor_id"] = np.fromiter((resolve_ancestor(i) for i in row_ids),
                                     dtype=np.int32, count=len(row_ids))

    # ---- group parent pointers --------------------------------------------
    # The nearest *stored* ancestor that heads a group at this level, skipping
    # the row's own group: when a rep's parent was merged into the rep's own
    # shower, answering with itself would dead-end the production history.
    # Self means "no parent group here", matching the primary convention.
    def group_parent(level_col):
        heads = {int(i): int(i) for i, g in zip(row_ids, columns[level_col])
                 if int(g) == int(i)}
        out = []
        for i in row_ids:
            cur, seen, ans = int(columns["parent_id"][row_index[i]]), set(), i
            while cur not in seen:
                seen.add(cur)
                if cur in heads and cur != i:
                    ans = cur
                    break
                nxt = int(columns["parent_id"][row_index[cur]])
                if nxt == cur:
                    break
                cur = nxt
            out.append(ans)
        return np.asarray(out, dtype=np.int32)

    columns["frag_parent_id"] = group_parent("frag_id")
    columns["inst_parent_id"] = group_parent("inst_id")

    # ---- interactions ------------------------------------------------------
    # Keep only interactions some stored particle belongs to, renumber them,
    # and rewrite interaction_id to the new row index.  Points are ordered by
    # their group's interaction first and rows by their own, so each
    # interaction's points and rows are both contiguous.
    live = sorted({int(v) for v in columns["interaction_id"] if v >= 0})
    remap = {old: new for new, old in enumerate(live)}
    columns["interaction_id"] = np.fromiter(
        (remap.get(int(v), -1) for v in columns["interaction_id"]),
        dtype=np.int32, count=len(row_ids))

    interactions: dict = {}
    if live:
        int_of_order = np.fromiter(
            (point_inter[int(p.id)] for p in order),
            dtype=np.int64, count=len(order))
        part_start, part_end = [], []
        pc_start, pc_end = [], []
        for old in live:
            idx = np.flatnonzero(int_of_order == old)
            if len(idx):
                lo, hi = int(idx[0]), int(idx[-1])
                pc_start.append(int(pc_offsets[lo]))
                pc_end.append(int(pc_offsets[hi + 1]))
            else:
                # Every point of this interaction went to another's group;
                # its particles are still stored, its points are elsewhere.
                pc_start.append(-1); pc_end.append(-1)
            rws = [k for k, r in enumerate(rows)
                   if int(order[r]._interaction_id) == old]
            part_start.append(rws[0] if rws else -1)
            part_end.append(rws[-1] + 1 if rws else -1)
        interactions = {
            # interaction_id is the dense index particles refer to;
            # vertex_id is the surviving link to the EDepSim vertex row.
            "interaction_id": np.arange(len(live), dtype=np.int32),
            "vertex_id":  np.asarray(live, dtype=np.int32),
            "pc_start":   np.asarray(pc_start, dtype=np.int64),
            "pc_end":     np.asarray(pc_end, dtype=np.int64),
            "part_start": np.asarray(part_start, dtype=np.int32),
            "part_end":   np.asarray(part_end, dtype=np.int32),
        }
        for name, col in (("x", "x"), ("y", "y"), ("z", "z"), ("time", "t"),
                          ("energy_sum", "energy_sum"), ("ke_sum", "ke_sum")):
            if vertices is not None and col in (vertices.dtype.names or ()):
                interactions[name] = np.asarray(
                    [float(vertices[col][o]) for o in live], dtype=np.float32)
            else:
                interactions[name] = np.full(len(live), np.nan, dtype=np.float32)

        # reaction is a free-text generator label; carried through verbatim.
        if vertices is not None and "reaction" in (vertices.dtype.names or ()):
            interactions["reaction"] = np.array(
                [(vertices["reaction"][o].decode()
                  if isinstance(vertices["reaction"][o], bytes)
                  else str(vertices["reaction"][o])) for o in live],
                dtype=object)
        else:
            interactions["reaction"] = np.array([""] * len(live), dtype=object)

    return EventLayout(order, rows, pc_offsets, columns, int(pc_offsets[-1]),
                       interactions)
