"""
Pixel hits as a point cloud — the detected image instead of the truth deposits.

The ordinary JAXTPC path (``point_source=deposits``) builds each particle's
point cloud from the Geant4 energy deposits that *survived* the readout.  The
geometry is therefore truth geometry: the readout only decides which deposits
are kept.

This module builds the other cloud.  A pixel readout records, for every fired
pixel and drift tick, a charge; converting ``(py, pz, tick)`` back to
``(x, y, z)`` gives the points the detector actually saw, carrying pixelation,
diffusion, threshold and — depending on *x_from* below — the drift-time
ambiguity.  Running the same partitioning over those points is what
``point_source=hits`` is for.

Every hit still has one Geant4 particle: a CSR entry belongs to exactly one
group, and ``group_to_track`` gives each group one track.  The assignment is
therefore unique without any nearest-neighbour matching.

The inverse transform
---------------------
``y`` and ``z`` are pure geometry and invert exactly::

    y = y_min + (py + 0.5) * pitch
    z = z_min + (pz + 0.5) * pitch

``x`` is drift time, and there the detector does not know what the truth
knows.  A tick counts from the start of the readout window, so it measures
``t_drift + t0`` — and ``t0``, the time the interaction happened, is exactly
what a real detector has to determine separately.  Writing ``T`` for the
elapsed time a hit's tick stands for, measured from the tick at which a
deposit sitting on the anode at Geant4 ``t = 0`` would be recorded::

    T = (tick - reference_tick) * dt

    x = x_anode - drift_direction * T * v                 # 'nominal'
    x = x_anode - drift_direction * (T - t0) * v          # 'true_t0'

``'nominal'`` is what a detector can actually do: assume the interaction
happened at the beam time.  Its x is then offset from the true x by exactly
``drift_direction * v * t0`` -- the interaction's own Geant4 time turned into
a displacement.  ``'true_t0'`` puts the real ``t0`` back, giving the drift
position the detector would measure with perfect timing: every other effect
(pixelation, diffusion, threshold, tick quantisation) survives, only the
ambiguity goes.

Measured on a 10-event pixel batch, ``t0`` is constant within an interaction
(median spread 0.000 μs, max 0.531 μs → 0.85 mm) and differs *across* the
interactions of one event by up to 1094 μs → 175 cm.  So ``'nominal'`` keeps
each interaction's shape to sub-mm and slides whole interactions along x past
one another, while ``'true_t0'`` reproduces truth x to ~0.3 mm.  Neither is
more correct; they answer different questions, which is why both are offered.

The reference tick, and what the readout window hides
-----------------------------------------------------
``reference_tick`` is the tick a deposit on the anode at Geant4 ``t = 0``
lands in, and it is the anchor the whole drift coordinate hangs from.  It is
kept explicit rather than folded into a fitted intercept because the two are
indistinguishable in a fit and only one of them is the anode: a run with a
pre-window has ``reference_tick > 0``, and a calibration that quietly
absorbed it would report an anode that is not where the anode is.

It also decides whether points can leave the detector.  On the batch this was
written against ``reference_tick`` is 0 and ``num_time_steps`` is 2701, so
the recordable ticks 0..2700 map exactly onto ``[anode, cathode]`` and a
nominal x *cannot* fall outside the volume.  The t0 displacement instead
pushes charge off the end of the window, where it is never recorded at all:
25.8% of volume-0 deposits in event 0 would need a tick past 2700, the
largest asking for 4634.  Move the reference tick, or lengthen the window
past one full drift, and the displacement becomes visible as out-of-volume
points instead.  Both are counted and reported rather than clipped -- a
nominal x outside the detector is the honest answer, and clipping it would
invent a wall the reconstruction does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import PointFeature


#: Scale JAXTPC divides a group's per-entry charge by before storing it.
#: Signed int16 is the current encoding; unsigned came from the older
#: diffusion-only path and is still read, matching JAXTPC's own loader.
_CHARGE_SCALE = {'charges_i16': 32767.0, 'charges_u16': 65535.0}

#: ``group_sizes`` is uint8, so a group of more than 255 entries cannot be
#: represented.  JAXTPC caps rather than raising, and a capped group silently
#: truncates the CSR, so it is worth naming when it happens.
_MAX_GROUP_SIZE = 255


@dataclass(frozen=True)
class VolumePixelGeometry:
    """
    Everything needed to turn one volume's pixel indices back into mm.

    Parameters are in mm and μs.  ``drift_direction`` is JAXTPC's: ``+1``
    when electrons drift toward the ``x_max`` face, ``-1`` toward ``x_min``,
    so the anode sits at that face and ``x`` moves away from it with tick.
    The volume's full x range is carried too, because whether a nominal x
    has left the detector is a question only this object can answer.
    """

    y_min_mm: float
    z_min_mm: float
    x_min_mm: float
    x_max_mm: float
    drift_direction: int
    pitch_mm: float
    drift_velocity_mm_us: float
    time_step_us: float
    reference_tick: float = 0.0

    @property
    def mm_per_tick(self) -> float:
        """One tick of drift, in mm: the sampling period times the velocity."""
        return self.drift_velocity_mm_us * self.time_step_us

    @property
    def x_anode_mm(self) -> float:
        """The face the electrons arrive at."""
        return self.x_max_mm if self.drift_direction == 1 else self.x_min_mm

    @property
    def x_cathode_mm(self) -> float:
        """The far face, one full drift away."""
        return self.x_min_mm if self.drift_direction == 1 else self.x_max_mm

    def to_xyz(self, py, pz, tick):
        """``(py, pz, tick)`` -> ``(x, y, z)`` in mm, as float32 arrays."""
        # +0.5 puts the point at the pixel centre rather than its lower edge;
        # without it every point is biased by half a pitch.
        y = self.y_min_mm + (np.asarray(py, dtype=np.float32) + 0.5) * self.pitch_mm
        z = self.z_min_mm + (np.asarray(pz, dtype=np.float32) + 0.5) * self.pitch_mm
        elapsed = np.asarray(tick, dtype=np.float32) - np.float32(self.reference_tick)
        x = self.x_anode_mm - self.drift_direction * (elapsed * self.mm_per_tick)
        return (x.astype(np.float32, copy=False),
                y.astype(np.float32, copy=False),
                z.astype(np.float32, copy=False))

    def outside(self, x):
        """Boolean mask of x values beyond this volume's own faces."""
        x = np.asarray(x)
        return (x < self.x_min_mm) | (x > self.x_max_mm)


def anode_of(x_range, drift_direction: int) -> float:
    """The anode face of a volume, given its x range and drift direction."""
    lo, hi = float(x_range[0]), float(x_range[1])
    return hi if drift_direction == 1 else lo


# ---------------------------------------------------------------------------
# CSR decode
# ---------------------------------------------------------------------------

def decode_plane_hits(plane) -> dict:
    """
    Expand one pixel plane's CSR correspondence into flat per-entry arrays.

    JAXTPC stores each group once — its peak pixel, tick and charge — plus one
    int8 delta per entry against that peak.  Undoing it is a repeat of the
    centres plus the deltas, which is why this is vectorised rather than the
    per-group Python loop JAXTPC's own loader uses.

    Returns
    -------
    dict
        ``py``, ``pz``, ``tick`` (int32) and ``charge`` (float32), one entry
        per fired pixel/tick, plus ``group`` (int32), the volume-local group
        number each entry belongs to.

    Notes
    -----
    The delta arrays are longer than ``group_sizes.sum()`` — JAXTPC pads them
    to a chunk boundary.  The CSR length is the sum, never the array length;
    reading to the end would append hundreds of zero-delta ghosts sitting on
    top of the last group's peak.
    """
    sizes = plane['group_sizes'][:].astype(np.int64)
    groups = plane['group_ids'][:].astype(np.int32)
    n = int(sizes.sum())

    capped = int((sizes >= _MAX_GROUP_SIZE).sum())

    key = next((k for k in _CHARGE_SCALE if k in plane), None)
    if key is None:
        raise KeyError(
            f"pixel plane {plane.name!r} has no per-entry charge dataset "
            f"(expected one of {sorted(_CHARGE_SCALE)}); it cannot be "
            f"expanded into hits.  A plane subset written by "
            f"pysupera-hits-subset keeps only the group centres and is not "
            f"usable as reader.jaxtpc_inst_path.")

    centre_py = np.repeat(plane['center_py'][:].astype(np.int32), sizes)
    centre_pz = np.repeat(plane['center_pz'][:].astype(np.int32), sizes)
    centre_t = np.repeat(plane['center_times'][:].astype(np.int32), sizes)
    peak = np.repeat(plane['peak_charges'][:].astype(np.float32), sizes)

    py = centre_py + plane['delta_py'][:n].astype(np.int32)
    pz = centre_pz + plane['delta_pz'][:n].astype(np.int32)
    tick = centre_t + plane['delta_times'][:n].astype(np.int32)
    charge = peak * (plane[key][:n].astype(np.float32) / _CHARGE_SCALE[key])

    return {'py': py, 'pz': pz, 'tick': tick, 'charge': charge,
            'group': np.repeat(groups, sizes), 'n_capped_groups': capped}


# ---------------------------------------------------------------------------
# Per-group truth quantities
# ---------------------------------------------------------------------------

def group_reductions(deposit_to_group, n_groups, t0_us=None, de=None):
    """
    Mean ``t0`` and summed ``de`` per group, from the deposits behind it.

    Both are truth quantities a hit does not carry.  ``t0`` is needed to undo
    the drift-time ambiguity (``x_from='true_t0'``) and ``de`` to give
    the energy column true deposited energy instead of charge
    (``energy='true_de'``); neither is computed unless asked for.

    Groups with no deposits come back as 0, which only reaches a point if the
    readout produced a hit for a group nothing deposited into — impossible by
    construction, and harmless if the schema ever changes.
    """
    counts = np.bincount(deposit_to_group, minlength=n_groups).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    out = {}
    if t0_us is not None:
        out['t0'] = (np.bincount(deposit_to_group, weights=t0_us,
                                 minlength=n_groups) / safe).astype(np.float32)
    if de is not None:
        out['de'] = np.bincount(deposit_to_group, weights=de,
                                minlength=n_groups).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Point-cloud assembly
# ---------------------------------------------------------------------------

#: Accepted values for the drift-coordinate convention.
#: ``nominal`` assumes the interaction happened at the beam time (Geant4
#: t=0), which is what a detector without an independent t0 must do;
#: ``true_t0`` puts the real interaction time back.
X_FROM = ('nominal', 'true_t0')

#: Accepted values for what lands in the energy column.
ENERGY_FROM = ('charge', 'true_de')


def build_hit_point_cloud(volumes, track_ids_edepsim, *,
                          x_from: str = 'nominal',
                          energy: str = 'charge',
                          time_step_us: float = 1.0,
                          charge_threshold: float = 0.0):
    """
    A flat point cloud of detected pixel hits, laid out per particle.

    Parameters
    ----------
    volumes : list of dict
        One per volume, each holding ``hits`` (the :func:`decode_plane_hits`
        output), ``geom`` (:class:`VolumePixelGeometry`), ``group_to_track``,
        ``group_offset`` (where this volume's groups start in the event-global
        numbering) and, when the corresponding option is selected,
        ``group_t0`` and ``group_de``.
    track_ids_edepsim : ndarray
        Track IDs in EDepSim particle order.  The returned offsets are
        aligned to it, so ``Particle.from_flat_arrays`` pairs each metadata
        row with the right slice.
    x_from : {'nominal', 'true_t0'}
        Whether the drift coordinate keeps the interaction time in it.  See
        the module docstring.
    energy : {'charge', 'true_de'}
        Charge as measured, or the group's true deposited energy shared out
        over its hits in proportion to charge.
    time_step_us : float
        Readout tick length, used to put the time column in μs.
    charge_threshold : float
        Drop hits carrying less than this much charge, in the units the hits
        file uses -- ionisation electrons induced on the pixel.  Compared
        against the *magnitude*, matching JAXTPC's own
        ``encode_correspondence_csr_pixel``: the readout response is
        bipolar, and a large negative lobe is signal, not noise.

        JAXTPC applies no threshold worth the name (1 electron in this
        batch), so without one every pixel carrying any induced-field
        influence is a point: 4.18M of them for an event whose truth is
        216k deposits, and the brightest 2% carry half the charge.

    Returns
    -------
    point_cloud_flat : (M, 7) float32
    offsets : (N, 2) int64
    true_x_shift : (M,) float32
        What to add to ``x`` to recover the true drift position:
        ``true_x = x + true_x_shift``.  Zero throughout under
        ``x_from='true_t0'``, so the relation holds either way.
    group_of_point : (M,) int64
        Event-global group number per row.  This is the hit-mode counterpart
        of the deposit index the truth path carries, and it is what makes the
        ``groups/`` provenance table exact here: a point *is* a hit, so its
        group is read rather than looked up through a voxel map.
    stats : dict
        Includes ``n_outside_volume``: nominal x values beyond their own
        volume's faces.  Counted, never clipped -- see the module docstring.
    """
    if x_from not in X_FROM:
        raise ValueError(f"x_from must be one of {X_FROM}, got {x_from!r}")
    if energy not in ENERGY_FROM:
        raise ValueError(f"energy must be one of {ENERGY_FROM}, got {energy!r}")

    xs, ys, zs, ts, es, gs, tr, sh = [], [], [], [], [], [], [], []
    n_capped = 0
    n_outside = 0
    n_cut = 0          # hits dropped by charge_threshold
    n_decoded = 0      # hits in the file, before that cut

    for vol in volumes:
        hits = vol['hits']
        if not len(hits['group']):
            continue
        n_capped += int(hits.get('n_capped_groups', 0))
        n_decoded += len(hits['group'])
        geom = vol['geom']
        g_local = hits['group']

        q_raw = hits['charge']
        if charge_threshold > 0:
            keep = np.abs(q_raw) >= charge_threshold
            n_cut += int((~keep).sum())
            if not keep.all():
                hits = {k: (v[keep] if isinstance(v, np.ndarray) else v)
                        for k, v in hits.items()}
                g_local = hits['group']
                if not len(g_local):
                    continue

        tick = hits['tick'].astype(np.float32)
        if x_from == 'true_t0':
            t0 = vol['group_t0']
            # t0 is per group and constant within an interaction, so removing
            # it per group is exact rather than an approximation.
            tick = tick - (t0[g_local] / np.float32(time_step_us))

        x, y, z = geom.to_xyz(hits['py'], hits['pz'], tick)
        n_outside += int(geom.outside(x).sum())

        # What would have to be added to x to put it back where the deposit
        # was:  true_x = x + shift.  Under 'nominal' that is the interaction
        # time turned into a displacement, drift_direction * t0 * v; under
        # 'true_t0' the stored x is already the true one, so it is zero.  Kept
        # per hit rather than per particle because a track crossing the
        # cathode has hits in both volumes, where drift_direction differs --
        # rare (2 particles in a test event) but 15% of the points.
        if x_from == 'true_t0':
            sh.append(np.zeros(len(x), dtype=np.float32))
        else:
            t0v = vol['group_t0'][g_local] * np.float32(
                geom.drift_direction * geom.drift_velocity_mm_us)
            sh.append(t0v.astype(np.float32, copy=False))
        xs.append(x); ys.append(y); zs.append(z)
        # The time column follows whichever x convention is in force, so the
        # two never disagree about when a point happened.
        ts.append((tick * np.float32(time_step_us)).astype(np.float32))

        q = hits['charge']
        if energy == 'charge':
            es.append(q)
        else:
            # Share the group's true dE over its hits by |charge|.  A hit has
            # no energy of its own, and this is the only apportionment the
            # file supports.  The magnitude, not the signed value: the
            # response is bipolar, and sharing by signed charge would hand
            # the undershoot pixels negative "deposited energy" -- which the
            # deposit-mode column this one exists to be comparable with can
            # never be.  Still exact in sum per group.
            n_groups = len(vol['group_to_track'])
            w = np.abs(q.astype(np.float64))
            wsum = np.bincount(g_local, weights=w, minlength=n_groups)
            wsum = np.where(wsum != 0, wsum, 1.0)
            share = w / wsum[g_local]
            es.append((vol['group_de'][g_local] * share).astype(np.float32))

        gs.append(g_local.astype(np.int64) + int(vol['group_offset']))
        tr.append(vol['group_to_track'][g_local].astype(np.int64))

    n_particles = len(track_ids_edepsim)
    if not xs:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((n_particles, 2), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                {'n_hits': 0, 'n_attached': 0, 'n_unmatched': 0,
                 'n_capped_groups': n_capped, 'n_outside_volume': 0,
                 'n_decoded': n_decoded, 'n_below_threshold': n_cut})

    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    t = np.concatenate(ts); e = np.concatenate(es)
    group = np.concatenate(gs); track = np.concatenate(tr)
    shift = np.concatenate(sh)
    n_hits = len(x)

    # Track ID -> row in the EDepSim particle list.  A lookup by searchsorted
    # rather than a dict: with millions of hits the dict comprehension is the
    # slowest line in the reader.
    tid = np.asarray(track_ids_edepsim)
    order_t = np.argsort(tid, kind='stable')
    sorted_t = tid[order_t]
    pos = np.searchsorted(sorted_t, track)
    inside = pos < len(sorted_t)
    pos_safe = np.where(inside, pos, 0)
    matched = inside & (sorted_t[pos_safe] == track)
    row = order_t[pos_safe]

    # Hits whose group names a track the EDepSim file does not list cannot be
    # attributed, exactly as in the deposit path; count them and drop them.
    keep = np.flatnonzero(matched)
    row = row[keep]

    # Lay the cloud out particle by particle: one stable sort, then the
    # offsets fall out of the counts.
    order = np.argsort(row, kind='stable')
    sel = keep[order]
    row_sorted = row[order]

    counts = np.bincount(row_sorted, minlength=n_particles)
    ends = np.cumsum(counts)
    offsets = np.empty((n_particles, 2), dtype=np.int64)
    offsets[:, 0] = ends - counts
    offsets[:, 1] = ends

    m = len(sel)
    flat = np.zeros((m, 7), dtype=np.float32)
    flat[:, PointFeature.x] = x[sel]
    flat[:, PointFeature.y] = y[sel]
    flat[:, PointFeature.z] = z[sel]
    flat[:, PointFeature.time] = t[sel]
    flat[:, PointFeature.energy] = e[sel]
    # dx stays 0: a hit is a pixel and a tick, it has no path length.  Leaving
    # it zero is honest; inventing one would quietly feed dE/dx consumers.
    flat[:, PointFeature.id] = np.arange(m, dtype=np.float32)

    stats = {'n_hits': n_hits, 'n_attached': m, 'n_unmatched': n_hits - m,
             'n_capped_groups': n_capped, 'n_outside_volume': n_outside,
             'n_decoded': n_decoded, 'n_below_threshold': n_cut}
    return flat, offsets, group[sel], shift[sel], stats


# ---------------------------------------------------------------------------
# Geometry calibration
# ---------------------------------------------------------------------------

#: How far a calibrated prediction may sit from truth before the geometry is
#: treated as wrong.  y and z are quantised to the pitch, so one pitch is the
#: natural bound; x is continuous and holds to a few tenths of a mm in
#: practice, so 5 mm is loose enough to survive a different detector and
#: tight enough to catch a mirrored volume or a wrong tick length.
_FIT_TOL_XY_PITCHES = 1.0
_FIT_TOL_X_MM = 5.0


class PixelGeometryError(ValueError):
    """The pixel geometry could not be established from the files."""


def window_truncation(x_true, t0_us, geom, num_time_steps):
    """
    How many deposits the readout window never got to record.

    A deposit at ``x`` with interaction time ``t0`` arrives at tick
    ``reference_tick + (|x - x_anode|/v + t0)/dt``.  When the window is
    exactly one full drift long -- as it is on the batch this was written
    against -- any positive ``t0`` pushes the cathode-side end of an
    interaction past the last tick, and it is simply not in the file.

    That loss is what makes hit mode look like it dropped charge for no
    reason, and it is ``t0``-dependent, so it biases a comparison against
    deposit mode rather than merely shrinking it.  Hence counted.

    Returns ``(n_before, n_after, n_total)``.
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    drift_us = np.abs(x_true - geom.x_anode_mm) / geom.drift_velocity_mm_us
    t0_us = np.asarray(t0_us, dtype=np.float64)
    want = geom.reference_tick + (drift_us + t0_us) / geom.time_step_us
    return (int((want < 0).sum()),
            int((want > num_time_steps - 1).sum()),
            int(len(want)))


def resolve_volume_geometry(volume_ranges_mm, *, pitch_mm, drift_direction,
                            drift_velocity_mm_us, time_step_us,
                            reference_tick=0.0, volume_id=0):
    """
    Build a volume's geometry from stated values, fitting nothing.

    This is the path the reader wants to take.  Every number here is either
    written in a file (the volume extent in the hits file's
    ``config/volume_ranges``, the drift velocity and sampling period in the
    sensor file's ``config`` attributes) or configured by hand (the pixel
    pitch and the drift direction, which no JAXTPC output file records).
    :func:`verify_volume_geometry` then checks the result against truth --
    a check that means something precisely because nothing was tuned to make
    it pass.

    Parameters
    ----------
    volume_ranges_mm : array-like, shape (3, 2)
        The volume's ``[[x0, x1], [y0, y1], [z0, z1]]`` extent in mm.
    pitch_mm, drift_direction, drift_velocity_mm_us, time_step_us : float
        Stated, not measured.  ``drift_direction`` is ``+1`` when electrons
        drift toward the ``x_max`` face and ``-1`` toward ``x_min``.
    reference_tick : float
        The tick a deposit on the anode at Geant4 ``t = 0`` is recorded in.

    Returns
    -------
    VolumePixelGeometry
    """
    r = np.asarray(volume_ranges_mm, dtype=np.float64)
    if r.shape != (3, 2):
        raise PixelGeometryError(
            f"volume {volume_id}: expected a (3, 2) x/y/z extent in mm, got "
            f"shape {r.shape}.  This comes from config/volume_ranges in the "
            f"JAXTPC hits file.")
    if drift_direction not in (-1, 1):
        raise PixelGeometryError(
            f"volume {volume_id}: drift_direction must be +1 (electrons "
            f"drift toward x_max) or -1 (toward x_min), got "
            f"{drift_direction!r}.")
    for name, val in (('pitch_mm', pitch_mm),
                      ('drift_velocity_mm_us', drift_velocity_mm_us),
                      ('time_step_us', time_step_us)):
        if val is None or not np.isfinite(val) or val <= 0:
            raise PixelGeometryError(
                f"volume {volume_id}: {name} must be a positive number, got "
                f"{val!r}.")

    return VolumePixelGeometry(
        y_min_mm=float(min(r[1])), z_min_mm=float(min(r[2])),
        x_min_mm=float(min(r[0])), x_max_mm=float(max(r[0])),
        drift_direction=int(drift_direction),
        pitch_mm=float(pitch_mm),
        drift_velocity_mm_us=float(drift_velocity_mm_us),
        time_step_us=float(time_step_us),
        reference_tick=float(reference_tick),
    )


def verify_volume_geometry(geom, py, pz, tick, t0, xyz_truth, volume_id=0):
    """
    Check a stated geometry against the truth deposits behind each group.

    Nothing here is tuned: *geom* is applied as given, the groups' hit
    centres are converted with it, and the distance to the deposits that
    produced them is measured.  A fit is run alongside purely to report what
    the data would have said, so a disagreement names the offending number
    rather than just failing.

    That distinction is the point of this function.  Deriving the constants
    from the same truth they are then checked against cannot fail for a
    geometry that is wrong but linear -- the fit simply absorbs the error,
    and a detector effect the study exists to measure is calibrated away
    instead.  Applying stated constants and measuring the residual can.

    Returns
    -------
    report : dict
        Stated values, what the fit measured, and the residuals.
    """
    n = len(py)
    if n < 8:
        raise PixelGeometryError(
            f"volume {volume_id}: only {n} group(s) have both a hit centre "
            f"and truth deposits, too few to check the pixel geometry "
            f"against.")

    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    tick = np.asarray(tick, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    xt, yt, zt = (np.asarray(xyz_truth[:, i], dtype=np.float64)
                  for i in range(3))

    measured = _measure_volume(py, pz, tick, t0, xt, yt, zt)

    # Where the data puts the anchor, given the stated anode and tick length.
    derived_ref = (geom.drift_direction
                   * (measured['fit_intercept_mm'] - geom.x_anode_mm)
                   / geom.mm_per_tick) if geom.mm_per_tick else 0.0

    # Apply the stated geometry, but anchored where the data says.  t0 is
    # removed too: the question here is whether the pitch, velocity and
    # drift direction describe this detector -- all measurements, all
    # checkable.  The reference tick is not one of them.  It is a choice
    # about where t=0 sits, so someone asking "what if the trigger were 100
    # ticks later" must not be told their detector is broken.  How far that
    # choice sits from the data is reported below, in mm, instead.
    elapsed_ticks = tick - derived_ref - t0 / geom.time_step_us
    x_pred = geom.x_anode_mm - geom.drift_direction * (
        elapsed_ticks * geom.mm_per_tick)
    y_pred = geom.y_min_mm + (py + 0.5) * geom.pitch_mm
    z_pred = geom.z_min_mm + (pz + 0.5) * geom.pitch_mm

    res = {'x': float(np.median(np.abs(xt - x_pred))),
           'y': float(np.median(np.abs(yt - y_pred))),
           'z': float(np.median(np.abs(zt - z_pred)))}

    tol_xy = _FIT_TOL_XY_PITCHES * geom.pitch_mm
    bad = [k for k, v in res.items()
           if v > (_FIT_TOL_X_MM if k == 'x' else tol_xy)]
    if bad:
        raise PixelGeometryError(
            f"volume {volume_id}: the stated pixel geometry does not "
            f"reproduce the truth deposits over {n} groups -- median "
            f"|residual| x={res['x']:.2f} y={res['y']:.2f} z={res['z']:.2f} "
            f"mm, off in {', '.join(bad)}.\n"
            f"  stated:   pitch {geom.pitch_mm:.4f} mm, "
            f"{geom.mm_per_tick:.4f} mm/tick, v "
            f"{geom.drift_velocity_mm_us:.4f} mm/us, dt "
            f"{geom.time_step_us:.4f} us, drift {geom.drift_direction:+d}, "
            f"anode {geom.x_anode_mm:+.1f} mm, reference tick "
            f"{geom.reference_tick:g}\n"
            f"  data says: pitch {measured['pitch_mm']:.4f} mm, "
            f"{measured['mm_per_tick']:.4f} mm/tick, v "
            f"{measured['drift_velocity_mm_us']:.4f} mm/us, dt "
            f"{measured['time_step_us']:.4f} us, drift "
            f"{measured['drift_direction']:+d}\n"
            f"Check reader.pixel_pitch_mm, reader.pixel_drift_direction and "
            f"the sensor file, or that the hits and step files are from the "
            f"same run.")

    return {'n_groups': n,
            'pitch_mm': geom.pitch_mm,
            'mm_per_tick': geom.mm_per_tick,
            'drift_direction': geom.drift_direction,
            'x_anode_mm': geom.x_anode_mm,
            'reference_tick': geom.reference_tick,
            'drift_velocity_mm_us': geom.drift_velocity_mm_us,
            'time_step_us': geom.time_step_us,
            'residual_mm': res,
            'reference_tick_derived': float(derived_ref),
            'reference_tick_offset_mm': float(
                (geom.reference_tick - derived_ref) * geom.mm_per_tick),
            'measured': measured,
            'source': 'stated'}


def _measure_volume(py, pz, tick, t0, xt, yt, zt):
    """What the data says the geometry is.  Reporting only -- never applied."""
    def fit1(idx, target):
        A = np.stack([np.ones_like(idx), idx], axis=1)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return coef
    (_cy0, pitch_y) = fit1(py, yt)
    (_cz0, pitch_z) = fit1(pz, zt)
    A = np.stack([np.ones_like(tick), tick, t0], axis=1)
    (icept, b_tick, b_t0), *_ = np.linalg.lstsq(A, xt, rcond=None)
    mm_per_tick = abs(b_tick)
    velocity = abs(b_t0)
    return {'pitch_mm': float(0.5 * (pitch_y + pitch_z)),
            'pitch_y_mm': float(pitch_y), 'pitch_z_mm': float(pitch_z),
            'mm_per_tick': float(mm_per_tick),
            'drift_direction': -1 if b_tick > 0 else 1,
            'drift_velocity_mm_us': float(velocity),
            'time_step_us': float(mm_per_tick / velocity) if velocity
                            else float('nan'),
            'fit_intercept_mm': float(icept)}


def calibrate_volume(py, pz, tick, t0, xyz_truth, volume_id=0,
                     x_range=None, reference_tick=None):
    """
    Recover one volume's pixel geometry from group centres and truth.

    Nothing in a JAXTPC output file states the pixel pitch, the anode face or
    the tick length — ``config/num_wires`` is empty for pixel readout and the
    rest lives only in the detector YAML.  Rather than ask for numbers no
    file carries, and that are silently wrong if mistyped, this reads them
    off the correspondence that is already there: every group has both a hit
    centre and the deposits behind it.

    ``y`` and ``z`` are single-variable fits.  ``x`` is a two-variable one,
    because a tick measures ``t_drift + t0``::

        x = x_anode - d*mm_per_tick*tick + d*(mm_per_tick/dt)*t0

    so regressing truth ``x`` on ``(tick, t0)`` yields an intercept, the mm
    per tick, the drift direction (from the sign) and the drift velocity at
    once.  The residual then says whether any of it is believable — which is
    the point of doing it this way rather than trusting a config.

    The intercept is *not* the anode.  It is
    ``x_anode + drift_direction * mm_per_tick * reference_tick``, and a fit
    cannot separate the two.  So the anode is taken from *x_range* -- the
    volume's stated extent, which is geometry and not in doubt -- and the
    reference tick is read off as what remains.  On a run whose window opens
    at Geant4 t=0 that residual comes out at zero, which is the cross-check;
    on a run with a pre-window it comes out at the pre-window, which is the
    number that was wanted.  Pass *reference_tick* to state it instead, and
    the residual then reports how far the truth disagrees.

    Parameters
    ----------
    x_range : (float, float), optional
        The volume's x extent in mm.  Without it the anode cannot be
        separated from the reference tick, and the latter is taken as 0.
    reference_tick : float, optional
        The tick at which a deposit on the anode at Geant4 ``t = 0`` is
        recorded.  ``None`` derives it from *x_range*.

    Returns
    -------
    VolumePixelGeometry
    report : dict
        Fitted values and residuals, for logging.
    """
    n = len(py)
    if n < 8:
        raise PixelGeometryError(
            f"volume {volume_id}: only {n} group(s) have both a hit centre "
            f"and truth deposits, which is too few to establish the pixel "
            f"geometry.  Set the reader's pixel geometry explicitly, or "
            f"calibrate on an event with more activity.")

    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    tick = np.asarray(tick, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    xt, yt, zt = (np.asarray(xyz_truth[:, i], dtype=np.float64)
                  for i in range(3))

    def fit1(idx, target):
        A = np.stack([np.ones_like(idx), idx], axis=1)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return coef, target - A @ coef

    (cy0, pitch_y), ry = fit1(py, yt)
    (cz0, pitch_z), rz = fit1(pz, zt)

    A = np.stack([np.ones_like(tick), tick, t0], axis=1)
    (x_anode, b_tick, b_t0), *_ = np.linalg.lstsq(A, xt, rcond=None)
    rx = xt - A @ np.array([x_anode, b_tick, b_t0])

    pitch = 0.5 * (pitch_y + pitch_z)
    if not np.isfinite(pitch) or pitch <= 0:
        raise PixelGeometryError(
            f"volume {volume_id}: fitted pixel pitch {pitch!r} is not a "
            f"positive length; the hit centres and the truth deposits do not "
            f"look like the same volume.")
    if abs(pitch_y - pitch_z) > 0.05 * pitch:
        raise PixelGeometryError(
            f"volume {volume_id}: fitted pitch differs between the two pixel "
            f"axes ({pitch_y:.4f} mm in y, {pitch_z:.4f} mm in z).  This "
            f"reader assumes a square pixel grid.")

    mm_per_tick = abs(b_tick)
    # b_tick = -drift_direction * mm_per_tick, so the sign of the fit is the
    # sign of the drift, read off the data instead of configured.
    drift_direction = -1 if b_tick > 0 else 1
    # b_t0 = drift_direction * velocity, so the two x slopes give dt as well.
    velocity = abs(b_t0)
    dt = mm_per_tick / velocity if velocity else float('nan')

    # Separate the anode from the reference tick using the stated geometry.
    if x_range is not None:
        x_lo, x_hi = float(min(x_range)), float(max(x_range))
        anode = x_hi if drift_direction == 1 else x_lo
        derived_ref = (drift_direction * (x_anode - anode) / mm_per_tick
                       if mm_per_tick else 0.0)
    else:
        x_lo = x_hi = None
        anode = float(x_anode)
        derived_ref = 0.0
    ref = float(derived_ref if reference_tick is None else reference_tick)

    # Measure the residual against the *derived* anchor, always.  The
    # residual is there to say whether the pitch, the velocity and the drift
    # direction describe this detector -- those are measurements, and a bad
    # one means the files do not belong together.  A supplied reference tick
    # is not a measurement but a choice about where t=0 sits, and someone
    # asking "what if the trigger were 100 ticks later" must not be told
    # their detector is broken.  How far that choice sits from the data is
    # reported instead, in mm, where a mistyped value is plain to see.
    x_pred = anode - drift_direction * ((tick - derived_ref) * mm_per_tick
                                        - t0 * velocity)
    rx = xt - x_pred

    res = {'x': float(np.median(np.abs(rx))),
           'y': float(np.median(np.abs(ry))),
           'z': float(np.median(np.abs(rz)))}
    bad = [k for k, v in res.items()
           if v > (_FIT_TOL_X_MM if k == 'x' else _FIT_TOL_XY_PITCHES * pitch)]
    if bad:
        raise PixelGeometryError(
            f"volume {volume_id}: the pixel geometry fitted from {n} groups "
            f"does not reproduce the truth deposits — median |residual| "
            f"x={res['x']:.2f} y={res['y']:.2f} z={res['z']:.2f} mm, off in "
            f"{', '.join(bad)}.  Either the hits and step files are from "
            f"different runs, or this volume is not a regular pixel grid.")

    if x_lo is None:
        # No stated extent: fall back to one full drift either side of the
        # anode, which is the most the readout can reach anyway.
        span = abs(mm_per_tick) * 1.0
        x_lo, x_hi = (anode, anode + span) if drift_direction == -1 \
            else (anode - span, anode)

    geom = VolumePixelGeometry(
        y_min_mm=float(cy0 - 0.5 * pitch),
        z_min_mm=float(cz0 - 0.5 * pitch),
        x_min_mm=float(x_lo), x_max_mm=float(x_hi),
        drift_direction=int(drift_direction),
        pitch_mm=float(pitch),
        drift_velocity_mm_us=float(velocity),
        time_step_us=float(dt),
        reference_tick=ref,
    )
    report = {
        'n_groups': n, 'pitch_mm': float(pitch),
        'mm_per_tick': float(mm_per_tick),
        'drift_direction': int(drift_direction),
        'x_anode_mm': float(anode),
        'fit_intercept_mm': float(x_anode),
        'reference_tick': ref,
        'reference_tick_derived': float(derived_ref),
        'reference_tick_offset_mm': float((ref - derived_ref) * mm_per_tick),
        'drift_velocity_mm_us': float(velocity),
        'time_step_us': float(dt),
        'residual_mm': res,
    }
    return geom, report
