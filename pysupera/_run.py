"""
CLI entry point for the nusupera partitioning workflow.

After installation (``pip install -e .``) this is available as a shell
command:

    run_pysupera io.input_path=/data/in.h5 io.output_path=/data/out.h5
    run_pysupera checker=cpu_multi checker.n_jobs=8
    run_pysupera distance_threshold=3.0 conditions.touching_le_scatter=false

JAXTPC visibility filtering
---------------------------
Pass two extra flags to restrict each particle's point cloud to only the
energy-deposit segments that were visible in the JAXTPC readout simulation
(i.e. segments whose ionisation group was reconstructed above threshold).
All particle-level metadata (PDG, track IDs, interaction type, etc.) still
comes from the EDepSim file:

    run_pysupera \\
        io.input_path=/data/edepsim.h5 \\
        io.output_path=/data/out.h5 \\
        reader.jaxtpc_seg_path=/data/sim_seg_0000.h5 \\
        reader.jaxtpc_inst_path=/data/sim_inst_0000.h5

Multi-run example (Hydra sweep):

    run_pysupera --multirun \\
        checker=gpu,bulk_gpu,numba \\
        distance_threshold=3.0,5.0,7.0

The working directory is NOT changed (``hydra.run.dir=.``) so relative paths
in ``io.*`` resolve against the directory where you invoke the script.
See ``conf/config.yaml`` for all available options.
"""

import os as _os
import hydra
from omegaconf import DictConfig

_CONF_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "conf")


class _NullCtx:
    """No-op context manager used when voxmap writing is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def append_event(self, *_): pass


@hydra.main(config_path=_CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Partition all events in the input file and write results."""
    import time
    import numpy as np
    from collections import defaultdict
    from pysupera import open_writer
    from pysupera.io import open_voxmap_writer
    from pysupera.partitioner import ParticlePartitioner
    from pysupera.merge import merge_em_showers
    from pysupera.config import build_conditions, build_preprocessor, build_merge_processor, build_voxelizer, build_reader, configure
    from pysupera.preproc import _voxelize_point_cloud
    from pysupera.utils import resolve_orphans

    configure(cfg)  # set module-level defaults (e.g. min_pc_size) before any Particle is created

    merge_processor = build_merge_processor(cfg)  # None when merge_duplicates: false
    voxelizer       = build_voxelizer(cfg)         # None when voxelize.enabled: false
    preprocessor    = build_preprocessor(cfg)      # None when defragment: false

    _write_voxmap = (voxelizer is not None
                     and bool(cfg.particle.voxelize.get('store_mapping', False)))

    _vox_info = (f"enabled (voxel_size={cfg.particle.voxelize.voxel_size})"
                 if voxelizer is not None else "disabled")
    print(f"[run] checker         : {cfg.checker.name}")
    print(f"[run] distance        : {cfg.distance_threshold}")
    print(f"[run] merge_duplicates: {'enabled' if merge_processor is not None else 'disabled'}")
    print(f"[run] voxelize        : {_vox_info}")
    print(f"[run] preprocessor    : {cfg.particle.get('preprocessor', {}).get('name', 'scipy') if cfg.particle.get('defragment', False) else 'disabled'}")
    print(f"[run] output_compact  : {bool(cfg.get('output_compact', False))}")
    print(f"[run] drop_le_scatter : {bool(cfg.get('drop_le_scatter', False))}")
    print(f"[run] write_event_clouds: {bool(cfg.get('write_event_clouds', True))}")
    print(f"[run] allow_empty_image: {bool(cfg.get('allow_empty_image', False))}")
    print(f"[run] repack          : {bool((cfg.get('repack', {}) or {}).get('enabled', False))}")
    print(f"[run] input           : {cfg.io.input_path}")
    print(f"[run] output          : {cfg.io.output_path}")
    _jaxtpc_seg  = cfg.reader.get("jaxtpc_seg_path",  None)
    _jaxtpc_inst = cfg.reader.get("jaxtpc_inst_path", None)
    if _jaxtpc_seg and _jaxtpc_inst:
        print(f"[run] jaxtpc mode     : visibility-filtered")
        print(f"[run]   seg           : {_jaxtpc_seg}")
        print(f"[run]   inst          : {_jaxtpc_inst}")
    else:
        print(f"[run] jaxtpc mode     : disabled (all EDepSim segments used)")
    if _write_voxmap:
        from pysupera.io import voxmap_path
        print(f"[run] voxmap        : {voxmap_path(cfg.io.output_path)}")

    conditions = build_conditions(cfg)

    # ── Accumulators (always collected; printed only when report=True) ──────
    profile: dict = defaultdict(float)   # stage name → cumulative wall-clock s
    stats:   dict = {
        'n_particles_in': [],   # int  : total particles entering partitioner
        'n_nonzero_pc':   [],   # int  : particles with at least one point
        'pc_size_min':    [],   # int  : smallest non-zero PC in event
        'pc_size_max':    [],   # int  : largest non-zero PC in event
        'pc_size_mean':   [],   # float: mean non-zero PC size in event
        'n_partitions':   [],   # int: final partition count after partition_combined
    }
    # JAXTPC visibility counts; stays empty unless the reader supplies them.
    mask_stats: dict = defaultdict(list)
    n_events = 0

    _max_events = int(cfg.get("max_events", -1))
    _show_progress = bool(cfg.get("progress", True))

    # Empty-image policy: halt by default, warn once when explicitly allowed.
    _allow_empty_image = bool(cfg.get("allow_empty_image", False))
    _warned_empty_image = False

    with build_reader(cfg) as store:
        _n_available = len(store)
        _n_to_process = _n_available if _max_events < 0 else min(_max_events, _n_available)
        print(f"[run] {_n_available} events in input file, processing {_n_to_process}")

        _output_compact     = bool(cfg.get('output_compact',     False))
        _drop_le_scatter    = bool(cfg.get('drop_le_scatter',    False))
        _write_event_clouds = bool(cfg.get('write_event_clouds', True))
        # _is_le is always defined: used independently by drop_le_scatter output
        # filtering AND by event-cloud LE/non-LE splitting (write_event_clouds).
        from pysupera.utils import SemanticType as _ST
        _le_type = _ST.kLEScatter
        _is_le = lambda p: p.sem_type == _le_type

        with open_writer(cfg.io.output_path,
                         compression=cfg.io.compression,
                         compression_opts=cfg.io.compression_opts) as writer, \
             (open_voxmap_writer(cfg.io.output_path,
                                 compression=cfg.io.compression,
                                 compression_opts=cfg.io.compression_opts)
              if _write_voxmap else _NullCtx()) as vox_writer:

            from tqdm import tqdm
            _bar = tqdm(
                total=_n_to_process,
                desc="[run] events",
                unit="ev",
                disable=not _show_progress,
                dynamic_ncols=True,
            )

            for event_idx, particles in enumerate(store):
                if _max_events >= 0 and event_idx >= _max_events:
                    break
                n_events += 1
                _ev_t: dict = {}   # per-event stage timings (seconds)

                # ── JAXTPC visibility counts ────────────────────────────────
                # Already computed by the reader from data it had in hand, so
                # collecting them costs nothing; only the printing is gated.
                _ms = getattr(store, 'last_mask_stats', None)
                if _ms:
                    for _k, _v in _ms.items():
                        mask_stats[_k].append(_v)
                    if cfg.verbose:
                        _tot = max(1, _ms['n_total'])
                        print(f"[run] event {event_idx}: segments "
                              f"total={_ms['n_total']:,} "
                              f"visible={_ms['n_visible']:,} "
                              f"masked={_ms['n_masked']:,} "
                              f"({100.0 * _ms['n_masked'] / _tot:.1f}%)"
                              + (f" unmatched={_ms['n_unmatched']:,}"
                                 if _ms['n_unmatched'] else ""))

                # skip merge_duplicates when voxelizer subsumes it
                if merge_processor is not None and not (voxelizer and voxelizer.merge_duplicates):
                    _t = time.perf_counter()
                    particles = merge_processor.process(particles)
                    _ev_t['merge_dup'] = time.perf_counter() - _t
                    profile['merge_duplicates'] += _ev_t['merge_dup']

                if voxelizer is not None:
                    _t = time.perf_counter()
                    particles = voxelizer.process(particles)
                    _ev_t['voxelize'] = time.perf_counter() - _t
                    profile['voxelize'] += _ev_t['voxelize']
                    if _write_voxmap:
                        _t = time.perf_counter()
                        vox_writer.append_event(voxelizer.last_diagnostics)
                        _ev_t['write_voxmap'] = time.perf_counter() - _t
                        profile['write_voxmap'] += _ev_t['write_voxmap']

                if preprocessor is not None:
                    _t = time.perf_counter()
                    particles = preprocessor.process(particles)
                    _ev_t['defragment'] = time.perf_counter() - _t
                    profile['defragment'] += _ev_t['defragment']

                # ── Orphan resolution: repair dangling parent_id / root_id ──────
                # Must run before ParticlePartitioner so the genealogy tree is
                # consistent for all conditions and merge_em_showers.
                resolve_orphans(particles, verbose=cfg.verbose)

                # ── Particle statistics (after preprocessing, before partition) ──
                _pc_lens = [len(p.point_cloud) for p in particles]
                _nonzero = [s for s in _pc_lens if s > 0]

                # ── Empty-image guard ───────────────────────────────────────
                # Every particle having an empty point cloud means the event
                # holds no charge at all.  This is nearly always a config
                # error (wrong seg/inst file, wrong step_key) rather than
                # physics, so halt unless explicitly allowed.
                if not _nonzero:
                    _msg = (
                        f"Event {event_idx}: empty image — all "
                        f"{len(particles)} particle(s) have an empty point "
                        f"cloud, so the event contains no points."
                    )
                    if not _allow_empty_image:
                        raise RuntimeError(
                            f"{_msg}  This usually indicates a configuration "
                            f"error rather than real physics: check "
                            f"io.input_path, reader.step_key/ass_key, and (in "
                            f"JAXTPC mode) that reader.jaxtpc_seg_path and "
                            f"reader.jaxtpc_inst_path point at the seg/step and "
                            f"hits/inst files respectively.  Set "
                            f"allow_empty_image=true to downgrade this to a "
                            f"one-time warning."
                        )
                    if not _warned_empty_image:
                        import warnings
                        warnings.warn(
                            f"{_msg}  allow_empty_image=true, so processing "
                            f"continues; this warning is issued only once.",
                            UserWarning,
                            stacklevel=2,
                        )
                        _warned_empty_image = True

                stats['n_particles_in'].append(len(particles))
                stats['n_nonzero_pc'].append(len(_nonzero))
                stats['pc_size_min'].append(min(_nonzero) if _nonzero else 0)
                stats['pc_size_max'].append(max(_nonzero) if _nonzero else 0)
                stats['pc_size_mean'].append(
                    sum(_nonzero) / len(_nonzero) if _nonzero else 0.0
                )

                # ParticlePartitioner creates and initialises the checker
                # internally; backend-specific kwargs are forwarded via **kwargs.
                _t = time.perf_counter()
                partitioner = ParticlePartitioner(
                    particles          = particles,
                    distance_threshold = cfg.distance_threshold,
                    backend            = cfg.checker.name,
                    n_jobs             = int(cfg.checker.get("n_jobs", -1)),
                    enable_diagnostics = cfg.enable_diagnostics,
                    verbose            = cfg.verbose,
                    chunk_size         = int(cfg.checker.get("chunk_size", 512)),
                    block_size         = int(cfg.checker.get("block_size", 128)),
                )
                _ev_t['init'] = time.perf_counter() - _t
                profile['partitioner_init'] += _ev_t['init']

                _t = time.perf_counter()
                partitions = partitioner.partition_combined(conditions, verbose=cfg.verbose)
                _ev_t['partition'] = time.perf_counter() - _t
                profile['partition_combined'] += _ev_t['partition']
                stats['n_partitions'].append(len(partitions))

                _t = time.perf_counter()
                partitioner.checker.cleanup()
                _ev_t['cleanup'] = time.perf_counter() - _t
                profile['checker_cleanup'] += _ev_t['cleanup']

                # ── Step-1 fragment representatives ────────────────────────
                # Deduplicate: rep_lookup maps every original particle ID to
                # its representative; collect unique reps via identity dict.
                fragments = list(
                    {id(r): r for r in partitioner.rep_lookup.values()}.values()
                )

                # Snapshot fragment member_ids BEFORE merge_em_showers mutates
                # them in place (it extends the initiator's member_ids with the
                # merged members), so the _frag_lookup we build afterwards
                # reflects the original step-1 partitioning.
                _frag_member_ids_snap = {
                    id(_fr): (list(_fr.member_ids) if _fr.member_ids is not None else [_fr.id])
                    for _fr in fragments
                }

                # Write fragments BEFORE step-2 merger, which mutates
                # the initiator's member_ids / point_cloud in place.
                # drop_le_scatter only affects output — algorithms always
                # run on the full set of fragments/instances.
                _frags_out = [f for f in fragments if not _is_le(f)] if _drop_le_scatter else fragments

                # Build frag_lookup early so we can stamp parent_frag_id on
                # reps before writing. Use the pre-step-2 snapshot so that
                # merge_em_showers' in-place mutation of member_ids does not
                # corrupt frag_id assignments.
                # particle_id → fragment index in written list (in event)
                _frag_lookup: dict = {}
                for _fi, _fr in enumerate(_frags_out):
                    for _pid in _frag_member_ids_snap[id(_fr)]:
                        _frag_lookup[int(_pid)] = _fi

                # particle_id lookup for parent-chain walks
                _part_by_id: dict = {int(_p.id): _p for _p in particles}

                def _find_parent_idx(_start_pid, _lookup, _pby):
                    """Walk parent_id chain; return first valid hit index, or -1."""
                    _visited: set = set()
                    _pid = _start_pid
                    while _pid is not None and _pid not in _visited:
                        _visited.add(_pid)
                        if _pid in _lookup:
                            return _lookup[_pid]
                        _pp = _pby.get(int(_pid))
                        if _pp is None or _pp.parent_id == _pp.id:
                            break
                        _pid = _pp.parent_id
                    return -1

                for _fi, _fr in enumerate(_frags_out):
                    _pfid = _find_parent_idx(_fr.parent_id, _frag_lookup, _part_by_id)
                    _fr.parent_frag_id = _pfid if _pfid >= 0 else _fi  # fallback = own

                _t = time.perf_counter()
                writer.append_fragments(_frags_out)
                _ev_t['write_frags'] = time.perf_counter() - _t
                profile['write_fragments'] += _ev_t['write_frags']

                # ── Step-2 EM shower instance merger ───────────────────────
                _t = time.perf_counter()
                instances = merge_em_showers(fragments)
                _ev_t['em_merge'] = time.perf_counter() - _t
                profile['merge_em_showers'] += _ev_t['em_merge']

                # Output filtering only — does not affect algorithm logic.
                _insts_out = [i for i in instances if not _is_le(i)] if _drop_le_scatter else instances

                # Build inst_lookup and stamp parent_inst_id (and parent_frag_id
                # for the instance rep's own fragment) before writing.
                _inst_lookup: dict = {}  # particle_id → instance index (in event)
                for _ii, _inst in enumerate(_insts_out):
                    for _pid in (_inst.member_ids if _inst.member_ids is not None else [_inst.id]):
                        _inst_lookup[int(_pid)] = _ii

                for _ii, _inst in enumerate(_insts_out):
                    _piid = _find_parent_idx(_inst.parent_id, _inst_lookup, _part_by_id)
                    _inst.parent_inst_id = _piid if _piid >= 0 else _ii  # fallback = own
                    _inst.parent_frag_id = _frag_lookup.get(int(_inst.id), -1)

                _t = time.perf_counter()
                writer.append_instances(_insts_out)
                _ev_t['write_inst'] = time.perf_counter() - _t
                profile['write_instances'] += _ev_t['write_inst']

                # ── Event-level union point clouds ──────────────────────
                if _write_event_clouds:
                    _t = time.perf_counter()

                    # _frag_lookup and _inst_lookup are already built above
                    # for parent_frag_id / parent_inst_id stamping; reuse them
                    # here so cloud col indices match the HDF5 table rows.
                    # LE particles get frag_id=-1 / inst_id=-1 when
                    # drop_le_scatter=True — that is intentional.

                    # Assert algorithmic correctness: every non-LE particle must
                    # appear in SOME instance produced by merge_em_showers, even if
                    # that instance is a kLEScatter one that gets filtered from output
                    # by drop_le_scatter.  Use the full instances list here, not the
                    # output-filtered _insts_out, so the check is independent of the
                    # drop_le_scatter flag.
                    _inst_lookup_full: dict = {}
                    for _ii, _inst in enumerate(instances):
                        for _pid in (_inst.member_ids if _inst.member_ids is not None else [_inst.id]):
                            _inst_lookup_full[int(_pid)] = _ii
                    _nle_ids = [int(_p.id) for _p in particles if not _is_le(_p)]
                    _uncovered = [_pid for _pid in _nle_ids if _pid not in _inst_lookup_full]
                    if _uncovered:
                        raise RuntimeError(
                            f"Event {event_idx}: {len(_uncovered)} non-LE particle(s) have no "
                            f"instance after step-2 merger: "
                            f"{_uncovered[:10]}"
                            f"{'...' if len(_uncovered) > 10 else ''}"
                        )

                    def _build_cloud9(src_parts):
                        """Return (N, 9) float32 array:
                        x, y, z, t, energy, interaction_id, root_id, frag_id, inst_id."""
                        rows = []
                        for _p in src_parts:
                            _pc = _p.point_cloud
                            if len(_pc) == 0:
                                continue
                            _n = len(_pc)
                            _pc5 = np.zeros((_n, 5), dtype=np.float32)
                            _take = min(_pc.shape[1], 5)
                            _pc5[:, :_take] = _pc[:, :_take]
                            _pid_int = int(_p.id)
                            _iid = np.full((_n, 1), float(_p._interaction_id),            dtype=np.float32)
                            _rid = np.full((_n, 1), float(_p.root_id),                    dtype=np.float32)
                            _fid = np.full((_n, 1), float(_frag_lookup.get(_pid_int, -1)), dtype=np.float32)
                            _eid = np.full((_n, 1), float(_inst_lookup.get(_pid_int, -1)), dtype=np.float32)
                            rows.append(np.concatenate([_pc5, _iid, _rid, _fid, _eid], axis=1))
                        return np.concatenate(rows) if rows else np.empty((0, 9), dtype=np.float32)

                    def _voxelize_cloud9(c9, vox_size, orig):
                        """Voxelize (N, 9) cloud; integer-id cols (5-8) use majority-vote."""
                        if len(c9) == 0:
                            return c9
                        c5_vox = _voxelize_point_cloud(c9[:, :5], vox_size, orig)
                        coords = c9[:, :3]
                        _org = coords.min(axis=0) if orig is None else orig
                        _idx = np.floor((coords - _org) / vox_size).astype(np.int64)
                        _, _inv = np.unique(_idx, axis=0, return_inverse=True)
                        extra_cols = []
                        for _col in range(5, 9):  # interaction_id, root_id, frag_id, inst_id
                            # Q4: use majority-vote (mode) so a voxel shared by multiple
                            # instances/fragments gets assigned the dominant ID, not the min.
                            _int_vals = c9[:, _col].astype(np.int32)
                            _pairs = np.column_stack([_inv, _int_vals])
                            _upairs, _ucnts = np.unique(_pairs, axis=0, return_counts=True)
                            _vox_col = _upairs[:, 0]
                            _val_col = _upairs[:, 1]
                            # Sort so highest-count entry per voxel comes first
                            _sort_idx = np.lexsort([-_ucnts, _vox_col])
                            _vox_sorted = _vox_col[_sort_idx]
                            _val_sorted = _val_col[_sort_idx]
                            _, _first = np.unique(_vox_sorted, return_index=True)
                            _out = _val_sorted[_first].astype(np.float32)
                            extra_cols.append(_out[:, None])
                        return np.concatenate([c5_vox] + extra_cols, axis=1)

                    _nle_parts = [p for p in particles if not _is_le(p)]
                    _le_parts  = [p for p in particles if _is_le(p)]
                    _nle_cloud = _build_cloud9(_nle_parts)
                    _le_cloud  = _build_cloud9(_le_parts)
                    if voxelizer is not None:
                        _nle_cloud = _voxelize_cloud9(_nle_cloud, voxelizer.voxel_size, voxelizer.origin)
                        _le_cloud  = _voxelize_cloud9(_le_cloud,  voxelizer.voxel_size, voxelizer.origin)
                    writer.append_event_clouds(_nle_cloud, _le_cloud)
                    _ev_t['write_clouds'] = time.perf_counter() - _t
                    profile['write_event_clouds'] += _ev_t['write_clouds']

                _t = time.perf_counter()
                _particles_to_write = (
                    []
                    if _output_compact
                    else ([p for p in particles if not _is_le(p)]
                          if _drop_le_scatter else particles)
                )
                writer.append_event(_particles_to_write)
                _ev_t['write_ev'] = time.perf_counter() - _t
                profile['write_event'] += _ev_t['write_ev']

                # Update progress bar with per-event stats
                _bar.update(1)
                _bar.set_postfix(
                    p=len(particles),
                    parts=len(partitions),
                    prt_ms=f"{_ev_t.get('partition', 0)*1e3:.0f}",
                    wev_ms=f"{_ev_t.get('write_ev', 0)*1e3:.0f}",
                )

            _bar.close()

    # ── Post-run repack (opt-in) ───────────────────────────────────────────
    # Runs only after every writer above has been closed, so the file is
    # complete and every dataset's final length is known.
    _repack_cfg = cfg.get("repack", {}) or {}
    if bool(_repack_cfg.get("enabled", False)):
        from pysupera.io import repack, voxmap_path
        _rc_comp = _repack_cfg.get("compression", None)
        _rc_opts = _repack_cfg.get("compression_opts", None)
        _targets = [cfg.io.output_path]
        if _write_voxmap:
            _targets.append(voxmap_path(cfg.io.output_path))
        for _tgt in _targets:
            repack(str(_tgt),
                   compression=None if _rc_comp in (None, "") else str(_rc_comp),
                   compression_opts=None if _rc_opts is None else int(_rc_opts))

    # ── End-of-run summary report ──────────────────────────────────────────
    if n_events > 0 and cfg.get("report", True):
        _W   = 34                          # label column width
        _SEP = "-" * (_W + 35)

        def _agg(vals):
            """Return (min, mean, max) for a list of numbers."""
            return min(vals), sum(vals) / len(vals), max(vals)

        # ── Particle / partition statistics ───────────────────────────────
        print(f"\n[run] Run statistics  ({n_events} event(s))")
        print(f"  {'Metric':<{_W}} {'min':>10} {'mean':>10} {'max':>10}")
        print(f"  {_SEP}")

        _int_rows = [
            ("Particles in",            stats['n_particles_in']),
            ("Particles w/ non-zero PC", stats['n_nonzero_pc']),
            ("PC size  min",             stats['pc_size_min']),
            ("PC size  max",             stats['pc_size_max']),
        ]
        for label, vals in _int_rows:
            lo, mu, hi = _agg(vals)
            print(f"  {label:<{_W}} {int(lo):>10,} {mu:>10.1f} {int(hi):>10,}")

        lo, mu, hi = _agg(stats['pc_size_mean'])
        print(f"  {'PC size  mean':<{_W}} {lo:>10.2f} {mu:>10.2f} {hi:>10.2f}")

        print(f"  {_SEP}")
        lo, mu, hi = _agg(stats['n_partitions'])
        print(f"  {'Partitions (final)':<{_W}} {int(lo):>10,} {mu:>10.1f} {int(hi):>10,}")

        # ── JAXTPC readout visibility (only when the reader reported it) ───
        if mask_stats.get('n_total'):
            print(f"\n[run] JAXTPC segment visibility  ({n_events} event(s))")
            print(f"  {'Metric':<{_W}} {'min':>10} {'mean':>10} {'max':>10}")
            print(f"  {_SEP}")
            for label, key in (
                ("Segments in seg file",     'n_total'),
                ("Visible (above threshold)", 'n_visible'),
                ("Masked out (undetected)",  'n_masked'),
                ("Attached to a particle",   'n_attached'),
            ):
                lo, mu, hi = _agg(mask_stats[key])
                print(f"  {label:<{_W}} {int(lo):>10,} {mu:>10.1f} {int(hi):>10,}")

            _tot = sum(mask_stats['n_total'])
            _msk = sum(mask_stats['n_masked'])
            _unm = sum(mask_stats['n_unmatched'])
            print(f"  {_SEP}")
            print(f"  {'Masked fraction (all events)':<{_W}} "
                  f"{100.0 * _msk / max(1, _tot):>9.2f}%")
            if _unm:
                # Visible in the readout but carrying a track ID absent from
                # the EDepSim particle list, so they reach no point cloud.
                print(f"  {'Visible but unmatched':<{_W}} "
                      f"{_unm:>10,} ({100.0 * _unm / max(1, _tot):.2f}%)")

        # ── Time profile ──────────────────────────────────────────────────
        total = sum(profile.values())
        print(f"\n[run] Time profile  ({n_events} event(s))")
        print(f"  {'Stage':<{_W}} {'Total (s)':>10} {'Per event (ms)':>15} {'%':>7}")
        print(f"  {_SEP}")
        for stage, secs in sorted(profile.items(), key=lambda x: -x[1]):
            pct = 100.0 * secs / total if total > 0 else 0.0
            print(f"  {stage:<{_W}} {secs:>10.3f} {1000*secs/n_events:>15.2f} {pct:>6.1f}%")
        print(f"  {_SEP}")
        print(f"  {'TOTAL':<{_W}} {total:>10.3f} {1000*total/n_events:>15.2f} {'100.0':>6}%")

    print("[run] Done.")


if __name__ == "__main__":
    main()
