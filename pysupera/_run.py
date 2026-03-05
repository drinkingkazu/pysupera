"""
CLI entry point for the nusupera partitioning workflow.

After installation (``pip install -e .``) this is available as a shell
command:

    run_pysupera io.input_path=/data/in.h5 io.output_path=/data/out.h5
    run_pysupera checker=cpu_multi checker.n_jobs=8
    run_pysupera distance_threshold=3.0 conditions.touching_le_scatter=false

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

@hydra.main(config_path=_CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Partition all events in the input file and write results."""
    import time
    from collections import defaultdict
    from pysupera import read_events, open_writer
    from pysupera.partitioner import ParticlePartitioner
    from pysupera.config import build_conditions, build_preprocessor, build_merge_processor, configure

    configure(cfg)  # set module-level defaults (e.g. min_pc_size) before any Particle is created

    merge_processor = build_merge_processor(cfg)  # None when merge_duplicates: false
    preprocessor    = build_preprocessor(cfg)      # None when defragment: false

    print(f"[run] checker         : {cfg.checker.name}")
    print(f"[run] distance        : {cfg.distance_threshold}")
    print(f"[run] merge_duplicates: {'enabled' if merge_processor is not None else 'disabled'}")
    print(f"[run] preprocessor    : {cfg.particle.get('preprocessor', {}).get('name', 'scipy') if cfg.particle.get('defragment', False) else 'disabled'}")
    print(f"[run] input         : {cfg.io.input_path}")
    print(f"[run] output        : {cfg.io.output_path}")

    conditions = build_conditions(cfg)

    # ── Accumulators (always collected; printed only when report=True) ──────
    profile: dict = defaultdict(float)   # stage name → cumulative wall-clock s
    stats:   dict = {
        'n_particles_in': [],   # int  : total particles entering partitioner
        'n_nonzero_pc':   [],   # int  : particles with at least one point
        'pc_size_min':    [],   # int  : smallest non-zero PC in event
        'pc_size_max':    [],   # int  : largest non-zero PC in event
        'pc_size_mean':   [],   # float: mean non-zero PC size in event
        'n_partitions':   defaultdict(list),  # condition.name → [int per event]
    }
    n_events = 0

    with read_events(cfg.io.input_path) as store:
        print(f"[run] {len(store)} events in input file")

        with open_writer(cfg.io.output_path,
                         compression=cfg.io.compression,
                         compression_opts=cfg.io.compression_opts) as writer:

            for event_idx, particles in enumerate(store.iter_events()):
                t0 = time.perf_counter()
                n_events += 1

                if merge_processor is not None:
                    _t = time.perf_counter()
                    particles = merge_processor.process(particles)
                    profile['merge_duplicates'] += time.perf_counter() - _t

                if preprocessor is not None:
                    _t = time.perf_counter()
                    particles = preprocessor.process(particles)
                    profile['defragment'] += time.perf_counter() - _t

                # ── Particle statistics (after preprocessing, before partition) ──
                _pc_lens = [len(p.point_cloud) for p in particles]
                _nonzero = [s for s in _pc_lens if s > 0]
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
                    chunk_size         = int(cfg.checker.get("chunk_size", 512)),
                    block_size         = int(cfg.checker.get("block_size", 128)),
                )
                profile['partitioner_init'] += time.perf_counter() - _t

                for condition in conditions:
                    _t = time.perf_counter()
                    partitions = partitioner.partition(condition, verbose=cfg.verbose)
                    profile[condition.name] += time.perf_counter() - _t
                    stats['n_partitions'][condition.name].append(len(partitions))

                partitioner.checker.cleanup()

                # Write the (possibly relabelled) particles for this event.
                writer.append_event(particles)

                if cfg.verbose:
                    elapsed = time.perf_counter() - t0
                    print(f"[run]   event {event_idx:>6d} "
                          f"| {len(particles):>5d} particles "
                          f"| {elapsed:.3f} s")

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
        for cname, counts in stats['n_partitions'].items():
            lo, mu, hi = _agg(counts)
            label = f"Partitions [{cname}]"
            print(f"  {label:<{_W}} {int(lo):>10,} {mu:>10.1f} {int(hi):>10,}")

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
