"""
pysupera Dash visualisation app.

Launch with::

    pysupera-app                        # default port 8050
    pysupera-app --port 8080
    pysupera-app --host 0.0.0.0         # expose on network

Then open http://localhost:8050 in a browser.

Controls (sidebar)
------------------
* HDF5 file path + event index
* Input format:

  - ``native``     — pysupera native HDF5 (default)
  - ``edepsim_h5`` — EDepSim HDF5 output; exposes step_key, particle_key,
                     electron_energy_threshold, and two optional JAXTPC
                     visibility-filter fields (jaxtpc_seg_path /
                     jaxtpc_inst_path).  Leave the JAXTPC fields blank to
                     use all EDepSim segments; fill both to restrict point
                     clouds to segments visible in the JAXTPC readout.

* Preprocessing: merge_duplicates, defragment, min_pc_size, backend
* Partitioner: distance_threshold, checker type, n_jobs
* Conditions: enable / disable each of the four conditions
* Run button

Visualisation (main panel)
--------------------------
Left subplot  – original particles, one colour per particle.
Right subplot – partitioned result, one colour per partition.
The two 3-D cameras are synchronised: rotating / zooming one panel mirrors
the other automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import shutil
import time
import traceback
from collections import OrderedDict, defaultdict

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Logo: auto-copy supera.png into the Dash assets folder on first run
# ---------------------------------------------------------------------------

_PKG_DIR    = os.path.dirname(__file__)
_ASSETS_DIR = os.path.join(_PKG_DIR, "assets")
os.makedirs(_ASSETS_DIR, exist_ok=True)
_LOGO_SRC  = os.path.join(_PKG_DIR, "..", "figures", "supera.png")
_LOGO_DEST = os.path.join(_ASSETS_DIR, "supera.png")
if os.path.isfile(_LOGO_SRC) and not os.path.isfile(_LOGO_DEST):
    shutil.copy2(_LOGO_SRC, _LOGO_DEST)
_LOGO_URL = "/assets/supera.png" if os.path.isfile(_LOGO_DEST) else None

# ---------------------------------------------------------------------------
# Checker backend name  →  Hydra config-group key
# ---------------------------------------------------------------------------

_CHECKER_CFG: dict[str, str] = {
    "cpu-single":           "cpu_single",
    "cpu-multi":            "cpu_multi",
    "gpu":                  "gpu",
    "bulk-gpu":             "bulk_gpu",
    "numba":                "numba",
    "cell-hash-cpu-single": "cell_hash_cpu_single",
    "cell-hash-cpu-multi":  "cell_hash_cpu_multi",
    "cell-hash-gpu":        "cell_hash_gpu",
}

# Checkers that accept an n_jobs override in their Hydra schema
_MULTI_CHECKERS = {"cpu-multi", "cell-hash-cpu-multi"}

# ---------------------------------------------------------------------------
# Fixed colours per SemanticType
# ---------------------------------------------------------------------------

_SEM_COLORS: dict[str, str] = {
    "kShower":    "#e74c3c",
    "kTrack":     "#3498db",
    "kDelta":     "#2ecc71",
    "kMichel":    "#f39c12",
    "kLEScatter": "#9b59b6",
    "kUnknown":   "#95a5a6",
}

# ---------------------------------------------------------------------------
# Preprocessing cache  (LRU, keyed by all params that affect load + preproc)
# ---------------------------------------------------------------------------

_MAX_CACHE_ENTRIES = 4
# Maps cache-key → (particles_for_display, log_lines_from_load_and_preproc)
_PARTICLE_CACHE: OrderedDict[str, tuple[list, list[str]]] = OrderedDict()


def _preproc_cache_key(
    file_path, event_idx, format_val,
    step_key, particle_key, elec_thresh,
    preproc_flags, min_pc_size, preproc_backend,
    sem_types, voxel_size,
) -> str:
    key_data = (
        file_path,
        int(event_idx),
        format_val,
        step_key,
        particle_key,
        float(elec_thresh),
        tuple(sorted(preproc_flags)),
        int(min_pc_size),
        preproc_backend,
        tuple(sorted(sem_types)),
        float(voxel_size),
    )
    return hashlib.md5(str(key_data).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixed colours per SemanticType (used in the merged left subplot)
# ---------------------------------------------------------------------------

_SEM_COLORS: dict[str, str] = {
    "kShower":    "#e74c3c",
    "kTrack":     "#3498db",
    "kDelta":     "#2ecc71",
    "kMichel":    "#f39c12",
    "kLEScatter": "#9b59b6",
    "kUnknown":   "#95a5a6",
}

# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def _make_colorscale_array(n: int) -> list[str]:
    """Return *n* distinct hex colours sampled from Plotly's alphabet palette."""
    palette = plotly.colors.qualitative.Alphabet  # 26 colours
    return [palette[i % len(palette)] for i in range(n)]


def _build_figure(
    particles_raw: list,
    partitions: list | None,
    *,
    marker_size: int = 2,
    show_legend: bool = True,
    draw_mode: str = "by_instance",  # "by_instance" | "by_sem_type"
) -> go.Figure:
    """
    Construct the side-by-side Plotly figure.

    draw_mode="by_instance":
        Left  – one trace per particle, coloured by particle index.
        Right – one trace per partition, coloured by partition index.
    draw_mode="by_sem_type":
        Left  – one trace per SemanticType with fixed colours.
        Right – one trace per SemanticType (all-partition points merged by type).
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Original particles", "Partitions"],
    )

    N_raw   = len(particles_raw)
    N_parts = len(partitions) if partitions else 0
    colours_raw   = _make_colorscale_array(N_raw)
    colours_parts = _make_colorscale_array(N_parts)

    def _collect_sem_groups(particle_iter):
        """Bucket points by sem_type name; returns dict[name -> (xs, ys, zs)]."""
        groups: dict[str, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        for p in particle_iter:
            pc = p.point_cloud
            if pc is None or len(pc) == 0:
                continue
            xs, ys, zs = groups[p.sem_type.name]
            xs.append(pc[:, 0])
            ys.append(pc[:, 1])
            zs.append(pc[:, 2])
        return groups

    # ── Left subplot ───────────────────────────────────────────────────────
    if draw_mode == "by_sem_type":
        for sem_name, (xs, ys, zs) in _collect_sem_groups(particles_raw).items():
            fig.add_trace(
                go.Scatter3d(
                    x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs),
                    mode="markers",
                    marker=dict(size=marker_size,
                                color=_SEM_COLORS.get(sem_name, "#e0e0e0")),
                    name=sem_name,
                    legendgroup=f"raw_{sem_name}",
                    legend="legend",
                    showlegend=show_legend,
                    meta={"sem_type": sem_name},
                    hovertemplate=(
                        f"sem={sem_name}<br>"
                        "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
    else:  # by_instance
        for i, p in enumerate(particles_raw):
            pc = p.point_cloud
            if pc is None or len(pc) == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=colours_raw[i]),
                    name=f"p{p.id}  {p.sem_type.name}",
                    legendgroup=f"raw_{i}",
                    legend="legend",
                    showlegend=show_legend,
                    meta={"sem_type": p.sem_type.name},
                    hovertemplate=(
                        f"id={p.id}  pdg={p.pdg}  sem={p.sem_type.name}<br>"
                        f"parent_id={p.parent_id}  root_id={p.root_id}<br>"
                        "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

    # ── Right subplot ──────────────────────────────────────────────────────
    if partitions:
        if draw_mode == "by_sem_type":
            all_part_particles = (p for part in partitions for p in part)
            for sem_name, (xs, ys, zs) in _collect_sem_groups(all_part_particles).items():
                fig.add_trace(
                    go.Scatter3d(
                        x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs),
                        mode="markers",
                        marker=dict(size=marker_size,
                                    color=_SEM_COLORS.get(sem_name, "#e0e0e0")),
                        name=sem_name,
                        legendgroup=f"part_sem_{sem_name}",
                        legend="legend2",
                        showlegend=show_legend,
                        meta={"sem_type": sem_name},
                        hovertemplate=(
                            f"sem={sem_name}<br>"
                            "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<extra></extra>"
                        ),
                    ),
                    row=1, col=2,
                )
        else:  # by_instance
            for i, part in enumerate(partitions):
                chunks = [p.point_cloud for p in part
                          if p.point_cloud is not None and len(p.point_cloud) > 0]
                if not chunks:
                    continue
                pts = np.concatenate(chunks, axis=0)
                rep = max(
                    (p for p in part if p.point_cloud is not None and len(p.point_cloud) > 0),
                    key=lambda p: len(p.point_cloud),
                    default=part[0],
                )
                ids_str = ",".join(str(p.id) for p in part[:8])
                if len(part) > 8:
                    ids_str += f"+{len(part)-8}"
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker=dict(size=marker_size, color=colours_parts[i]),
                        name=f"part {i}  ({len(part)}p)",
                        legendgroup=f"part_{i}",
                        legend="legend2",
                        showlegend=show_legend,
                        meta={"sem_type": rep.sem_type.name},
                        hovertemplate=(
                            f"partition {i}  ({len(part)} particles)<br>"
                            f"rep: id={rep.id}  sem={rep.sem_type.name}<br>"
                            f"rep parent_id={rep.parent_id}  root_id={rep.root_id}<br>"
                            f"members: [{ids_str}]<br>"
                            "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<extra></extra>"
                        ),
                    ),
                    row=1, col=2,
                )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        uirevision="keep",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        hoverlabel=dict(font=dict(size=15), namelength=-1),
        # Left legend anchored inside left subplot (x 0..0.5)
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10),
            title=dict(text="Original"),
            visible=show_legend,
        ),
        # Right legend anchored inside right subplot (x 0.5..1)
        legend2=dict(
            x=0.51, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10),
            title=dict(text="Partitions"),
            visible=show_legend,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

_SIDEBAR_STYLE = dict(
    width="270px",
    minWidth="270px",
    padding="12px",
    backgroundColor="#16213e",
    color="#e0e0e0",
    overflowY="auto",
    fontSize="13px",
    display="flex",
    flexDirection="column",
    gap="6px",
)

_SECTION_STYLE = dict(
    borderTop="1px solid #0f3460",
    paddingTop="8px",
    marginTop="4px",
)

_LABEL_STYLE = dict(color="#a8b2d8", marginBottom="2px", display="block")

_INPUT_STYLE = dict(
    width="100%",
    backgroundColor="#0f3460",
    color="#e0e0e0",
    border="1px solid #1a4a8a",
    borderRadius="4px",
    padding="4px 6px",
    boxSizing="border-box",
)

_BTN_STYLE = dict(
    width="100%",
    padding="10px",
    backgroundColor="#e94560",
    color="white",
    border="none",
    borderRadius="6px",
    cursor="pointer",
    fontSize="14px",
    fontWeight="bold",
    marginTop="8px",
)


def _section(title: str, *children, collapsed: bool = False) -> html.Details:
    """Collapsible sidebar section using native <details>/<summary>."""
    return html.Details([
        html.Summary(title, style=dict(
            color="#e94560", fontWeight="bold",
            fontSize="12px", letterSpacing="0.05em",
            cursor="pointer", userSelect="none",
            paddingBottom="6px",
        )),
        html.Div(list(children), style=dict(paddingLeft="2px")),
    ], open=(not collapsed), style=_SECTION_STYLE)


def _labelled(label: str, component) -> html.Div:
    return html.Div([
        html.Label(label, style=_LABEL_STYLE),
        component,
    ])


def _input(id_: str, type_="text", value=None, **kw) -> dcc.Input:
    return dcc.Input(
        id=id_, type=type_, value=value,
        style=_INPUT_STYLE,
        persistence=True, persistence_type="session",
        **kw
    )


def _dropdown(id_: str, options: list[dict], value) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=id_,
        options=options,
        value=value,
        clearable=False,
        style=dict(backgroundColor="#0f3460", color="#e0e0e0"),
    )


def build_layout() -> html.Div:
    sidebar = html.Div([
        # ── Logo at top of sidebar ───────────────────────────────────────
    ] + ([
        html.Img(src=_LOGO_URL, style=dict(
            width="80%", maxWidth="180px",
            display="block", margin="4px auto 10px auto",
            opacity="0.85",
        ))
    ] if _LOGO_URL else []) + [

        # ── I/O ──────────────────────────────────────────────────────────
        _section(
            "INPUT",
            _labelled("HDF5 file", _input("file-path", value="one.h5")),
            _labelled("Event index", _input("event-idx", "number", 0, min=0, step=1)),
        ),

        # ── Input format ──────────────────────────────────────────────────
        # Input format:
        # - native     : pysupera native HDF5
        # - edepsim_h5 : EDepSim HDF5; optionally add JAXTPC visibility
        #                filtering by filling the seg/inst path fields below.
        _section(
            "INPUT FORMAT",
            _labelled("Format",
                _dropdown("format-selector", [
                    {"label": "native (pysupera HDF5)", "value": "native"},
                    {"label": "EDepSim HDF5",            "value": "edepsim_h5"},
                ], "native")),
            # EDepSim options (shown when format == edepsim_h5)
            html.Div([
                _labelled("step_key (HDF5 dataset)",
                    _input("step-key", value="pstep/lar_vol")),
                _labelled("particle_key (HDF5 dataset)",
                    _input("particle-key", value="particle/geant4")),
                _labelled("electron_energy_threshold",
                    _input("elec-threshold", "number", 0.05, min=0.0, step="any")),
                html.Hr(style={"borderColor": "#0f3460", "margin": "6px 0"}),
                # JAXTPC visibility filtering — leave blank to use all EDepSim
                # segments; fill both paths to restrict point clouds to only the
                # segments that were visible in the JAXTPC readout.
                html.Div("JAXTPC visibility filter (optional):",
                         style=dict(fontSize="11px", color="#5a6a9a",
                                    marginBottom="4px")),
                _labelled("jaxtpc_seg_path",
                    _input("jaxtpc-seg-path", value="")),
                _labelled("jaxtpc_inst_path",
                    _input("jaxtpc-inst-path", value="")),
            ], id="edepsim-options", style={"display": "none"}),
            collapsed=True,
        ),

        # ── Preprocessing (collapsed by default) ───────────────────────────────────────────
        _section(
            "PREPROCESSING",
            dcc.Checklist(
                id="preproc-flags",
                options=[
                    {"label": " merge duplicates", "value": "merge_duplicates"},
                    {"label": " voxelize",          "value": "voxelize"},
                    {"label": " defragment",        "value": "defragment"},
                ],
                value=["merge_duplicates", "defragment"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),
            html.Div([
                    _labelled("voxel_size",
                        _input("voxel-size", "number", 0.3, min=0.01, step="any")),
                ], id="voxelize-options", style={"display": "none"}),
            html.Div([
                    _labelled("min_pc_size",
                        _input("min-pc-size", "number", 10, min=1, step=1)),
                    _labelled("Backend",
                        _dropdown("preproc-backend", [
                            {"label": "scipy (CPU)",   "value": "scipy"},
                            {"label": "gpu (CuPy)",    "value": "gpu"},
                            {"label": "rapids (GPU)",  "value": "rapids"},
                        ], "scipy")),
                    _labelled("Apply only to sem_types (empty = all)",
                        dcc.Checklist(
                            id="sem-types",
                            options=[
                                {"label": " kShower",    "value": "kShower"},
                                {"label": " kTrack",     "value": "kTrack"},
                                {"label": " kDelta",     "value": "kDelta"},
                                {"label": " kMichel",    "value": "kMichel"},
                                {"label": " kLEScatter", "value": "kLEScatter"},
                            ],
                            value=["kShower", "kDelta", "kMichel", "kLEScatter"],
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"display": "block", "marginBottom": "2px"},
                        )),
                ], id="defrag-options", style={"display": "none"}),
            collapsed=True,
        ),

        # ── Partitioner ────────────────────────────────────────────────
        _section(
            "PARTITIONER",
            _labelled("Distance threshold",
                _input("dist-thresh", "number", 0.8, min=0.0, step="any")),
            _labelled("Checker",
                _dropdown("checker-type", [
                    {"label": "cpu-single",              "value": "cpu-single"},
                    {"label": "cpu-multi",               "value": "cpu-multi"},
                    {"label": "cell-hash-cpu-single",    "value": "cell-hash-cpu-single"},
                    {"label": "cell-hash-cpu-multi",     "value": "cell-hash-cpu-multi"},
                    {"label": "gpu (RAPIDS)",            "value": "gpu"},
                    {"label": "bulk-gpu (CuPy)",         "value": "bulk-gpu"},
                    {"label": "numba",                   "value": "numba"},
                ], "cpu-single")),
            html.Div(
                _labelled("n_jobs (−1 = all cores)",
                    _input("n-jobs", "number", -1, min=-1, step=1)),
                id="n-jobs-div", style={"display": "none"},
            ),
            collapsed=True,
        ),

        # ── Conditions (collapsed by default)
        _section(
            "CONDITIONS",
            dcc.Checklist(
                id="conditions",
                options=[
                    {"label": " PhotonDecay",        "value": "photon_decay"},
                    {"label": " TouchingEMShower",   "value": "touching_em_shower"},
                    {"label": " CombineLEScatters",  "value": "combine_le_scatters"},
                    {"label": " AbsorbLEScatter",    "value": "absorb_le_scatter"},
                ],
                value=["photon_decay", "touching_em_shower",
                       "combine_le_scatters", "absorb_le_scatter"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),
            collapsed=True,
        ),

        # ── View options ────────────────────────────────────────────────
        _section(
            "VIEW OPTIONS",
            dcc.Checklist(
                id="view-options",
                options=[
                    {"label": " show legend",        "value": "show_legend"},
                    {"label": " sync cameras",       "value": "sync_cameras"},
                    {"label": " colour by sem type", "value": "draw_by_sem"},
                ],
                value=["show_legend", "sync_cameras"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
                persistence=True, persistence_type="session",
            ),
        ),

        # ── Particle filter (by sem_type + min PC size) ──────────────────
        _section(
            "PARTICLE FILTER",
            html.Div("Uncheck to hide particles of that type:",
                     style=dict(fontSize="11px", color="#5a6a9a",
                                marginBottom="4px")),
            dcc.Checklist(
                id="mask-sem-types",
                options=[
                    {"label": " kShower",    "value": "kShower"},
                    {"label": " kTrack",     "value": "kTrack"},
                    {"label": " kDelta",     "value": "kDelta"},
                    {"label": " kMichel",    "value": "kMichel"},
                    {"label": " kLEScatter", "value": "kLEScatter"},
                ],
                value=["kShower", "kTrack", "kDelta", "kMichel", "kLEScatter"],
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "2px"},
                persistence=True, persistence_type="session",
            ),
            html.Div(
                _labelled(
                    "Min points to display (display-only filter)",
                    _input("min-display-pc", "number", 0, min=0, step=1),
                ),
                style={"marginTop": "8px"},
            ),
        ),

        # ── Run ────────────────────────────────────────────────────────
        html.Div([
            html.Button("▶  Run", id="run-btn", n_clicks=0, style=_BTN_STYLE),
            dcc.Checklist(
                id="verbose-toggle",
                options=[{"label": " verbose", "value": "verbose"}],
                value=[],
                inputStyle={"marginRight": "5px"},
                style={"display": "inline-block", "marginLeft": "12px",
                       "fontSize": "12px", "verticalAlign": "middle"},
            ),
        ], style={"display": "flex", "alignItems": "center",
                  "borderTop": "1px solid #0f3460",
                  "paddingTop": "8px", "marginTop": "4px"}),

        # ── Status badge ─────────────────────────────────────────────────
        html.Div(id="run-status", style=dict(
            marginTop="6px", fontSize="11px", color="#a8b2d8",
        )),
    ], style=_SIDEBAR_STYLE)

    main = html.Div([
        dcc.Graph(
            id="main-graph",
            figure=_build_figure([], None),
            config={"scrollZoom": True, "displayModeBar": True},
            style={"height": "100%"},
            responsive=True,
        ),
        # Invisible output nodes for clientside callbacks
        html.Div(id="sync-dummy", style={"display": "none"}),
        html.Div(id="mask-dummy", style={"display": "none"}),
    ], style=dict(flex="1", minWidth="0", minHeight="0"))

    return html.Div([
        # ── Header ───────────────────────────────────────────────────────
        html.Div([
            html.Span("pysupera", style=dict(
                fontWeight="bold", fontSize="20px", color="#e94560")),
            html.Span("  LArTPC particle partitioning visualizer",
                      style=dict(fontSize="13px", color="#a8b2d8")),
        ], style=dict(
            padding="10px 16px",
            backgroundColor="#0f3460",
            display="flex",
            alignItems="center",
        )),

        # ── Body ─────────────────────────────────────────────────────────
        html.Div([
            # ── Top: sidebar + 3-D plot (fills remaining vertical space) ───
            html.Div([sidebar, main], id="top-section", style=dict(
                display="flex",
                flex="1",
                minHeight="0",
                overflow="hidden",
                backgroundColor="#1a1a2e",
            )),

            # ── Drag handle between plot and log ────────────────────────
            html.Div(id="log-drag-handle", style=dict(
                height="6px", cursor="ns-resize",
                backgroundColor="#e94560", flexShrink="0",
            )),

            # ── Bottom: full-width collapsible log panel ─────────────────
            html.Div([
                # Title bar (always visible — click "show" to expand/collapse)
                html.Div([
                    html.Span("▼  OUTPUT LOG",
                              style=dict(fontWeight="bold",
                                         letterSpacing="0.05em")),
                    html.Span("  │  drag red bar above to resize",
                              style=dict(fontSize="11px", color="#5a6a9a",
                                         marginLeft="4px")),
                    dcc.Checklist(
                        id="log-show",
                        options=[{"label": " show", "value": "show"}],
                        value=["show"],
                        inputStyle={"marginRight": "5px"},
                        style={"marginLeft": "auto", "fontSize": "12px"},
                    ),
                ], style=dict(
                    display="flex", alignItems="center",
                    padding="4px 12px",
                    backgroundColor="#0f3460",
                    borderTop="2px solid #e94560",
                    fontSize="12px", fontWeight="bold", color="#a8b2d8",
                    userSelect="none",
                )),
                # Scrollable log body (resizable via CSS)
                html.Div(
                    html.Pre(id="log-content", style={
                        "margin": "0", "padding": "8px 12px",
                        "fontFamily": "'Courier New', monospace",
                        "whiteSpace": "pre-wrap",
                    }),
                    id="log-body",
                    style=dict(
                        height="220px", minHeight="0",
                        overflowY="auto",
                        backgroundColor="#0a0a1a",
                        color="#a8b2d8",
                        fontSize="12px", lineHeight="1.5",
                    ),
                ),
            ], style=dict(width="100%", flexShrink="0")),

        ], style=dict(
            display="flex", flexDirection="column",
            height="calc(100vh - 44px)",
            backgroundColor="#1a1a2e",
        )),
    ], style=dict(fontFamily="'Segoe UI', sans-serif",
                  backgroundColor="#1a1a2e", color="#e0e0e0"))


# ---------------------------------------------------------------------------
# App & callbacks
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="pysupera visualizer")
app.layout = build_layout()


# ── Show/hide defrag options ─────────────────────────────────────────────────
@app.callback(
    Output("defrag-options", "style"),
    Input("preproc-flags", "value"),
)
def toggle_defrag(flags):
    visible = {"display": "block", "marginTop": "6px"}
    hidden  = {"display": "none"}
    return visible if "defragment" in (flags or []) else hidden


# ── Show/hide voxelize options ────────────────────────────────────────────────
@app.callback(
    Output("voxelize-options", "style"),
    Input("preproc-flags", "value"),
)
def toggle_voxelize(flags):
    visible = {"display": "block", "marginTop": "6px"}
    hidden  = {"display": "none"}
    return visible if "voxelize" in (flags or []) else hidden


# ── Show/hide EDepSim reader options ─────────────────────────────────────────
# ── Show/hide EDepSim options (and embedded JAXTPC optional fields) ───────────
@app.callback(
    Output("edepsim-options", "style"),
    Input("format-selector", "value"),
)
def toggle_edepsim_options(fmt):
    visible = {"display": "block", "marginTop": "6px"}
    hidden  = {"display": "none"}
    return visible if fmt == "edepsim_h5" else hidden


# ── Show/hide n_jobs ─────────────────────────────────────────────────────────
@app.callback(
    Output("n-jobs-div", "style"),
    Input("checker-type", "value"),
)
def toggle_njobs(checker):
    multi = {"cpu-multi", "cell-hash-cpu-multi"}
    visible = {"display": "block"}
    hidden  = {"display": "none"}
    return visible if checker in multi else hidden


# ── Show/hide log body ─────────────────────────────────────────────────────
@app.callback(
    Output("log-body", "style"),
    Input("log-show", "value"),
)
def toggle_log(show_val):
    base = dict(
        minHeight="40px", overflowY="auto", resize="vertical",
        backgroundColor="#0a0a1a", color="#a8b2d8",
        fontSize="12px", lineHeight="1.5",
    )
    if "show" in (show_val or []):
        base["height"] = "220px"
    else:
        base["display"] = "none"
    return base


# ── Main run callback ────────────────────────────────────────────────────────
@app.callback(
    Output("main-graph",  "figure"),
    Output("run-status",  "children"),
    Output("log-content", "children"),
    Input("run-btn",      "n_clicks"),
    State("file-path",    "value"),
    State("event-idx",    "value"),
    State("preproc-flags","value"),
    State("min-pc-size",  "value"),
    State("preproc-backend", "value"),
    State("sem-types",    "value"),
    State("dist-thresh",  "value"),
    State("checker-type", "value"),
    State("n-jobs",       "value"),
    State("voxel-size",   "value"),
    State("conditions",   "value"),
    State("verbose-toggle", "value"),
    State("view-options",  "value"),
    State("format-selector",  "value"),
    State("step-key",         "value"),
    State("particle-key",     "value"),
    State("elec-threshold",   "value"),
    State("jaxtpc-seg-path",  "value"),
    State("jaxtpc-inst-path", "value"),
    State("min-display-pc",   "value"),
    prevent_initial_call=True,
)
def run_pipeline(
    n_clicks,
    file_path, event_idx,
    preproc_flags, min_pc_size, preproc_backend, sem_types_input,
    dist_thresh, checker_type, n_jobs, voxel_size_input,
    condition_keys, verbose_flags, view_options,
    format_val, step_key_val, particle_key_val, elec_thresh_val,
    jaxtpc_seg_path_val, jaxtpc_inst_path_val,
    min_display_pc_val,
):
    preproc_flags   = preproc_flags  or []
    condition_keys  = condition_keys or []
    sem_types_input = sem_types_input or []
    verbose_flags   = verbose_flags  or []
    view_options    = view_options   or []
    show_legend    = "show_legend" in view_options
    draw_mode      = "by_sem_type" if "draw_by_sem" in view_options else "by_instance"
    min_display_pc = int(min_display_pc_val) if min_display_pc_val is not None else 0
    event_idx      = int(event_idx   if event_idx   is not None else 0)
    dist_thresh     = float(dist_thresh if dist_thresh is not None else 0.8)
    n_jobs          = int(n_jobs      if n_jobs      is not None else -1)
    min_pc_size     = int(min_pc_size if min_pc_size is not None else 10)
    checker_type    = checker_type or "cpu-single"
    preproc_backend = preproc_backend or "scipy"
    verbose         = "verbose" in verbose_flags
    voxel_size_val  = float(voxel_size_input if voxel_size_input is not None else 0.3)
    format_val      = format_val or "native"
    step_key_val    = step_key_val    or "pstep/lar_vol"
    particle_key_val = particle_key_val or "particle/geant4"
    elec_thresh_val = float(elec_thresh_val) if elec_thresh_val is not None else 0.05

    lines: list[str] = []
    t_total = time.perf_counter()

    # ── Check preprocessing cache ─────────────────────────────────────────
    _preproc_key = _preproc_cache_key(
        file_path, event_idx, format_val,
        step_key_val, particle_key_val, elec_thresh_val,
        preproc_flags, min_pc_size, preproc_backend,
        sem_types_input, voxel_size_val,
    )
    _cached = _PARTICLE_CACHE.get(_preproc_key)

    # ── Build config via load_cfg (same path as the notebook) ────────────
    # load_cfg composes the full Hydra config with all defaults, then calls
    # configure(cfg) internally — which sets _DEFAULT_MIN_PC_SIZE before
    # any Particle objects are created from HDF5.
    try:
        from pysupera.config import load_cfg

        sem_types_str = (
            f"[{','.join(sem_types_input)}]" if sem_types_input else "[]"
        )
        overrides = [
            f"particle.min_pc_size={min_pc_size}",
            f"particle.merge_duplicates={'true' if 'merge_duplicates' in preproc_flags else 'false'}",
            f"particle.defragment={'true' if 'defragment' in preproc_flags else 'false'}",
            f"particle.voxelize.enabled={'true' if 'voxelize' in preproc_flags else 'false'}",
            f"particle.voxelize.voxel_size={voxel_size_val}",
            f"particle.preprocessor.name={preproc_backend}",
            f"particle.preprocessor.sem_types={sem_types_str}",
            f"checker={_CHECKER_CFG.get(checker_type, 'cpu_single')}",
            f"distance_threshold={dist_thresh}",
            # dummy I/O paths (not used in the app — we load the file ourselves)
            "io.input_path=__app_stub__",
            "io.output_path=__app_stub__",
        ]
        # n_jobs only valid for multi-threaded checkers (schema enforces this)
        if checker_type in _MULTI_CHECKERS:
            overrides.append(f"checker.n_jobs={n_jobs}")

        if verbose:
            lines.append("── overrides passed to load_cfg ─────────────────────")
            for _ov in overrides:
                lines.append(f"  {_ov}")
            lines.append("─────────────────────────────────────────────────────")

        cfg = load_cfg(overrides)

        if verbose:
            lines.append("── resolved cfg ──────────────────────────────")
            lines.append(f"  distance_threshold : {cfg.distance_threshold}")
            lines.append(f"  checker.name       : {cfg.checker.name}")
            if hasattr(cfg.checker, 'n_jobs'):
                lines.append(f"  checker.n_jobs     : {cfg.checker.n_jobs}")
            lines.append(f"  min_pc_size        : {cfg.particle.min_pc_size}")
            lines.append(f"  merge_duplicates   : {cfg.particle.merge_duplicates}")
            lines.append(f"  defragment         : {cfg.particle.defragment}")
            _pp = cfg.particle.preprocessor
            lines.append(f"  preproc.name       : {_pp.name}")
            lines.append(f"  preproc.sem_types  : {list(_pp.sem_types)}")
            lines.append("──────────────────────────────────────────────")

    except Exception:
        err = f"✘ Config error:\n{traceback.format_exc(limit=3)}"
        return dash.no_update, "✘", err

    # ── Load event + preprocessing (skipped on cache hit) ──────────────────
    if _cached is not None:
        _PARTICLE_CACHE.move_to_end(_preproc_key)
        particles_for_display, _cache_log = _cached
        lines.extend(_cache_log)
        lines.append("  ↩ load+preproc result from cache")
    else:
        _load_preproc_lines: list[str] = []
        try:
            from collections import Counter

            if format_val == "edepsim_h5":
                _jaxtpc = bool(jaxtpc_seg_path_val) and bool(jaxtpc_inst_path_val)
                if _jaxtpc:
                    # Both JAXTPC paths provided → visibility-filtered reader.
                    # Only segments visible in the JAXTPC readout are returned.
                    from pysupera.readers import JaxtpcHDF5Reader
                    with JaxtpcHDF5Reader(
                        edepsim_path=file_path,
                        seg_path=jaxtpc_seg_path_val,
                        inst_path=jaxtpc_inst_path_val,
                        particle_key=particle_key_val,
                        electron_energy_threshold=elec_thresh_val,
                        min_pc_size=min_pc_size,
                    ) as reader:
                        n_events = len(reader)
                        particles = reader[min(event_idx, n_events - 1)]
                    _load_preproc_lines.append(
                        f"✔ Loaded {file_path}  ({n_events} events)  [edepsim_h5 + jaxtpc visibility]"
                    )
                    _load_preproc_lines.append(f"  seg : {jaxtpc_seg_path_val}")
                    _load_preproc_lines.append(f"  inst: {jaxtpc_inst_path_val}")
                else:
                    from pysupera.readers import EDepSimHDF5Reader
                    with EDepSimHDF5Reader(
                        file_path,
                        particle_key=particle_key_val,
                        step_key=step_key_val,
                        electron_energy_threshold=elec_thresh_val,
                        min_pc_size=min_pc_size,
                    ) as reader:
                        n_events = len(reader)
                        particles = reader[min(event_idx, n_events - 1)]
                    _load_preproc_lines.append(
                        f"✔ Loaded {file_path}  ({n_events} events)  [edepsim_h5]"
                    )
            else:
                from pysupera import read_events
                with read_events(file_path) as store:
                    n_events = len(store)
                    particles = store[min(event_idx, n_events - 1)]
                _load_preproc_lines.append(
                    f"✔ Loaded {file_path}  ({n_events} events)  [native pysupera]"
                )

            _load_preproc_lines.append(
                f"  Event {event_idx}: {len(particles)} particles"
            )
            if verbose:
                _counts = Counter(str(p.sem_type) for p in particles)
                for sem, cnt in sorted(_counts.items()):
                    _load_preproc_lines.append(f"    {sem}: {cnt}")

        except Exception:
            err = f"✘ Load error:\n{traceback.format_exc(limit=3)}"
            return dash.no_update, "✘", err

        # ── Preprocessing ────────────────────────────────────────────────
        try:
            from pysupera.config import build_merge_processor, build_preprocessor, build_voxelizer

            if "merge_duplicates" in preproc_flags:
                _t = time.perf_counter()
                merger = build_merge_processor(cfg, verbose=verbose)
                if merger:
                    _buf = io.StringIO()
                    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
                        particles = merger.process(particles)
                    _cap = _buf.getvalue()
                    if _cap:
                        _load_preproc_lines.append(_cap.rstrip("\n"))
                _load_preproc_lines.append(
                    f"  merge_duplicates: {time.perf_counter()-_t:.3f}s"
                    f"  → {len(particles)} particles"
                )
                if verbose:
                    _counts = Counter(str(p.sem_type) for p in particles)
                    for sem, cnt in sorted(_counts.items()):
                        _load_preproc_lines.append(f"    {sem}: {cnt}")

            if "voxelize" in preproc_flags:
                _t = time.perf_counter()
                voxelizer = build_voxelizer(cfg)
                if voxelizer:
                    particles = voxelizer.process(particles)
                _load_preproc_lines.append(
                    f"  voxelize (voxel_size={voxel_size_val}): "
                    f"{time.perf_counter()-_t:.3f}s  \u2192 {len(particles)} particles"
                )

            if "defragment" in preproc_flags:
                _t = time.perf_counter()
                preprocessor = build_preprocessor(cfg)
                if preprocessor:
                    particles = preprocessor.process(particles)
                sem_label = (f" sem_types={sem_types_input}"
                             if sem_types_input else " (all types)")
                _load_preproc_lines.append(
                    f"  defragment ({preproc_backend}{sem_label}): "
                    f"{time.perf_counter()-_t:.3f}s  → {len(particles)} particles"
                )
                if verbose:
                    _counts = Counter(str(p.sem_type) for p in particles)
                    for sem, cnt in sorted(_counts.items()):
                        _load_preproc_lines.append(f"    {sem}: {cnt}")

        except Exception:
            err = f"✘ Preprocessing error:\n{traceback.format_exc(limit=3)}"
            return dash.no_update, "✘", err

        # Store preprocessed particles in cache (evict oldest if full)
        particles_for_display = list(particles)
        _PARTICLE_CACHE[_preproc_key] = (particles_for_display, list(_load_preproc_lines))
        if len(_PARTICLE_CACHE) > _MAX_CACHE_ENTRIES:
            _PARTICLE_CACHE.popitem(last=False)
        lines.extend(_load_preproc_lines)

    # particles alias for the partition step (works on both cache-hit and miss)
    particles = particles_for_display

    # ── Partition ────────────────────────────────────────────────────────
    try:
        from pysupera.partitioner import ParticlePartitioner
        from pysupera.config import build_conditions

        # build_conditions(cfg) uses cfg.conditions flags; filter by the UI
        # selection so unchecked boxes actually disable the condition.
        _all_conditions = build_conditions(cfg)
        _enabled_keys = set(condition_keys)
        _key_map = {
            "PhotonDecay":       "photon_decay",
            "TouchingEMShower":  "touching_em_shower",
            "CombineLEScatters": "combine_le_scatters",
            "AbsorbLEScatter":   "absorb_le_scatter",
        }
        conditions = [c for c in _all_conditions
                      if _key_map.get(type(c).__name__, "") in _enabled_keys]

        # Use the values that load_cfg actually resolved, not raw UI strings,
        # so the partitioner is in sync with the rest of the pipeline.
        _backend  = cfg.checker.name
        _njobs    = getattr(cfg.checker, "n_jobs", 1)
        _dist     = float(cfg.distance_threshold)

        if verbose:
            lines.append("── partitioner params ──────────────────────────────")
            lines.append(f"  distance_threshold : {_dist}")
            lines.append(f"  backend            : {_backend}")
            lines.append(f"  n_jobs             : {_njobs}")
            lines.append(f"  conditions ({len(conditions)}): "
                         + ", ".join(type(c).__name__ for c in conditions))
            lines.append("────────────────────────────────────────────────────")

        if not conditions:
            partitions = [[p] for p in particles]
            lines.append("  No conditions selected — each particle is its own partition")
        else:
            _t = time.perf_counter()
            alg = ParticlePartitioner(
                particles,
                distance_threshold=_dist,
                backend=_backend,
                n_jobs=_njobs,
                enable_diagnostics=cfg.enable_diagnostics,
            )
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
                partitions = alg.partition_combined(conditions, verbose=True)
            _cap = _buf.getvalue()
            if _cap:
                lines.append(_cap.rstrip("\n"))
            _dur = time.perf_counter() - _t
            lines.append(f"  partition_combined: {_dur:.3f}s")
            lines.append(f"  {len(particles)} particles → {len(partitions)} partitions")
            sizes = sorted([len(pt) for pt in partitions], reverse=True)
            if verbose:
                lines.append(f"  all partition sizes: {sizes}")
            else:
                lines.append(f"  sizes (top 5): {sizes[:5]}")

    except Exception:
        err = f"✘ Partition error:\n{traceback.format_exc(limit=3)}"
        return dash.no_update, "✘", err

    total_time = time.perf_counter() - t_total
    lines.append(f"\ntotal: {total_time:.3f}s")

    # ── Apply display-only point-cloud size filter ────────────────────────
    if min_display_pc > 0:
        particles_for_display = [
            p for p in particles_for_display if len(p.point_cloud) >= min_display_pc
        ]
        partitions = [
            [p for p in part if len(p.point_cloud) >= min_display_pc]
            for part in partitions
        ]
        partitions = [part for part in partitions if part]
        lines.append(
            f"  display filter: pc ≥ {min_display_pc} pts →"
            f" {len(particles_for_display)} particles, {len(partitions)} partitions shown"
        )

    # ── Build figure ─────────────────────────────────────────────────────
    try:
        fig = _build_figure(particles_for_display, partitions,
                            show_legend=show_legend, draw_mode=draw_mode)
    except Exception:
        err = f"✘ Figure error:\n{traceback.format_exc(limit=3)}"
        return dash.no_update, "✘", err

    brief = f"✔ {total_time:.2f}s │ {len(partitions)} partitions"
    return fig, brief, "\n".join(lines)


# ── Helper available to all clientside callbacks ──────────────────────────────
# dcc.Graph(id="main-graph") creates an outer wrapper div with that id.
# Plotly attaches _fullLayout / data / event-emitter to the INNER
# div.js-plotly-plot child.  We must resolve that inner element every time.

# ── Legend toggle (clientside, instant) ───────────────────────────────────────
app.clientside_callback(
    """
    function(viewOpts) {
        var opts = viewOpts || [];
        window._cameraSyncEnabled = opts.indexOf('sync_cameras') >= 0;
        var showLegend = opts.indexOf('show_legend') >= 0;
        window._legendVisible = showLegend;
        setTimeout(function() {
            var wrap = document.getElementById('main-graph');
            var gd   = (wrap && wrap.querySelector('.js-plotly-plot')) || wrap;
            if (gd && gd._fullLayout) {
                Plotly.relayout(gd, {
                    'legend.visible':  showLegend,
                    'legend2.visible': showLegend
                });
            }
        }, 50);
        return '';
    }
    """,
    Output("sync-dummy", "title"),
    Input("view-options", "value"),
)

# ── Particle filter by sem_type (clientside, instant) ────────────────────────
app.clientside_callback(
    """
    function(visibleTypes, _figure) {
        var visible = visibleTypes || [];
        setTimeout(function() {
            var wrap = document.getElementById('main-graph');
            var gd   = (wrap && wrap.querySelector('.js-plotly-plot')) || wrap;
            if (!gd || !gd.data || !gd.data.length) return;
            var updates = gd.data.map(function(trace) {
                var st = trace.meta && trace.meta.sem_type ? trace.meta.sem_type : null;
                return (st === null || visible.indexOf(st) >= 0) ? true : false;
            });
            Plotly.restyle(gd, {visible: updates});
        }, 80);
        return '';
    }
    """,
    Output("mask-dummy", "children"),
    Input("mask-sem-types", "value"),
    Input("main-graph", "figure"),
)

# ── Camera sync + ResizeObserver (clientside) ──────────────────────────────
app.clientside_callback(
    """
    function(figure) {
        setTimeout(function() {
            var wrap    = document.getElementById('main-graph');
            var gd      = (wrap && wrap.querySelector('.js-plotly-plot')) || wrap;
            var topSec  = document.getElementById('top-section');

            // ── ResizeObserver: keep Plotly filling top-section ───────
            if (topSec && !topSec._resizeObserverAttached) {
                topSec._resizeObserverAttached = true;
                var ro = new ResizeObserver(function() {
                    var w = document.getElementById('main-graph');
                    var g = (w && w.querySelector('.js-plotly-plot')) || w;
                    if (g) Plotly.Plots.resize(g);
                });
                ro.observe(topSec);
            }

            // ── Re-apply legend visibility after re-render ────────────
            if (gd && gd._fullLayout && typeof window._legendVisible !== 'undefined') {
                Plotly.relayout(gd, {
                    'legend.visible':  window._legendVisible,
                    'legend2.visible': window._legendVisible,
                });
            }

            // ── Camera sync ───────────────────────────────────────
            if (gd && figure && figure.data && figure.data.length > 0
                    && !gd._cameraSyncAttached) {
                gd._cameraSyncAttached = true;
                var syncing = false;
                gd.on('plotly_relayout', function(eventdata) {
                    if (syncing) return;
                    if (!window._cameraSyncEnabled) return;
                    syncing = true;
                    var update = {};
                    var changed = false;
                    for (var key in eventdata) {
                        if (key.startsWith('scene.camera')) {
                            update[key.replace('scene.', 'scene2.')] = eventdata[key];
                            changed = true;
                        } else if (key.startsWith('scene2.camera')) {
                            update[key.replace('scene2.', 'scene.')] = eventdata[key];
                            changed = true;
                        }
                    }
                    if (changed) {
                        Plotly.relayout(gd, update).then(function() { syncing = false; });
                    } else {
                        syncing = false;
                    }
                });
            }

            // ── Log-panel drag handle ────────────────────────────────
            var handle = document.getElementById('log-drag-handle');
            if (handle && !handle._dragAttached) {
                handle._dragAttached = true;
                handle.addEventListener('mousedown', function(e) {
                    e.preventDefault();
                    var logBody  = document.getElementById('log-body');
                    var startY   = e.clientY;
                    var startH   = logBody.offsetHeight;
                    function onMove(e) {
                        // dragging UP = bigger log, dragging DOWN = smaller log
                        var delta  = startY - e.clientY;
                        var newH   = Math.max(0, Math.min(window.innerHeight * 0.85,
                                                          startH + delta));
                        logBody.style.height = newH + 'px';
                    }
                    function onUp() {
                        document.removeEventListener('mousemove', onMove);
                        document.removeEventListener('mouseup', onUp);
                    }
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                });
            }
        }, 200);
        return '';
    }
    """,
    Output("sync-dummy", "children"),
    Input("main-graph", "figure"),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="pysupera Dash visualizer")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug / hot-reload mode")
    args = parser.parse_args()

    print(f"pysupera visualizer  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
