#!/usr/bin/env python3
"""
Interactive 3-D point-cloud viewer for pysupera HDF5 output files.

Usage
-----
    python -m pysupera.vis path/to/output.h5 [--port 8765]
    # or, from the pysupera package:
    python -c "from pysupera.vis import serve; serve('output.h5')"

Renders ``non_le_cloud`` and ``le_scatter_cloud`` datasets using Three.js /
WebGL.  Handles 100 k – 1 M points at interactive frame-rates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

try:
    import hdf5plugin  # noqa: F401 — registers LZ4/Blosc decoders
except ImportError:
    pass


# ────────────────────────────────────────────────────────────────────────────
# Embedded HTML / JS / CSS  (single-file, served from memory)
# ────────────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pysupera Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;overflow:hidden;
     font:13px/1.5 'Segoe UI',system-ui,sans-serif;
     background:#111;color:#ddd}
#sidebar{width:280px;min-width:280px;background:#1a1a2a;overflow-y:auto;
         border-right:1px solid #2e2e4a;display:flex;flex-direction:column;
         padding:0 0 12px 0}
#canvas-wrap{flex:1;position:relative;background:#0d0d1a}
canvas{display:block}
h2{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
   color:#6e6e9e;padding:10px 12px 4px;border-top:1px solid #2e2e4a;
   margin-top:6px}
h2:first-child{border-top:none;margin-top:0}
.filename{padding:8px 12px;font-size:12px;color:#8888cc;
          border-bottom:1px solid #2e2e4a;word-break:break-all}
.row{display:flex;align-items:center;gap:6px;padding:3px 12px}
.row label{flex:1;color:#aaa;font-size:12px}
.row input[type=number]{width:72px;background:#0d0d1a;border:1px solid #3a3a5a;
  color:#ddd;padding:2px 5px;border-radius:3px;font-size:12px}
.row input[type=range]{flex:1;accent-color:#7777ee}
.row select{flex:1;background:#0d0d1a;border:1px solid #3a3a5a;
  color:#ddd;padding:2px 4px;border-radius:3px;font-size:12px}
.row input[type=checkbox]{accent-color:#7777ee}
.row .val{width:36px;text-align:right;font-size:12px;color:#aaa}
.row-ev{display:flex;align-items:center;gap:4px;padding:4px 12px}
.row-ev button{background:#2e2e50;border:1px solid #4a4a72;color:#ccc;
  padding:2px 10px;border-radius:3px;cursor:pointer;font-size:14px;
  line-height:1.4}
.row-ev button:hover{background:#3e3e70}
.row-ev input[type=number]{width:60px;background:#0d0d1a;border:1px solid #3a3a5a;
  color:#ddd;padding:2px 5px;border-radius:3px;font-size:13px;text-align:center}
.row-ev span{color:#888;font-size:12px}
.layer-row{display:flex;align-items:center;gap:6px;padding:3px 10px}
.layer-row input{accent-color:#7777ee}
.layer-row label{font-size:12px;color:#bbb;flex:1;cursor:pointer}
.layer-row .cnt{font-size:11px;color:#666;white-space:nowrap}
.cbar{height:16px;margin:2px 12px 4px;border-radius:3px;border:1px solid #3a3a5a}
.range-row{display:flex;gap:4px;padding:0 12px 2px}
.range-row input{flex:1;background:#0d0d1a;border:1px solid #3a3a5a;
  color:#ddd;padding:2px 5px;border-radius:3px;font-size:11px}
.range-row label{font-size:11px;color:#888;align-self:center}
.stat-row{display:flex;padding:1px 12px;font-size:11px;color:#888}
.stat-row span:first-child{flex:1}
.stat-row span:last-child{color:#aaa}
#status{padding:6px 12px;font-size:11px;color:#7788ff;margin-top:auto}
.btn-row{display:flex;gap:6px;padding:3px 12px}
.btn{background:#2e2e50;border:1px solid #4a4a72;color:#ccc;
  padding:3px 10px;border-radius:3px;cursor:pointer;font-size:11px}
.btn:hover{background:#3e3e70}
#overlay{position:absolute;top:8px;right:8px;
  background:#1a1a2acc;border:1px solid #3a3a5a;
  border-radius:4px;padding:4px 10px;font-size:10px;color:#666}
</style>

<script type="importmap">
{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.163.0/examples/jsm/"
}}
</script>
</head>
<body>

<!-- ── Sidebar ──────────────────────────────────────────────────────── -->
<div id="sidebar">
  <div class="filename" id="filename">loading…</div>

  <h2>Event</h2>
  <div class="row-ev">
    <button id="btn-prev">&#8249;</button>
    <input type="number" id="ev-input" value="0" min="0">
    <span id="ev-total">/ —</span>
    <button id="btn-next">&#8250;</button>
  </div>

  <h2>Layers</h2>
  <div class="layer-row">
    <input type="checkbox" id="chk-nle" checked>
    <label for="chk-nle">Non-LE cloud</label>
    <span class="cnt" id="cnt-nle">—</span>
  </div>
  <div class="layer-row">
    <input type="checkbox" id="chk-le" checked>
    <label for="chk-le">LE scatter cloud</label>
    <span class="cnt" id="cnt-le">—</span>
  </div>

  <h2>Appearance</h2>
  <div class="row">
    <label>Point size</label>
    <input type="range" id="sl-size" min="0.5" max="10" step="0.5" value="2">
    <span class="val" id="lbl-size">2</span>
  </div>
  <div class="row">
    <label>Opacity</label>
    <input type="range" id="sl-opacity" min="0.05" max="1" step="0.05" value="1">
    <span class="val" id="lbl-opacity">1.0</span>
  </div>

  <h2>Color</h2>
  <div class="row">
    <label>Color by</label>
    <select id="sel-col">
      <option value="4" selected>Energy</option>
      <option value="3">Time</option>
      <option value="5">Interaction ID</option>
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2">Z</option>
    </select>
  </div>
  <div class="row">
    <label>Colormap</label>
    <select id="sel-cmap">
      <option value="0" selected>Viridis</option>
      <option value="1">Plasma</option>
      <option value="2">Turbo</option>
      <option value="3">Hot</option>
      <option value="4">Cool</option>
      <option value="5">Grayscale</option>
    </select>
  </div>
  <div class="row">
    <label>Auto range</label>
    <input type="checkbox" id="chk-auto" checked>
  </div>
  <div id="cbar" class="cbar"></div>
  <div class="range-row">
    <label>Min</label>
    <input type="number" id="rng-min" step="any" value="0">
    <label>Max</label>
    <input type="number" id="rng-max" step="any" value="1">
  </div>

  <h2>Stats</h2>
  <div class="stat-row"><span>Non-LE pts</span><span id="cnt2-nle">—</span></div>
  <div class="stat-row"><span>LE pts</span><span id="cnt2-le">—</span></div>
  <div class="stat-row"><span>Total pts</span><span id="cnt2-tot">—</span></div>

  <h2>Scene</h2>
  <div class="btn-row">
    <button class="btn" id="btn-center">Center camera</button>
    <button class="btn" id="btn-axes">Toggle axes</button>
  </div>

  <div id="status">Initializing…</div>
</div>

<!-- ── Canvas ────────────────────────────────────────────────────────── -->
<div id="canvas-wrap">
  <div id="overlay">WebGL</div>
</div>

<!-- ── Three.js app ──────────────────────────────────────────────────── -->
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Colormaps (CSS gradient strings for the color bar) ───────────────
const CMAP_CSS = [
  'linear-gradient(to right,#440154,#31688e,#35b779,#fde725)',          // viridis
  'linear-gradient(to right,#0d0887,#7e03a8,#cc4778,#f89540,#f0f921)', // plasma
  'linear-gradient(to right,#23171b,#4076f4,#29efb1,#fdb830,#900c00)', // turbo
  'linear-gradient(to right,#000,#f00,#ff0,#fff)',                      // hot
  'linear-gradient(to right,#0ff,#f0f)',                                // cool
  'linear-gradient(to right,#000,#fff)',                                // gray
];

// ── Vertex shader ────────────────────────────────────────────────────
const VS = `
attribute vec3 extra;
uniform int  uColorCol;
uniform float uMin, uMax, uSize;
varying float vN;
void main(){
  float v;
  if(uColorCol==0) v=position.x;
  else if(uColorCol==1) v=position.y;
  else if(uColorCol==2) v=position.z;
  else if(uColorCol==3) v=extra.x;
  else if(uColorCol==4) v=extra.y;
  else v=extra.z;
  vN=clamp((v-uMin)/max(uMax-uMin,1e-10),0.0,1.0);
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
  gl_PointSize=uSize;
}`;

// ── Fragment shader with 6 colormaps ────────────────────────────────
const FS = `
precision mediump float;
uniform int   uCmap;
uniform float uOpacity;
varying float vN;

vec3 viridis(float t){
  vec3 c0=vec3(.277,.0054,.334),c1=vec3(.105,1.404,1.384),
       c2=vec3(-.331,.215,.095),c3=vec3(-4.63,-5.80,-19.33),
       c4=vec3(6.23,14.18,56.69),c5=vec3(4.78,-13.74,-65.35),
       c6=vec3(-5.44,4.65,26.31);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))),0.,1.);
}
vec3 plasma(float t){
  vec3 c0=vec3(.059,.023,.543),c1=vec3(2.177,.238,.754),
       c2=vec3(-2.689,-7.456,3.110),c3=vec3(6.130,42.35,-28.52),
       c4=vec3(-11.11,-82.67,60.14),c5=vec3(10.02,71.41,-54.07),
       c6=vec3(-3.66,-22.93,18.19);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))),0.,1.);
}
vec3 turbo(float t){
  vec3 c0=vec3(.114,.063,.225),c1=vec3(6.716,3.182,7.572),
       c2=vec3(-66.09,-4.928,-10.09),c3=vec3(228.77,25.05,-91.54),
       c4=vec3(-334.84,-69.32,288.59),c5=vec3(218.76,67.52,-305.20),
       c6=vec3(-52.89,-21.55,110.52);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))),0.,1.);
}
vec3 hot(float t){return clamp(vec3(t*3.,t*3.-1.,t*3.-2.),0.,1.);}
vec3 cool(float t){return vec3(t,1.-t,1.);}
void main(){
  vec2 c=gl_PointCoord-.5; if(dot(c,c)>.25) discard;
  float t=vN;
  vec3 col;
  if(uCmap==0) col=viridis(t);
  else if(uCmap==1) col=plasma(t);
  else if(uCmap==2) col=turbo(t);
  else if(uCmap==3) col=hot(t);
  else if(uCmap==4) col=cool(t);
  else col=vec3(t);
  gl_FragColor=vec4(col,uOpacity);
}`;

// ── App state ────────────────────────────────────────────────────────
const S = {
  nEvents: 0, evIdx: 0,
  showNLE: true, showLE: true,
  col: 4, cmap: 0, autoRange: true,
  size: 2.0, opacity: 1.0,
  ranges: { nle: null, le: null },
};

// ── Three.js init ────────────────────────────────────────────────────
const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
wrap.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1e7);
camera.position.set(0, 0, 800);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;

const axesHelper = new THREE.AxesHelper(200);
scene.add(axesHelper);

// ── Point cloud objects ──────────────────────────────────────────────
let objNLE = null, objLE = null;

function makeUniforms() {
  return {
    uColorCol: {value: S.col},
    uMin:      {value: 0.0},
    uMax:      {value: 1.0},
    uSize:     {value: S.size},
    uCmap:     {value: S.cmap},
    uOpacity:  {value: S.opacity},
  };
}

function buildPoints(f32) {
  const N = f32.length / 6;
  if (N === 0) return null;
  const pos  = new Float32Array(N * 3);
  const ext  = new Float32Array(N * 3);   // t, energy, iid
  for (let i = 0; i < N; i++) {
    const b = i * 6;
    pos[i*3]   = f32[b];   pos[i*3+1] = f32[b+1]; pos[i*3+2] = f32[b+2];
    ext[i*3]   = f32[b+3]; ext[i*3+1] = f32[b+4]; ext[i*3+2] = f32[b+5];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('extra',    new THREE.BufferAttribute(ext, 3));
  const mat = new THREE.ShaderMaterial({
    uniforms: makeUniforms(), vertexShader: VS, fragmentShader: FS,
    transparent: true, depthWrite: false,
  });
  const pts = new THREE.Points(geo, mat);
  return {pts, n: N};
}

function stats6(f32) {
  const N = f32.length / 6;
  if (N === 0) return null;
  const mn = new Float32Array(6).fill(Infinity);
  const mx = new Float32Array(6).fill(-Infinity);
  for (let i = 0; i < N; i++) {
    const b = i * 6;
    for (let c = 0; c < 6; c++) {
      if (f32[b+c] < mn[c]) mn[c] = f32[b+c];
      if (f32[b+c] > mx[c]) mx[c] = f32[b+c];
    }
  }
  return {mn, mx};
}

function clearObjects() {
  for (const o of [objNLE, objLE]) {
    if (o) { scene.remove(o.pts); o.pts.geometry.dispose(); o.pts.material.dispose(); }
  }
  objNLE = objLE = null;
}

// ── Load event data ──────────────────────────────────────────────────
async function loadEvent(idx) {
  setStatus('Loading…');
  clearObjects();

  let nleF32 = new Float32Array(0), leF32 = new Float32Array(0);
  try {
    const [ab1, ab2] = await Promise.all([
      fetch(`/api/event/${idx}/non_le`).then(r => r.arrayBuffer()),
      fetch(`/api/event/${idx}/le`).then(r => r.arrayBuffer()),
    ]);
    nleF32 = new Float32Array(ab1);
    leF32  = new Float32Array(ab2);
  } catch(e) { setStatus('Fetch error: ' + e); return; }

  S.ranges.nle = stats6(nleF32);
  S.ranges.le  = stats6(leF32);

  objNLE = buildPoints(nleF32);
  objLE  = buildPoints(leF32);
  if (objNLE) { scene.add(objNLE.pts); objNLE.pts.visible = S.showNLE; }
  if (objLE)  { scene.add(objLE.pts);  objLE.pts.visible  = S.showLE;  }

  applyColorRange();
  updateStats();
  setStatus('Ready');
}

// ── Uniform / color sync ─────────────────────────────────────────────
function applyColorRange() {
  if (S.autoRange) {
    let mn = Infinity, mx = -Infinity;
    for (const r of [S.ranges.nle, S.ranges.le]) {
      if (!r) continue;
      if (r.mn[S.col] < mn) mn = r.mn[S.col];
      if (r.mx[S.col] > mx) mx = r.mx[S.col];
    }
    if (isFinite(mn)) {
      document.getElementById('rng-min').value = +mn.toPrecision(5);
      document.getElementById('rng-max').value = +mx.toPrecision(5);
    }
  }
  applyUniforms();
}

function applyUniforms() {
  const mn = parseFloat(document.getElementById('rng-min').value) || 0;
  const mx = parseFloat(document.getElementById('rng-max').value) || 1;
  for (const o of [objNLE, objLE]) {
    if (!o) continue;
    const u = o.pts.material.uniforms;
    u.uColorCol.value = S.col;
    u.uMin.value      = mn;
    u.uMax.value      = mx;
    u.uSize.value     = S.size;
    u.uCmap.value     = S.cmap;
    u.uOpacity.value  = S.opacity;
  }
  updateColorBar(mn, mx);
}

function updateColorBar(mn, mx) {
  const bar = document.getElementById('cbar');
  bar.style.background = CMAP_CSS[S.cmap];
  bar.title = `${mn.toPrecision(4)} → ${mx.toPrecision(4)}`;
}

function updateStats() {
  const n1 = objNLE ? objNLE.n : 0;
  const n2 = objLE  ? objLE.n  : 0;
  const fmt = n => n.toLocaleString();
  document.getElementById('cnt-nle').textContent  = fmt(n1) + ' pts';
  document.getElementById('cnt-le').textContent   = fmt(n2) + ' pts';
  document.getElementById('cnt2-nle').textContent = fmt(n1);
  document.getElementById('cnt2-le').textContent  = fmt(n2);
  document.getElementById('cnt2-tot').textContent = fmt(n1 + n2);
}

function setStatus(msg) { document.getElementById('status').textContent = msg; }

// ── Center camera on loaded geometry ────────────────────────────────
function centerCamera() {
  const box = new THREE.Box3();
  for (const o of [objNLE, objLE]) {
    if (o && o.pts.visible) box.expandByObject(o.pts);
  }
  if (box.isEmpty()) return;
  const center = new THREE.Vector3();
  const size   = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const dist = Math.max(size.x, size.y, size.z) * 1.5;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0, 0.4 * dist, dist));
  camera.near = dist * 1e-4;
  camera.far  = dist * 10;
  camera.updateProjectionMatrix();
  controls.update();
}

// ── UI wiring ────────────────────────────────────────────────────────
function goTo(idx) {
  idx = Math.max(0, Math.min(idx, S.nEvents - 1));
  S.evIdx = idx;
  document.getElementById('ev-input').value = idx;
  loadEvent(idx);
}

document.getElementById('btn-prev').addEventListener('click', () => goTo(S.evIdx - 1));
document.getElementById('btn-next').addEventListener('click', () => goTo(S.evIdx + 1));
document.getElementById('ev-input').addEventListener('change', e => goTo(parseInt(e.target.value)||0));

document.getElementById('chk-nle').addEventListener('change', e => {
  S.showNLE = e.target.checked;
  if (objNLE) objNLE.pts.visible = S.showNLE;
});
document.getElementById('chk-le').addEventListener('change', e => {
  S.showLE = e.target.checked;
  if (objLE) objLE.pts.visible = S.showLE;
});

document.getElementById('sl-size').addEventListener('input', e => {
  S.size = parseFloat(e.target.value);
  document.getElementById('lbl-size').textContent = S.size;
  applyUniforms();
});
document.getElementById('sl-opacity').addEventListener('input', e => {
  S.opacity = parseFloat(e.target.value);
  document.getElementById('lbl-opacity').textContent = S.opacity.toFixed(2);
  applyUniforms();
});

document.getElementById('sel-col').addEventListener('change', e => {
  S.col = parseInt(e.target.value);
  applyColorRange();
});
document.getElementById('sel-cmap').addEventListener('change', e => {
  S.cmap = parseInt(e.target.value);
  applyUniforms();
});
document.getElementById('chk-auto').addEventListener('change', e => {
  S.autoRange = e.target.checked;
  document.getElementById('rng-min').disabled = S.autoRange;
  document.getElementById('rng-max').disabled = S.autoRange;
  if (S.autoRange) applyColorRange(); else applyUniforms();
});
document.getElementById('rng-min').addEventListener('change', () => { if(!S.autoRange) applyUniforms(); });
document.getElementById('rng-max').addEventListener('change', () => { if(!S.autoRange) applyUniforms(); });

document.getElementById('btn-center').addEventListener('click', centerCamera);
document.getElementById('btn-axes').addEventListener('click', () => {
  axesHelper.visible = !axesHelper.visible;
});

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  goTo(S.evIdx - 1);
  if (e.key === 'ArrowRight') goTo(S.evIdx + 1);
  if (e.key === 'c') centerCamera();
  if (e.key === 'a') axesHelper.visible = !axesHelper.visible;
});

// ── Resize ───────────────────────────────────────────────────────────
function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();

// ── FPS overlay ──────────────────────────────────────────────────────
let fpsT = performance.now(), fpsN = 0;
function tickFPS() {
  fpsN++;
  const now = performance.now();
  if (now - fpsT > 1000) {
    document.getElementById('overlay').textContent = `${fpsN} fps`;
    fpsN = 0; fpsT = now;
  }
}

// ── Render loop ──────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  tickFPS();
}

// ── Init ─────────────────────────────────────────────────────────────
(async () => {
  let info;
  try { info = await fetch('/api/info').then(r => r.json()); }
  catch(e) { setStatus('Cannot reach server.'); return; }

  S.nEvents = info.n_events;
  document.getElementById('filename').textContent = info.filename;
  document.getElementById('ev-total').textContent = `/ ${info.n_events}`;
  document.getElementById('ev-input').max = info.n_events - 1;

  if (!info.has_non_le && !info.has_le) {
    setStatus('No cloud datasets found in this file.');
    animate(); return;
  }

  animate();
  await loadEvent(0);
  centerCamera();
})();
</script>
</body>
</html>
"""

# ────────────────────────────────────────────────────────────────────────────
# HTTP request handler
# ────────────────────────────────────────────────────────────────────────────
_h5file: "h5py.File | None" = None  # set in serve()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, _fmt, *_args):
        pass  # suppress per-request console noise

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send('text/html; charset=utf-8', _HTML.encode())
        elif path == '/api/info':
            self._handle_info()
        elif path.startswith('/api/event/'):
            self._handle_event(path)
        else:
            self.send_error(404)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _send(self, ctype: str, data: bytes, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _handle_info(self):
        f = _h5file
        info = {
            'filename': os.path.basename(f.filename),
            'n_events': int(f['n_events'][()]) if 'n_events' in f else 0,
            'has_non_le': 'non_le_cloud/flat' in f,
            'has_le':     'le_scatter_cloud/flat' in f,
        }
        self._send('application/json', json.dumps(info).encode())

    def _handle_event(self, path: str):
        # Expected: /api/event/{idx}/{group}  where group = non_le | le
        parts = path.rstrip('/').split('/')
        if len(parts) != 5:
            self.send_error(400, 'Bad path'); return
        try:
            ev_idx = int(parts[3])
        except ValueError:
            self.send_error(400, 'Non-integer event index'); return

        grp_key = parts[4]
        if grp_key == 'non_le':
            grp = 'non_le_cloud'
        elif grp_key == 'le':
            grp = 'le_scatter_cloud'
        else:
            self.send_error(400, 'Unknown group'); return

        f = _h5file
        off_key = f'{grp}/offsets'
        flat_key = f'{grp}/flat'
        if off_key not in f or flat_key not in f:
            self._send('application/octet-stream', b''); return

        offsets = f[off_key][:]
        n_ev = len(offsets) - 1
        if ev_idx < 0 or ev_idx >= n_ev:
            self.send_error(400, f'Event {ev_idx} out of range [0,{n_ev})'); return

        start = int(offsets[ev_idx])
        end   = int(offsets[ev_idx + 1])
        if start >= end:
            self._send('application/octet-stream', b''); return

        arr = np.ascontiguousarray(f[flat_key][start:end], dtype=np.float32)
        self._send('application/octet-stream', arr.tobytes())


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def serve(h5_path: str, port: int = 8765, open_browser: bool = True) -> None:
    """
    Start the viewer for *h5_path* on ``http://localhost:{port}/``.

    Blocks until Ctrl-C is pressed.

    Parameters
    ----------
    h5_path : str
        Path to a pysupera HDF5 output file.
    port : int, optional
        Local TCP port.  Default 8765.
    open_browser : bool, optional
        Whether to open the system browser automatically.  Default True.
    """
    global _h5file

    if not os.path.exists(h5_path):
        sys.exit(f'File not found: {h5_path!r}')

    try:
        import hdf5plugin  # noqa
    except ImportError:
        pass
    _h5file = h5py.File(h5_path, 'r')

    try:
        server = HTTPServer(('localhost', port), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = f'http://localhost:{port}/'
        print(f'[pysupera.vis]  Serving {h5_path!r}')
        print(f'[pysupera.vis]  Open:   {url}')
        print(f'[pysupera.vis]  Stop:   Ctrl-C')
        print()
        print('  Keyboard shortcuts in the viewer:')
        print('    ← / →     previous / next event')
        print('    c         center camera on loaded points')
        print('    a         toggle axes helper')

        if open_browser:
            time.sleep(0.3)
            webbrowser.open(url)

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[pysupera.vis]  Stopping.')
    finally:
        server.shutdown()
        _h5file.close()


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='pysupera interactive 3-D point-cloud viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('h5_file', help='Path to pysupera HDF5 output file')
    parser.add_argument('--port', type=int, default=8765,
                        help='Local HTTP port (default: 8765)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open a browser window automatically')
    args = parser.parse_args()
    serve(args.h5_file, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
