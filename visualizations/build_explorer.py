import os, sys, json, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SD_PAIRS, PROFILES, WEIGHTED_ASTAR_W
from algorithms import ALL_ALGORITHMS
from algorithms.base import composite_cost, haversine

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LANDMARKS_PATH = os.path.join(DATA_DIR, "landmarks.pkl")

METRICS = ["road_length", "traffic_jam", "road_condition", "women_safety",
           "police_availability", "snatching_risk", "flood_risk",
           "lighting", "footpath", "noise"]

# Skip IDS in visualization — its exploration_order is 100k+ nodes, too heavy
VIS_ALGORITHMS = ["BFS", "DFS", "UCS", "Greedy", "A*", "Weighted A*"]


def load_graph(name):
    path = os.path.join(DATA_DIR, f"dhaka_{name}.gpickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def nearest_node(G, lat, lon):
    best, best_d = None, float("inf")
    for n in G.nodes:
        d = haversine(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"])
        if d < best_d:
            best, best_d = n, d
    return best


def export_graph(G):
    """Export graph to compact JSON for the browser."""
    nodes = {}
    for n in G.nodes:
        nodes[str(n)] = [round(G.nodes[n]["y"], 6), round(G.nodes[n]["x"], 6)]

    edges = []
    adjacency = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        su, sv = str(u), str(v)
        metrics = {m: round(data.get(m, 0), 4) for m in METRICS}
        hw = data.get("highway", "residential")
        if isinstance(hw, list):
            hw = hw[0]
        edges.append({"u": su, "v": sv, "hw": str(hw), **metrics})
        adjacency.setdefault(su, [])
        if sv not in adjacency[su]:
            adjacency[su].append(sv)

    return {"nodes": nodes, "edges": edges, "adjacency": adjacency}


def run_all_traces(G, mode_name, landmark_data_raw=None):
    """Run all visualization algorithms on all (pair × profile) combos.
    Returns dict {trace_key: {exploration, path, cost, ...}}."""
    traces = {}
    total = len(SD_PAIRS) * len(PROFILES) * len(VIS_ALGORITHMS)
    done = 0

    for pair in SD_PAIRS:
        src = nearest_node(G, pair["source"][0], pair["source"][1])
        dst = nearest_node(G, pair["destination"][0], pair["destination"][1])
        pair_key = pair["name"]

        for prof_name, profile in PROFILES.items():
            # Build landmark data for this profile (only if available)
            ld = None
            if landmark_data_raw is not None:
                ld = {
                    "profile_name": prof_name,
                    "per_profile": landmark_data_raw["per_profile"],
                }

            for algo_name in VIS_ALGORITHMS:
                done += 1
                algo_fn = ALL_ALGORITHMS[algo_name]

                kwargs = {"w": WEIGHTED_ASTAR_W}
                # Informed algorithms: pick heuristic based on availability
                if algo_name in ("Greedy", "A*", "Weighted A*"):
                    if ld is not None:
                        kwargs["heuristic_version"] = "h_ALT"
                        kwargs["landmark_data"] = ld
                    else:
                        kwargs["heuristic_version"] = "h_B"

                print(f"  [{mode_name} {done}/{total}] {algo_name} / {prof_name} / {pair_key}")
                r = algo_fn(G, src, dst, profile, **kwargs)

                trace_key = f"{algo_name}|{prof_name}|{pair_key}"
                traces[trace_key] = {
                    "exploration": [str(n) for n in r.exploration_order],
                    "path": [str(n) for n in r.path],
                    "cost": round(r.path_cost, 4),
                    "expanded": r.nodes_expanded,
                    "frontier": r.max_frontier_size,
                    "time_ms": round(r.execution_time * 1000, 2),
                    "found": r.found,
                    "src": str(src),
                    "dst": str(dst),
                }

    return traces


def build_html(drive_graph, walk_graph, drive_traces, walk_traces):
    """Build the self-contained HTML file."""
    data_js = (
        f"const DRIVE_GRAPH = {json.dumps(drive_graph)};\n"
        f"const WALK_GRAPH = {json.dumps(walk_graph)};\n"
        f"const DRIVE_TRACES = {json.dumps(drive_traces)};\n"
        f"const WALK_TRACES = {json.dumps(walk_traces)};\n"
        f"const PROFILES = {json.dumps(PROFILES)};\n"
        f"const METRICS = {json.dumps(METRICS)};\n"
        f"const SD_PAIRS = {json.dumps([p['name'] for p in SD_PAIRS])};\n"
        f"const ALGO_NAMES = {json.dumps(VIS_ALGORITHMS)};\n"
        f"const PROFILE_NAMES = {json.dumps(list(PROFILES.keys()))};\n"
    )
    return HTML_TEMPLATE.replace("/*__DATA__*/", data_js)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dhaka Route Explorer — Algorithm Playback</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background:#1a1a2e; color:#e0e0e0; display:flex; height:100vh; }

#sidebar {
  width: 390px; min-width:390px; background:#16213e; display:flex; flex-direction:column;
  border-right: 2px solid #0f3460; overflow-y:auto;
}
#sidebar h1 { font-size:16px; padding:12px 16px 6px; color:#e94560; letter-spacing:0.5px; }

.ctrl-group { padding:6px 16px; border-bottom:1px solid #0f3460; }
.ctrl-group label { display:block; font-size:10px; color:#888; text-transform:uppercase; margin-bottom:3px; letter-spacing:0.5px; }
.ctrl-group select, .ctrl-group input[type=range] {
  width:100%; background:#0f3460; color:#e0e0e0; border:1px solid #1a1a5e;
  border-radius:4px; padding:5px 7px; font-size:12px;
}
.ctrl-group select:focus { outline:2px solid #e94560; }

#step-controls { padding:10px 16px; display:flex; gap:4px; flex-wrap:wrap; align-items:center; border-bottom:1px solid #0f3460; }
#step-controls button {
  background:#0f3460; color:#e0e0e0; border:1px solid #1a1a5e; border-radius:4px;
  padding:5px 9px; cursor:pointer; font-size:12px; transition: background 0.15s;
}
#step-controls button:hover { background:#e94560; }
#step-controls button.active { background:#e94560; }
#speed-label { font-size:10px; color:#888; margin-left:auto; }

#info-panel { padding:8px 16px; font-size:12px; line-height:1.5; border-bottom:1px solid #0f3460; }
#info-panel .stat { display:flex; justify-content:space-between; padding:1px 0; }
#info-panel .stat .val { color:#e94560; font-weight:600; font-variant-numeric: tabular-nums; }

#edge-panel {
  padding:10px 16px; flex:1; overflow-y:auto; font-size:11px;
}
#edge-panel h3 { color:#e94560; font-size:12px; margin-bottom:5px; }
#edge-panel table { width:100%; border-collapse:collapse; }
#edge-panel td { padding:1px 3px; }
#edge-panel td:first-child { color:#888; }
.metric-bar { display:inline-block; height:6px; border-radius:2px; vertical-align:middle; margin-left:4px; }

#map { flex:1; }

.legend {
  background:rgba(22,33,62,0.92); padding:8px 12px; border-radius:6px;
  font-size:11px; line-height:1.6; color:#e0e0e0; border:1px solid #0f3460;
}
.legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.legend-swatch { display:inline-block; width:20px; height:3px; margin-right:5px; vertical-align:middle; border-radius:2px; }

.loc-label {
  font-size:11px; font-weight:700; color:#fff; white-space:nowrap;
  text-shadow: 0 0 6px rgba(0,0,0,0.9), 0 0 2px rgba(0,0,0,1);
  pointer-events:none;
}
.loc-label-major { font-size:12px; color:#ffea00; }
.loc-dot {
  width:5px; height:5px; border-radius:50%; background:#ffea00; border:1px solid rgba(0,0,0,0.5);
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
}
.leaflet-control-layers { background:rgba(22,33,62,0.92) !important; color:#e0e0e0 !important; border:1px solid #0f3460 !important; }
.leaflet-control-layers label { color:#e0e0e0 !important; }

#progress-bar { height:3px; background:#0f3460; margin:0 16px; border-radius:2px; }
#progress-fill { height:100%; background:#e94560; border-radius:2px; width:0%; transition: width 0.1s; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Dhaka Route Explorer</h1>

  <div class="ctrl-group">
    <label>Network</label>
    <select id="sel-mode">
      <option value="drive">Drive (h_ALT)</option>
      <option value="walk">Walk (h_B)</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Algorithm</label>
    <select id="sel-algo"></select>
  </div>
  <div class="ctrl-group">
    <label>Weight Profile</label>
    <select id="sel-profile"></select>
  </div>
  <div class="ctrl-group">
    <label>Source → Destination</label>
    <select id="sel-pair"></select>
  </div>
  <div class="ctrl-group">
    <label>Edge Coloring</label>
    <select id="sel-color">
      <option value="highway">By Road Type</option>
      <option value="composite">By Composite Cost</option>
      <option value="road_length">road_length</option>
      <option value="traffic_jam">traffic_jam</option>
      <option value="road_condition">road_condition</option>
      <option value="women_safety">women_safety</option>
      <option value="police_availability">police_availability</option>
      <option value="snatching_risk">snatching_risk</option>
      <option value="flood_risk">flood_risk</option>
      <option value="lighting">lighting</option>
      <option value="footpath">footpath</option>
      <option value="noise">noise</option>
    </select>
  </div>

  <div id="progress-bar"><div id="progress-fill"></div></div>

  <div id="step-controls">
    <button id="btn-reset" title="Reset">⏮</button>
    <button id="btn-back" title="Step back (←)">◀</button>
    <button id="btn-step" title="Step forward (→)">▶</button>
    <button id="btn-play" title="Play / Pause (Space)">⏵</button>
    <button id="btn-end" title="Jump to end">⏭</button>
    <span id="speed-label">Speed</span>
    <input type="range" id="speed" min="1" max="300" value="30" style="width:70px">
  </div>

  <div id="info-panel">
    <div class="stat"><span>Step</span><span class="val" id="s-step">0 / 0</span></div>
    <div class="stat"><span>Nodes Expanded</span><span class="val" id="s-expanded">0</span></div>
    <div class="stat"><span>Current Frontier</span><span class="val" id="s-frontier">0</span></div>
    <div class="stat"><span>Max Frontier</span><span class="val" id="s-maxfrontier">0</span></div>
    <div class="stat"><span>Path Cost</span><span class="val" id="s-cost">—</span></div>
    <div class="stat"><span>Path Length</span><span class="val" id="s-pathlen">—</span></div>
    <div class="stat"><span>Total Time</span><span class="val" id="s-time">—</span></div>
    <div class="stat"><span>Status</span><span class="val" id="s-status">Ready</span></div>
  </div>

  <div id="edge-panel">
    <h3>Click an edge for metrics</h3>
    <div id="edge-info"></div>
  </div>
</div>

<div id="map"></div>

<script>
/*__DATA__*/

// ---- State ----
let currentGraph = null;
let currentTraces = null;
let currentTrace = null;
let step = 0;
let playing = false;
let playTimer = null;

// ---- Leaflet setup ----
const map = L.map('map', {preferCanvas: true}).setView([23.765, 90.39], 13);

const baseDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
  maxZoom: 19, attribution: '&copy; OSM &amp; CARTO'
});
const baseLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png', {
  maxZoom: 19, attribution: '&copy; OSM &amp; CARTO'
});
const baseSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19, attribution: '&copy; Esri'
});
baseDark.addTo(map);

const labelsOverlay = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png', {
  maxZoom: 19, pane: 'overlayPane'
});
L.control.layers(
  {'Dark': baseDark, 'Light': baseLight, 'Satellite': baseSatellite},
  {'Map Labels': labelsOverlay},
  {position: 'topright', collapsed: true}
).addTo(map);

// Dhaka location markers
const LOCATIONS = [
  {name:"Dhanmondi", lat:23.746, lng:90.374, major:true},
  {name:"Gulshan-1", lat:23.781, lng:90.413, major:true},
  {name:"Banani", lat:23.794, lng:90.403, major:true},
  {name:"Farmgate", lat:23.757, lng:90.387, major:true},
  {name:"Mohammadpur", lat:23.766, lng:90.359, major:true},
  {name:"Shahbagh", lat:23.738, lng:90.396, major:true},
  {name:"Karwan Bazar", lat:23.751, lng:90.393, major:true},
  {name:"Mohakhali", lat:23.778, lng:90.405, major:true},
  {name:"Ramna", lat:23.740, lng:90.405, major:true},
  {name:"Motijheel", lat:23.728, lng:90.417},
  {name:"Tejgaon", lat:23.763, lng:90.393},
  {name:"Mirpur-10", lat:23.807, lng:90.369},
  {name:"Sadarghat", lat:23.708, lng:90.407},
  {name:"Lalbagh", lat:23.719, lng:90.389},
  {name:"Paltan", lat:23.734, lng:90.412},
  {name:"Old Dhaka", lat:23.723, lng:90.406},
];
const locationLayer = L.layerGroup().addTo(map);
LOCATIONS.forEach(loc => {
  const cls = loc.major ? 'loc-label loc-label-major' : 'loc-label';
  const icon = L.divIcon({
    className: '',
    html: `<div style="position:relative"><div class="loc-dot"></div><div class="${cls}" style="position:absolute;left:10px;top:-8px">${loc.name}</div></div>`,
    iconSize: [0,0], iconAnchor: [0,0]
  });
  L.marker([loc.lat, loc.lng], {icon, interactive:false, zIndexOffset:1000}).addTo(locationLayer);
});

// ---- Render layers ----
let edgeLayer = null;
let exploredLayer = L.layerGroup().addTo(map);
let frontierLayer = L.layerGroup().addTo(map);
let pathLayer = L.layerGroup().addTo(map);
let markerLayer = L.layerGroup().addTo(map);
let currentNodeMarker = null;

// ---- Legend ----
const legend = L.control({position:'bottomright'});
legend.onAdd = function(){
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = `
    <b>Playback Legend</b><br>
    <span class="legend-dot" style="background:#00e676"></span> Source<br>
    <span class="legend-dot" style="background:#ff1744"></span> Destination<br>
    <span class="legend-dot" style="background:#ff9100"></span> Current Node<br>
    <span class="legend-dot" style="background:#448aff"></span> Explored<br>
    <span class="legend-dot" style="background:#ffea00"></span> Frontier<br>
    <span class="legend-swatch" style="background:#00e676"></span> Final Path`;
  return d;
};
legend.addTo(map);

// ---- Populate selects ----
const selMode = document.getElementById('sel-mode');
const selAlgo = document.getElementById('sel-algo');
const selProfile = document.getElementById('sel-profile');
const selPair = document.getElementById('sel-pair');
const selColor = document.getElementById('sel-color');

ALGO_NAMES.forEach(a => { const o=document.createElement('option'); o.value=a; o.textContent=a; selAlgo.appendChild(o); });
PROFILE_NAMES.forEach(p => { const o=document.createElement('option'); o.value=p; o.textContent=p; selProfile.appendChild(o); });
SD_PAIRS.forEach(p => { const o=document.createElement('option'); o.value=p; o.textContent=p; selPair.appendChild(o); });

// ---- Color helpers ----
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + t * 185);
  const g = Math.round(1 + t * 220);
  const b = Math.round(84 + (1-t) * 170);
  return `rgb(${r},${g},${b})`;
}
function costColor(v) {
  v = Math.max(0, Math.min(1, v));
  if (v < 0.5) {
    const t = v * 2;
    return `rgb(${Math.round(t*255)},${Math.round(200+t*55)},${Math.round((1-t)*80)})`;
  } else {
    const t = (v - 0.5) * 2;
    return `rgb(255,${Math.round(255*(1-t))},0)`;
  }
}
function hwColor(hw) {
  if (hw.includes('trunk') || hw.includes('primary')) return '#555577';
  if (hw.includes('secondary')) return '#444466';
  if (hw.includes('tertiary')) return '#3a3a55';
  if (hw.includes('footway') || hw.includes('path')) return '#2a4a44';
  return '#2a2a44';
}
function hwWeight(hw) {
  if (hw.includes('trunk') || hw.includes('primary')) return 2.2;
  if (hw.includes('secondary')) return 1.5;
  if (hw.includes('tertiary')) return 1.1;
  return 0.8;
}

// ---- Graph rendering ----
function getEdgeStyle(e) {
  const mode = selColor.value;
  if (mode === 'highway') return {color: hwColor(e.hw), weight: hwWeight(e.hw), opacity: 0.45};
  if (mode === 'composite') {
    const prof = PROFILES[selProfile.value];
    let cc = 0;
    METRICS.forEach(m => cc += (prof[m]||0) * (e[m]||0));
    return {color: costColor(cc), weight: 1.2, opacity: 0.55};
  }
  const v = e[mode] || 0;
  return {color: costColor(v), weight: 1.2, opacity: 0.55};
}

function renderGraph() {
  if (edgeLayer) map.removeLayer(edgeLayer);
  const g = currentGraph;
  const features = [];
  g.edges.forEach(e => {
    const u = g.nodes[e.u], v = g.nodes[e.v];
    if (!u || !v) return;
    features.push({
      type:'Feature', properties: e,
      geometry: {type:'LineString', coordinates:[[u[1],u[0]],[v[1],v[0]]]}
    });
  });
  edgeLayer = L.geoJSON({type:'FeatureCollection', features}, {
    style: f => getEdgeStyle(f.properties),
    onEachFeature: (f, layer) => layer.on('click', () => showEdgeInfo(f.properties))
  }).addTo(map);
  edgeLayer.bringToBack();
}

function showEdgeInfo(props) {
  const panel = document.getElementById('edge-info');
  const prof = PROFILES[selProfile.value];
  let cc = 0;
  METRICS.forEach(m => cc += (prof[m]||0) * (props[m]||0));

  const rows = METRICS.map(m => {
    const val = props[m] || 0;
    const w = prof[m] || 0;
    const contrib = val * w;
    const barW = Math.round(val * 80);
    const color = val < 0.3 ? '#00e676' : val < 0.6 ? '#ffea00' : '#ff1744';
    return `<tr>
      <td>${m.replace(/_/g,' ')}</td>
      <td>${val.toFixed(3)}</td>
      <td><span class="metric-bar" style="width:${barW}px;background:${color}"></span></td>
      <td>&times;${w.toFixed(2)}</td>
      <td style="text-align:right">${contrib.toFixed(4)}</td>
    </tr>`;
  }).join('');

  panel.innerHTML = `
    <table>
      <tr><td colspan="5" style="color:#e94560;padding-bottom:3px"><b>${props.u} &rarr; ${props.v}</b> <span style="color:#888">(${props.hw})</span></td></tr>
      ${rows}
      <tr><td colspan="3" style="border-top:1px solid #0f3460;padding-top:3px"><b>Composite</b></td>
          <td colspan="2" style="border-top:1px solid #0f3460;padding-top:3px;text-align:right;color:#e94560"><b>${cc.toFixed(4)}</b></td></tr>
    </table>`;
}

// ---- Algorithm trace playback ----
function loadTrace() {
  stopPlay();
  step = 0;
  const key = `${selAlgo.value}|${selProfile.value}|${selPair.value}`;
  currentTrace = currentTraces[key];

  markerLayer.clearLayers();
  if (!currentTrace) {
    document.getElementById('s-status').textContent = 'No trace';
    return;
  }

  const src = currentGraph.nodes[currentTrace.src];
  const dst = currentGraph.nodes[currentTrace.dst];
  if (src) L.circleMarker([src[0],src[1]], {radius:8, color:'#00e676', fillColor:'#00e676', fillOpacity:0.9, weight:2}).bindTooltip('Source').addTo(markerLayer);
  if (dst) L.circleMarker([dst[0],dst[1]], {radius:8, color:'#ff1744', fillColor:'#ff1744', fillOpacity:0.9, weight:2}).bindTooltip('Destination').addTo(markerLayer);

  if (src && dst) map.fitBounds([[src[0],src[1]],[dst[0],dst[1]]], {padding:[60,60]});

  renderStep();
}

function renderStep() {
  if (!currentTrace) return;
  exploredLayer.clearLayers();
  frontierLayer.clearLayers();
  pathLayer.clearLayers();
  if (currentNodeMarker) { map.removeLayer(currentNodeMarker); currentNodeMarker = null; }

  const expl = currentTrace.exploration;
  const total = expl.length;
  const cur = Math.min(step, total - 1);
  const explored = new Set();

  // Explored nodes (gradient by order)
  for (let i = 0; i <= cur; i++) {
    const nid = expl[i];
    explored.add(nid);
    const pos = currentGraph.nodes[nid];
    if (!pos) continue;
    const t = total > 1 ? i / (total - 1) : 0;
    L.circleMarker([pos[0], pos[1]], {
      radius: 3, color: viridis(t), fillColor: viridis(t), fillOpacity: 0.8, weight: 0
    }).addTo(exploredLayer);
  }

  // Current node
  if (cur >= 0 && cur < total) {
    const pos = currentGraph.nodes[expl[cur]];
    if (pos) {
      currentNodeMarker = L.circleMarker([pos[0], pos[1]], {
        radius: 7, color: '#ff9100', fillColor: '#ff9100', fillOpacity: 1, weight: 2
      }).bindTooltip(`Step ${cur+1}: node ${expl[cur]}`).addTo(map);
    }
  }

  // Frontier
  const adj = currentGraph.adjacency;
  const frontierSet = new Set();
  for (const nid of explored) {
    const nbrs = adj[nid] || [];
    for (const nb of nbrs) {
      if (!explored.has(nb) && !frontierSet.has(nb)) {
        frontierSet.add(nb);
        const pos = currentGraph.nodes[nb];
        if (pos) L.circleMarker([pos[0], pos[1]], {
          radius: 2.5, color: '#ffea00', fillColor: '#ffea00', fillOpacity: 0.7, weight: 0
        }).addTo(frontierLayer);
      }
    }
  }

  // Final path (at last step)
  let pathShown = false;
  if (cur >= total - 1 && currentTrace.found && currentTrace.path.length > 1) {
    const coords = currentTrace.path.map(nid => {
      const p = currentGraph.nodes[nid];
      return p ? [p[0], p[1]] : null;
    }).filter(Boolean);
    L.polyline(coords, {color:'#00e676', weight:4, opacity:0.9}).addTo(pathLayer);
    pathShown = true;
  }

  // Info panel
  document.getElementById('s-step').textContent = `${cur + 1} / ${total}`;
  document.getElementById('s-expanded').textContent = cur + 1;
  document.getElementById('s-frontier').textContent = frontierSet.size;
  document.getElementById('s-maxfrontier').textContent = currentTrace.frontier;
  document.getElementById('s-cost').textContent = pathShown ? currentTrace.cost.toFixed(4) : '—';
  document.getElementById('s-pathlen').textContent = pathShown ? currentTrace.path.length - 1 : '—';
  document.getElementById('s-time').textContent = currentTrace.time_ms.toFixed(2) + ' ms';
  document.getElementById('s-status').textContent = pathShown ? (currentTrace.found ? 'Path found' : 'No path') : 'Exploring...';
  document.getElementById('progress-fill').style.width = `${((cur+1)/total)*100}%`;
}

// ---- Controls ----
function stepForward() { if (currentTrace && step < currentTrace.exploration.length - 1) { step++; renderStep(); } else { stopPlay(); } }
function stepBack() { if (step > 0) { step--; renderStep(); } }
function reset() { step = 0; renderStep(); }
function jumpEnd() { if (currentTrace) { step = currentTrace.exploration.length - 1; renderStep(); } }

function togglePlay() { playing ? stopPlay() : startPlay(); }
function startPlay() {
  playing = true;
  document.getElementById('btn-play').textContent = '⏸';
  document.getElementById('btn-play').classList.add('active');
  scheduleNext();
}
function stopPlay() {
  playing = false;
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  document.getElementById('btn-play').textContent = '⏵';
  document.getElementById('btn-play').classList.remove('active');
}
function scheduleNext() {
  if (!playing) return;
  const speed = parseInt(document.getElementById('speed').value);
  const delay = Math.max(5, 1000 / speed);
  const batch = speed > 100 ? Math.floor(speed / 20) : 1;
  playTimer = setTimeout(() => {
    if (!currentTrace || !playing) return;
    for (let i = 0; i < batch && step < currentTrace.exploration.length - 1; i++) step++;
    renderStep();
    if (step < currentTrace.exploration.length - 1) scheduleNext();
    else stopPlay();
  }, delay);
}

document.getElementById('btn-step').addEventListener('click', stepForward);
document.getElementById('btn-back').addEventListener('click', stepBack);
document.getElementById('btn-reset').addEventListener('click', reset);
document.getElementById('btn-end').addEventListener('click', jumpEnd);
document.getElementById('btn-play').addEventListener('click', togglePlay);

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') stepForward();
  else if (e.key === 'ArrowLeft') stepBack();
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'Home') reset();
  else if (e.key === 'End') jumpEnd();
});

function switchMode() {
  const m = selMode.value;
  currentGraph = m === 'drive' ? DRIVE_GRAPH : WALK_GRAPH;
  currentTraces = m === 'drive' ? DRIVE_TRACES : WALK_TRACES;
  renderGraph();
  loadTrace();
}

selMode.addEventListener('change', switchMode);
selAlgo.addEventListener('change', loadTrace);
selProfile.addEventListener('change', () => { renderGraph(); loadTrace(); });
selPair.addEventListener('change', loadTrace);
selColor.addEventListener('change', renderGraph);

switchMode();
</script>
</body>
</html>"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading graphs...")
    G_drive = load_graph("drive")
    G_walk = load_graph("walk")

    # Load landmarks for drive (if available)
    landmark_data_raw = None
    if os.path.exists(LANDMARKS_PATH):
        print(f"Loading landmarks from {LANDMARKS_PATH}...")
        with open(LANDMARKS_PATH, "rb") as f:
            all_landmarks = pickle.load(f)
        landmark_data_raw = all_landmarks.get("dhaka_drive")
        if landmark_data_raw:
            print(f"  Drive: {len(landmark_data_raw['landmarks'])} landmarks available → h_ALT")
    else:
        print(f"  No landmarks.pkl — all informed algos will use h_B")

    print("\nExporting drive graph...")
    drive_graph = export_graph(G_drive)
    print(f"  {len(drive_graph['nodes'])} nodes, {len(drive_graph['edges'])} edges")

    print("\nExporting walk graph...")
    walk_graph = export_graph(G_walk)
    print(f"  {len(walk_graph['nodes'])} nodes, {len(walk_graph['edges'])} edges")

    print("\nRunning traces on drive network...")
    drive_traces = run_all_traces(G_drive, "drive", landmark_data_raw)
    print(f"  {len(drive_traces)} traces")

    print("\nRunning traces on walk network (h_B only)...")
    walk_traces = run_all_traces(G_walk, "walk", None)
    print(f"  {len(walk_traces)} traces")

    print("\nBuilding HTML...")
    html = build_html(drive_graph, walk_graph, drive_traces, walk_traces)

    out_path = os.path.join(OUT_DIR, "explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n✓ Saved to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
