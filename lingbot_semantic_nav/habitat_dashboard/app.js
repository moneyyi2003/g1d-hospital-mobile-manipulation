"use strict";

const app = {
  config: null,
  state: null,
  mapImages: {},
  activeLayer: null,
  camera: { loading: false, pendingUrl: null, displayedUrl: null },
  transientMessage: { text: "", until: 0 },
};
const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", async () => {
  $("commandForm").addEventListener("submit", submitCommand);
  $("cancelButton").addEventListener("click", cancelNavigation);
  setupSpeech();
  try {
    app.config = await request("/api/config");
    const layers = app.config.map.layers || [{ id: "occupancy", label: "Occupancy", asset: app.config.map.asset }];
    app.activeLayer = layers[0].id;
    await Promise.all(layers.map(async (layer) => { app.mapImages[layer.id] = await loadImage(layer.asset); }));
    hydrate();
    if (app.config.mode === "habitat") connectCameraStream();
    await refresh();
    window.setTimeout(poll, 100);
  } catch (error) { $("message").textContent = `载入失败：${error.message}`; }
});

async function request(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function loadImage(url) { return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url; }); }

function hydrate() {
  const mappingMode = app.config.mode === "rgb_only_mapping_navigation";
  if (mappingMode) {
    document.title = "LingBot · RGB 重建地图导航";
    $("appName").textContent = "LingBot RGB Mapping";
    $("appSubtitle").textContent = "RGB-ONLY RECONSTRUCTION NAVIGATION";
    $("backendStatus").innerHTML = "<i></i>LINGBOT-MAP · NO HABITAT GT";
    $("navigationSource").textContent = "RGB ONLY · LINGBOT-MAP · OCCUPANCY A*";
    $("cameraLabel").textContent = "RGB RECONSTRUCTION INPUT";
  } else {
    document.title = "LingBot · Habitat 实时导航";
    $("appName").textContent = "LingBot Habitat";
    $("backendStatus").innerHTML = "<i></i>HABITAT-SIM 0.3.3";
    $("cameraLabel").textContent = "RGB SENSOR · LIVE";
  }
  $("sceneName").textContent = app.config.scene;
  const layers = app.config.map.layers || [];
  $("mapTabs").innerHTML = layers.map((layer) => `<button type="button" data-layer="${escapeHtml(layer.id)}">${escapeHtml(layer.label)}</button>`).join("");
  $("mapTabs").querySelectorAll("button").forEach((button) => button.onclick = () => {
    app.activeLayer = button.dataset.layer;
    renderMapControls();
    drawMap();
  });
  renderMapControls();

  const regionPlaces = app.config.places.filter((place) => place.metadata?.target_type === "semantic_region");
  $("placeCount").textContent = `${regionPlaces.length} REGIONS`;
  $("examples").innerHTML = app.config.examples.map((text) => `<button type="button">${escapeHtml(text)}</button>`).join("");
  $("examples").querySelectorAll("button").forEach((button) => button.onclick = () => { $("commandInput").value = button.textContent; });
  $("places").innerHTML = regionPlaces.map((place) => {
    const outside = Boolean(place.metadata?.exploration?.outside_start_region);
    const suffix = outside ? " · 房间外" : "";
    return `<button type="button" data-id="${escapeHtml(place.id)}" class="${outside ? "outside" : ""}"><i></i><span><strong>${escapeHtml(place.name)}</strong><small>${escapeHtml(place.region + suffix)}</small></span></button>`;
  }).join("");
  $("places").querySelectorAll("button").forEach((button) => button.onclick = () => {
    const place = regionPlaces.find((item) => item.id === button.dataset.id);
    $("commandInput").value = `请带我到${place.name}`;
  });
  const regions = app.config.detected_regions || [];
  $("regionCount").textContent = `${regions.length} REGIONS`;
  $("regions").innerHTML = regions.map((item) => `<span data-place-id="${escapeHtml(item.place_id || "")}" class="${item.navigable ? "navigable" : ""} ${item.outside_start_region ? "outside" : ""}" title="${item.area_m2.toFixed(2)} map² · ${item.cells} cells · confidence ${Number(item.confidence || 0).toFixed(2)}"><i></i>${escapeHtml(item.name)}<small>${item.outside_start_region ? `房间外 · ${item.region_hops_from_start} hop` : `${Number(item.confidence || 0).toFixed(2)} conf`}</small></span>`).join("");
  $("regions").querySelectorAll(".navigable").forEach((chip) => chip.onclick = () => {
    const place = app.config.places.find((item) => item.id === chip.dataset.placeId);
    if (place) $("commandInput").value = `请带我到${place.name}`;
  });
  const objects = app.config.detected_objects || app.config.object_candidates || [];
  const navigableObjectCount = objects.filter((item) => item.navigable).length;
  $("candidateCount").textContent = `${navigableObjectCount}/${objects.length} NAVIGABLE`;
  $("candidates").innerHTML = objects.map((item) => `<span data-id="${escapeHtml(item.id)}" class="${item.navigable ? "accepted navigable" : "pending"} ${item.outside_start_region ? "outside" : ""}" title="${item.navigable ? "LingBot occupancy 上存在安全停靠点，点击填入指令" : "LingBot occupancy 上没有可达停靠点"}"><i></i>${escapeHtml(item.name)}<small>${item.navigable ? "可导航" : "不可达"} · ${item.observation_count} obs</small></span>`).join("");
  $("candidates").querySelectorAll(".navigable").forEach((chip) => chip.onclick = () => {
    const item = objects.find((candidate) => candidate.id === chip.dataset.id);
    $("commandInput").value = `请带我到${item.name}`;
  });
}

function renderMapControls() {
  const layers = app.config.map.layers || [];
  const active = layers.find((layer) => layer.id === app.activeLayer);
  $("mapTabs").querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.layer === app.activeLayer));
  $("mapDescription").textContent = active ? active.description : "";
}

async function submitCommand(event) {
  event.preventDefault();
  const command = $("commandInput").value.trim();
  if (!command) return $("commandInput").focus();
  $("submitButton").disabled = true;
  $("message").textContent = "正在解析目标并在 LingBot 重建占据图上规划路线…";
  const occupancyLayer = (app.config.map.layers || []).find((layer) => layer.id === "occupancy");
  if (occupancyLayer) {
    app.activeLayer = occupancyLayer.id;
    renderMapControls();
  }
  try {
    app.state = await request("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }) });
    render();
  } catch (error) {
    app.transientMessage = { text: `指令未执行：${error.message}`, until: Date.now() + 5000 };
    $("message").textContent = app.transientMessage.text;
  }
  finally { $("submitButton").disabled = false; }
}
async function cancelNavigation() { app.state = await request("/api/cancel", { method: "POST" }); render(); }
async function refresh() { app.state = await request("/api/state"); render(); }
function connectCameraStream() {
  const image = $("cameraView");
  image.src = `/stream/camera.mjpg?session=${Date.now()}`;
  $("cameraEmpty").hidden = true;
  image.onerror = () => window.setTimeout(connectCameraStream, 1000);
}
async function poll() {
  try { await refresh(); }
  catch (error) { $("message").textContent = `连接中断：${error.message}`; }
  window.setTimeout(poll, 100);
}

function render() {
  const state = app.state;
  if (!state) return;
  const badge = $("stateBadge"); badge.textContent = state.state.toUpperCase(); badge.className = `badge ${state.state}`;
  $("message").textContent = app.transientMessage.until > Date.now()
    ? app.transientMessage.text
    : state.message;
  $("destination").textContent = state.destination_name || "—";
  $("elapsed").textContent = `${state.elapsed_sec.toFixed(1)} s`;
  $("action").textContent = String(state.action || "—").replaceAll("_", " ");
  $("linearVelocity").textContent = `${state.linear_velocity_mps.toFixed(2)} m/s`;
  $("angularVelocity").textContent = `${state.angular_velocity_rps.toFixed(2)} rad/s`;
  $("distance").textContent = state.distance_remaining == null ? "—" : `${state.distance_remaining.toFixed(2)} m`;
  $("frame").textContent = state.frame;
  $("collisions").textContent = state.collisions;
  $("coordinates").textContent = `x ${state.pose.x.toFixed(2)} · z ${state.pose.y.toFixed(2)} · yaw ${state.pose.yaw.toFixed(2)}`;
  $("route").innerHTML = state.route.map((step, i) => `<span><b>${i + 1}</b>${escapeHtml(step.name)}<small>${step.action}</small></span>`).join("");
  $("pipeline").innerHTML = (state.pipeline || []).map((stage) => `<span class="${stage.state}"><i></i>${escapeHtml(stage.name)}</span>`).join("");
  const target = state.target;
  const objectTarget = target && target.metadata && String(target.metadata.target_type || "").endsWith("object_instance");
  $("objectEvidence").hidden = !objectTarget;
  if (objectTarget) {
    const configured = app.config.places.find((place) => place.id === target.id);
    if (configured?.review_asset) $("objectReview").src = configured.review_asset;
    $("objectEvidenceType").textContent = target.metadata.verification?.status === "demo_enabled"
      ? "DEMO NAVIGABLE CANDIDATE"
      : "VERIFIED OBJECT INSTANCE";
    $("objectLabel").textContent = `${target.name} · ${target.metadata.instance_id}`;
    $("objectGeometry").textContent = `停靠距离 ${target.metadata.standoff_m.toFixed(2)} m · 净空 ${target.metadata.clearance_m.toFixed(2)} m`;
  }
  document.querySelectorAll(".places button").forEach((button) => button.classList.toggle("active", button.dataset.id === state.destination));
  if (app.config.mode !== "habitat") updateCamera(state.camera_url);
  drawMap();
}

function point(pose) {
  const bounds = app.config.map.bounds;
  const x = (pose.x - bounds.min_x) / (bounds.max_x - bounds.min_x) * app.config.map.width;
  const y = (pose.y - bounds.min_z) / (bounds.max_z - bounds.min_z) * app.config.map.height;
  return {
    x,
    y: app.config.map.flip_y ? app.config.map.height - y : y,
  };
}
function path(ctx, poses) { ctx.beginPath(); poses.forEach((pose, i) => { const p = point(pose); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); }); }
function updateCamera(url) {
  if (!url || app.camera.displayedUrl === url || app.camera.pendingUrl === url) return;
  app.camera.pendingUrl = url;
  loadPendingCamera();
}
function loadPendingCamera() {
  if (app.camera.loading || !app.camera.pendingUrl) return;
  const url = app.camera.pendingUrl;
  app.camera.pendingUrl = null;
  app.camera.loading = true;
  const image = new Image();
  image.id = "cameraView";
  image.alt = "Habitat 机器人第一视角";
  image.onload = () => {
    $("cameraView").replaceWith(image);
    $("cameraEmpty").hidden = true;
    app.camera.displayedUrl = url;
    app.camera.loading = false;
    loadPendingCamera();
  };
  image.onerror = () => {
    app.camera.loading = false;
    loadPendingCamera();
  };
  image.src = url;
}
function drawMap() {
  const mapImage = app.mapImages[app.activeLayer];
  if (!mapImage || !app.state) return;
  const canvas = $("mapCanvas"); canvas.width = app.config.map.width; canvas.height = app.config.map.height;
  const ctx = canvas.getContext("2d"); ctx.drawImage(mapImage, 0, 0, canvas.width, canvas.height);
  if (app.activeLayer === "occupancy") {
    ctx.globalCompositeOperation = "multiply"; ctx.fillStyle = "#a9d1d3"; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.globalCompositeOperation = "source-over";
  }
  if (app.activeLayer === "semantic" || app.activeLayer === "rgb_pointcloud") {
    for (const item of app.config.object_candidates || []) {
      if (!Number.isFinite(item.x) || !Number.isFinite(item.y)) continue;
      const p = point(item); ctx.beginPath(); ctx.moveTo(p.x - 2.5, p.y - 2.5); ctx.lineTo(p.x + 2.5, p.y + 2.5); ctx.moveTo(p.x + 2.5, p.y - 2.5); ctx.lineTo(p.x - 2.5, p.y + 2.5); ctx.strokeStyle = "rgba(255,173,92,.75)"; ctx.lineWidth = 1.2; ctx.stroke();
    }
  }
  for (const place of app.config.places) { const p = point({ x: place.entrance_pose.x, y: place.entrance_pose.y }); ctx.beginPath(); ctx.arc(p.x, p.y, 2.3, 0, Math.PI * 2); ctx.fillStyle = "#17353d"; ctx.fill(); }
  if ((app.state.exploration_trajectory || []).length > 1) { path(ctx, app.state.exploration_trajectory); ctx.save(); ctx.setLineDash([2, 5]); ctx.strokeStyle = "rgba(108,240,155,.9)"; ctx.lineWidth = 2.2; ctx.stroke(); ctx.restore(); }
  if (app.state.planned_trajectory.length > 1) { path(ctx, app.state.planned_trajectory); ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.7; ctx.stroke(); ctx.restore(); }
  if (app.state.trajectory.length > 1) { path(ctx, app.state.trajectory); ctx.strokeStyle = "#00d9df"; ctx.lineWidth = 2.8; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.shadowColor = "#00d9df"; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0; }
  if (app.state.planned_trajectory.length) { const goal = point(app.state.planned_trajectory.at(-1)); ctx.beginPath(); ctx.arc(goal.x, goal.y, 5, 0, Math.PI * 2); ctx.strokeStyle = "#ffad5c"; ctx.lineWidth = 2; ctx.stroke(); }
  const robot = point(app.state.pose);
  ctx.save();
  ctx.translate(robot.x, robot.y);
  // The triangle's nose points along canvas +X.  ROS yaw=0 also points along
  // map +X; only invert yaw when the map image flips its Y axis for display.
  const canvasYaw = app.config.map.flip_y ? -app.state.pose.yaw : app.state.pose.yaw;
  ctx.rotate(canvasYaw);
  ctx.beginPath(); ctx.moveTo(7, 0); ctx.lineTo(-5, -4); ctx.lineTo(-3, 0); ctx.lineTo(-5, 4); ctx.closePath(); ctx.fillStyle = "#6cf09b"; ctx.shadowColor = "#6cf09b"; ctx.shadowBlur = 7; ctx.fill(); ctx.restore();
}

function setupSpeech() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const button = $("micButton");
  if (!Recognition) { button.disabled = true; return; }
  const recognition = new Recognition(); recognition.lang = "zh-CN"; recognition.interimResults = false;
  button.onclick = () => recognition.start();
  recognition.onstart = () => { button.classList.add("listening"); $("speechHint").textContent = "正在听，请说出目标地点…"; };
  recognition.onresult = (event) => { $("commandInput").value = event.results[0][0].transcript; };
  recognition.onend = () => { button.classList.remove("listening"); if ($("commandInput").value.trim()) $("commandForm").requestSubmit(); };
  recognition.onerror = () => { button.classList.remove("listening"); $("speechHint").textContent = "无法使用麦克风，请检查浏览器权限"; };
}
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;" })[c]); }
