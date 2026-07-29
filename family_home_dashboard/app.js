const $ = (id) => document.getElementById(id);
const app = {
  config: null,
  mapData: null,
  state: null,
  cameraStarted: false,
};

const categoryColors = {
  wall: "#77837c",
  existing_furniture: "#b1bd79",
  bed: "#7395ff",
  dining_table: "#e9ad47",
  kitchen_counter: "#ee7670",
  cabinet: "#b89070",
};

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status}`);
  return payload;
}

function worldPoint(pose) {
  const { bounds, width, height, flip_y: flipY } = app.config.map;
  const x = (pose.x - bounds.min_x) / (bounds.max_x - bounds.min_x) * width;
  const rawY = (pose.y - bounds.min_y) / (bounds.max_y - bounds.min_y) * height;
  return { x, y: flipY ? height - rawY : rawY };
}

function worldRect(bounds) {
  const a = worldPoint({ x: bounds[0], y: bounds[1] });
  const b = worldPoint({ x: bounds[2], y: bounds[3] });
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(b.x - a.x),
    height: Math.abs(b.y - a.y),
  };
}

function setupCanvas(id) {
  const canvas = $(id);
  canvas.width = app.config.map.width * 4;
  canvas.height = app.config.map.height * 4;
  const ctx = canvas.getContext("2d");
  ctx.scale(4, 4);
  return ctx;
}

function trace(ctx, values) {
  ctx.beginPath();
  values.forEach((pose, index) => {
    const p = worldPoint(pose);
    index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
  });
}

function drawOverlay(ctx) {
  if (!app.state) return;
  const planned = app.state.planned_trajectory || [];
  if (planned.length > 1) {
    trace(ctx, planned);
    ctx.save();
    ctx.setLineDash([2.5, 2.5]);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(241,246,242,.94)";
    ctx.stroke();
    ctx.restore();
    const goal = worldPoint(planned.at(-1));
    ctx.beginPath();
    ctx.arc(goal.x, goal.y, 2.3, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffb55e";
    ctx.lineWidth = 1.3;
    ctx.stroke();
  }
  const actual = app.state.trajectory || [];
  if (actual.length > 1) {
    trace(ctx, actual);
    ctx.lineWidth = 1.7;
    ctx.strokeStyle = "#43d9e6";
    ctx.lineJoin = "round";
    ctx.stroke();
  }
  for (const place of app.config.places) {
    const p = worldPoint(place.pose);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.45, 0, Math.PI * 2);
    ctx.fillStyle = "#ffb55e";
    ctx.fill();
  }
  const pose = app.state.pose;
  const robot = worldPoint(pose);
  ctx.save();
  ctx.translate(robot.x, robot.y);
  ctx.rotate(app.config.map.flip_y ? -pose.yaw : pose.yaw);
  ctx.beginPath();
  ctx.moveTo(4.2, 0);
  ctx.lineTo(-3.1, -2.6);
  ctx.lineTo(-1.9, 0);
  ctx.lineTo(-3.1, 2.6);
  ctx.closePath();
  ctx.fillStyle = "#b8f34a";
  ctx.shadowColor = "#b8f34a";
  ctx.shadowBlur = 4;
  ctx.fill();
  ctx.restore();
}

function drawPointcloud() {
  const ctx = setupCanvas("pointcloudCanvas");
  const { width, height } = app.config.map;
  ctx.fillStyle = "#060b09";
  ctx.fillRect(0, 0, width, height);
  for (const point of app.mapData.pointcloud) {
    const p = worldPoint(point);
    ctx.fillStyle = categoryColors[point.category] || "#7ce0b2";
    ctx.globalAlpha = .72;
    ctx.fillRect(p.x - .4, p.y - .4, .8, .8);
  }
  ctx.globalAlpha = 1;
  drawOverlay(ctx);
}

function drawSemantic() {
  const ctx = setupCanvas("semanticCanvas");
  const { width, height } = app.config.map;
  ctx.fillStyle = "#09100c";
  ctx.fillRect(0, 0, width, height);
  for (const fixture of app.mapData.fixtures) {
    const rect = worldRect(fixture.bounds_xy);
    const color = fixture.color_rgb.map((value) => Math.round(value * 255));
    ctx.fillStyle = `rgb(${color.join(",")})`;
    ctx.globalAlpha = .88;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.globalAlpha = 1;
    if (rect.width > 8 && rect.height > 5) {
      ctx.fillStyle = "#f2f6f3";
      ctx.font = "2.4px monospace";
      ctx.textAlign = "center";
      ctx.fillText(fixture.category, rect.x + rect.width / 2, rect.y + rect.height / 2);
    }
  }
  drawOverlay(ctx);
}

function drawOccupancy() {
  const ctx = setupCanvas("occupancyCanvas");
  const { width, height } = app.config.map;
  ctx.fillStyle = "#070b09";
  ctx.fillRect(0, 0, width, height);
  const rows = app.mapData.occupancy_rows;
  const cellWidth = width / rows[0].length;
  const cellHeight = height / rows.length;
  rows.forEach((row, rowIndex) => {
    [...row].forEach((value, colIndex) => {
      ctx.fillStyle = value === "." ? "#dce4df" : "#242b27";
      const yIndex = app.config.map.flip_y ? rows.length - rowIndex - 1 : rowIndex;
      ctx.fillRect(colIndex * cellWidth, yIndex * cellHeight, cellWidth + .1, cellHeight + .1);
    });
  });
  drawOverlay(ctx);
}

function drawRegions() {
  const ctx = setupCanvas("regionCanvas");
  const { width, height } = app.config.map;
  ctx.fillStyle = "#080d0b";
  ctx.fillRect(0, 0, width, height);
  for (const region of app.mapData.regions) {
    const rect = worldRect(region.bounds_xy);
    ctx.fillStyle = `rgb(${region.color_rgb.join(",")})`;
    ctx.globalAlpha = .18;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.globalAlpha = .9;
    ctx.strokeStyle = `rgb(${region.color_rgb.join(",")})`;
    ctx.lineWidth = .8;
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.fillStyle = "#edf3ef";
    ctx.font = "bold 3px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(region.name, rect.x + rect.width / 2, rect.y + rect.height / 2);
  }
  ctx.globalAlpha = 1;
  drawOverlay(ctx);
}

function drawMaps() {
  if (!app.config || !app.mapData || !app.state) return;
  drawPointcloud();
  drawSemantic();
  drawOccupancy();
  drawRegions();
}

function render() {
  const state = app.state;
  if (!state) return;
  const badge = $("stateBadge");
  badge.innerHTML = `<i></i>${state.state.toUpperCase()}`;
  badge.className = `state-badge ${state.state}`;
  $("message").textContent = state.message;
  $("task").textContent = state.task || "—";
  $("action").textContent = state.action || "—";
  $("pose").textContent = `x ${state.pose.x.toFixed(2)} · y ${state.pose.y.toFixed(2)} · ψ ${state.pose.yaw.toFixed(2)}`;
  $("velocity").textContent = `${(state.linear_velocity_mps || 0).toFixed(2)} m/s · ${(state.angular_velocity_rps || 0).toFixed(2)} rad/s`;
  $("waypoint").textContent = `${state.waypoint || 0} / ${state.waypoint_count || 0}`;
  $("timing").textContent = `${state.frame || 0} / ${(state.elapsed_sec || 0).toFixed(1)} s`;
  $("hudAction").textContent = (state.action || "standby").toUpperCase();
  $("hudTiming").textContent = `FRAME ${String(state.frame || 0).padStart(4, "0")}`;
  $("submitButton").disabled = Boolean(state.process_running);
  const progress = state.waypoint_count
    ? Math.min(100, (state.waypoint || 0) / state.waypoint_count * 100)
    : 0;
  $("progressBar").style.width = `${state.state === "succeeded" ? 100 : progress}%`;
  const cameraAvailable =
    state.process_running || ["succeeded", "failed", "canceled"].includes(state.state);
  if (cameraAvailable && !app.cameraStarted) {
    app.cameraStarted = true;
    $("cameraView").src = `${app.config.camera_stream}?session=${Date.now()}`;
    $("cameraView").style.display = "block";
    $("cameraEmpty").style.display = "none";
  }
  drawMaps();
}

async function poll() {
  try {
    app.state = await request("/api/state");
    render();
  } catch (error) {
    $("message").textContent = `控制台连接中断：${error.message}`;
  }
  window.setTimeout(poll, 180);
}

$("commandForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    app.cameraStarted = false;
    app.state = await request("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: $("commandInput").value.trim() }),
    });
    render();
  } catch (error) {
    $("message").textContent = `指令未执行：${error.message}`;
  }
});

$("cancelButton").addEventListener("click", async () => {
  try {
    app.state = await request("/api/cancel", { method: "POST" });
    render();
  } catch (error) {
    $("message").textContent = `停止失败：${error.message}`;
  }
});

async function initialize() {
  [app.config, app.mapData] = await Promise.all([
    request("/api/config"),
    request("/api/map-data"),
  ]);
  $("sourceBadge").textContent = app.config.map.source_label;
  $("mapTruth").textContent = app.mapData.truth_boundary || "FORMAL MAP BUNDLE";
  $("placeChips").innerHTML = app.config.places
    .map((place) => `<button type="button" data-id="${place.id}">${place.name}</button>`)
    .join("");
  $("placeChips").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const place = app.config.places.find((item) => item.id === button.dataset.id);
      $("commandInput").value = place.example;
    };
  });
  for (const layer of app.config.layers) {
    const status = $(`${layer.id}Status`);
    status.textContent = layer.status === "formal" ? "FORMAL" : "BOOTSTRAP";
    status.className = `layer-status ${layer.status}`;
    $(`${layer.id}Description`).textContent = layer.description;
  }
  await poll();
}

initialize().catch((error) => {
  $("message").textContent = `初始化失败：${error.message}`;
});
