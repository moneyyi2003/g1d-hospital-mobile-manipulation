const $ = (id) => document.getElementById(id);
const app = { config: null, state: null, images: {}, cameraStarted: false };

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status}`);
  return payload;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

function point(pose) {
  const map = app.config.map;
  const x = (pose.x - map.bounds.min_x) / (map.bounds.max_x - map.bounds.min_x) * map.width;
  const rawY = (pose.y - map.bounds.min_z) / (map.bounds.max_z - map.bounds.min_z) * map.height;
  return { x, y: map.flip_y ? map.height - rawY : rawY };
}

function trace(ctx, values) {
  ctx.beginPath();
  values.forEach((pose, index) => {
    const p = point(pose);
    index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
  });
}

function drawMap(canvasId, image) {
  if (!app.state || !image) return;
  const canvas = $(canvasId);
  canvas.width = app.config.map.width;
  canvas.height = app.config.map.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const planned = app.state.planned_trajectory || [];
  if (planned.length > 1) {
    trace(ctx, planned);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,.95)";
    ctx.stroke();
    ctx.restore();
    const goal = point(planned.at(-1));
    ctx.beginPath();
    ctx.arc(goal.x, goal.y, 5, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffad5c";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const actual = app.state.trajectory || [];
  if (actual.length > 1) {
    trace(ctx, actual);
    ctx.strokeStyle = "#3ce3e8";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.shadowColor = "#3ce3e8";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  for (const place of app.config.places) {
    const p = point(place.pose);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.7, 0, Math.PI * 2);
    ctx.fillStyle = "#ffad5c";
    ctx.fill();
  }
  const robot = point(app.state.pose);
  ctx.save();
  ctx.translate(robot.x, robot.y);
  ctx.rotate(app.config.map.flip_y ? -app.state.pose.yaw : app.state.pose.yaw);
  ctx.beginPath();
  ctx.moveTo(8, 0);
  ctx.lineTo(-6, -5);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-6, 5);
  ctx.closePath();
  ctx.fillStyle = "#63efa0";
  ctx.shadowColor = "#63efa0";
  ctx.shadowBlur = 8;
  ctx.fill();
  ctx.restore();
}

function render() {
  const state = app.state;
  if (!state) return;
  const badge = $("stateBadge");
  badge.textContent = state.state.toUpperCase();
  badge.className = `badge ${state.state}`;
  $("message").textContent = state.message;
  $("task").textContent = state.task || "—";
  const resolution = state.intent_resolution;
  $("intent").textContent = resolution
    ? `${resolution.parser} · ${resolution.place_name} · ${(resolution.confidence || 0).toFixed(2)}`
    : "—";
  $("action").textContent = state.action || "—";
  $("pose").textContent = `x ${state.pose.x.toFixed(2)} · y ${state.pose.y.toFixed(2)} · ψ ${state.pose.yaw.toFixed(2)}`;
  $("velocity").textContent = `${(state.linear_velocity_mps || 0).toFixed(2)} m/s · ${(state.angular_velocity_rps || 0).toFixed(2)} rad/s`;
  $("waypoint").textContent = `${state.waypoint || 0} / ${state.waypoint_count || 0}`;
  $("timing").textContent = `${state.frame || 0} · ${(state.elapsed_sec || 0).toFixed(1)} s`;
  $("submitButton").disabled = Boolean(state.process_running);
  drawMap("pointcloudCanvas", app.images.rgb_pointcloud);
  drawMap("occupancyCanvas", app.images.occupancy);
  const cameraAvailable =
    state.process_running || ["succeeded", "failed", "canceled"].includes(state.state);
  if (cameraAvailable && !app.cameraStarted) {
    app.cameraStarted = true;
    $("cameraView").src = `${app.config.camera_stream}?session=${Date.now()}`;
    $("cameraView").style.display = "block";
    $("cameraEmpty").style.display = "none";
  }
}

async function poll() {
  try {
    app.state = await request("/api/state");
    render();
  } catch (error) {
    $("message").textContent = `控制台连接中断：${error.message}`;
  }
  window.setTimeout(poll, 150);
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
  app.state = await request("/api/cancel", { method: "POST" });
  render();
});

async function initialize() {
  app.config = await request("/api/config");
  $("placeChips").innerHTML = app.config.places
    .map((place) => {
      const example = place.examples?.[0] || `请带我到${place.name}`;
      return `<button type="button" data-id="${place.id}">${example}</button>`;
    })
    .join("");
  $("placeChips").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const place = app.config.places.find((item) => item.id === button.dataset.id);
      $("commandInput").value = place.examples?.[0] || `请带我到${place.name}`;
    };
  });
  const layers = Object.fromEntries(app.config.map.layers.map((layer) => [layer.id, layer.asset]));
  [app.images.rgb_pointcloud, app.images.occupancy] = await Promise.all([
    loadImage(layers.rgb_pointcloud),
    loadImage(layers.occupancy),
  ]);
  await poll();
}

initialize().catch((error) => { $("message").textContent = `初始化失败：${error.message}`; });
