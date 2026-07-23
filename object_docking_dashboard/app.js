const $ = (id) => document.getElementById(id);
const app = {
  config: null,
  state: null,
  scene: null,
  images: {},
  cameraStarted: false,
};

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
    image.src = `${url}?v=${Date.now()}`;
  });
}

function sceneById(sceneId) {
  return app.config.scenes.find((item) => item.id === sceneId);
}

function point(pose) {
  const map = app.scene.map;
  const x = (pose.x - map.bounds.min_x) / (map.bounds.max_x - map.bounds.min_x) * map.width;
  const rawY = (pose.y - map.bounds.min_z) / (map.bounds.max_z - map.bounds.min_z) * map.height;
  return {x, y: map.flip_y ? map.height - rawY : rawY};
}

function trace(ctx, poses) {
  ctx.beginPath();
  poses.forEach((pose, index) => {
    const p = point(pose);
    index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
  });
}

function drawMarker(ctx, pose, color, radius) {
  const p = point(pose);
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  return p;
}

function drawMap(canvasId, image) {
  if (!app.state || !app.scene || !image) return;
  const canvas = $(canvasId);
  canvas.width = app.scene.map.width;
  canvas.height = app.scene.map.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const planned = app.state.planned_trajectory || [];
  if (planned.length > 1) {
    trace(ctx, planned);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,173,92,.98)";
    ctx.stroke();
    ctx.restore();
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

  const plan = app.state.docking_plan;
  if (plan?.target?.position) {
    const objectPoint = drawMarker(ctx, plan.target.position, "#ff5f6d", 5.5);
    ctx.strokeStyle = "rgba(255,95,109,.6)";
    ctx.lineWidth = 2;
    ctx.strokeRect(objectPoint.x - 8, objectPoint.y - 8, 16, 16);
  }
  if (plan?.docking_pose) {
    const dock = drawMarker(ctx, plan.docking_pose, "#ffad5c", 5);
    ctx.save();
    ctx.translate(dock.x, dock.y);
    ctx.rotate(app.scene.map.flip_y ? -plan.docking_pose.yaw : plan.docking_pose.yaw);
    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.lineTo(-6, -5);
    ctx.lineTo(-6, 5);
    ctx.closePath();
    ctx.strokeStyle = "#ffad5c";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  const robotPose = app.state.pose;
  if (robotPose) {
    const robot = point(robotPose);
    ctx.save();
    ctx.translate(robot.x, robot.y);
    ctx.rotate(app.scene.map.flip_y ? -robotPose.yaw : robotPose.yaw);
    ctx.beginPath();
    ctx.moveTo(9, 0);
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
}

function render() {
  const state = app.state;
  if (!state) return;
  const badge = $("stateBadge");
  badge.textContent = state.state.toUpperCase();
  badge.className = `badge ${state.state}`;
  $("message").textContent = state.message;
  $("scene").textContent = sceneById(state.scene_id)?.name || state.scene_id || "—";
  $("target").textContent = state.object_target?.name || state.task || "—";
  const plan = state.docking_plan;
  $("distance").textContent = plan
    ? `${plan.constraint.requested_standoff_m.toFixed(2)} m`
    : "—";
  $("dockPose").textContent = plan
    ? `${plan.docking_pose.x.toFixed(2)}, ${plan.docking_pose.y.toFixed(2)}, ${plan.docking_pose.yaw.toFixed(2)}`
    : "—";
  $("robotPose").textContent = state.pose
    ? `${state.pose.x.toFixed(2)}, ${state.pose.y.toFixed(2)}, ${state.pose.yaw.toFixed(2)}`
    : "—";
  $("action").textContent = state.action || "—";
  $("waypoint").textContent = `${state.waypoint || 0} / ${state.waypoint_count || 0}`;
  $("timing").textContent = `${state.frame || 0} · ${(state.elapsed_sec || 0).toFixed(1)} s`;
  $("result").textContent = state.result
    ? `${state.result.position_error_m.toFixed(3)} m · ${state.result.yaw_error_rad.toFixed(3)} rad`
    : "—";

  $("submitButton").disabled = Boolean(state.process_running);
  $("sceneSelect").disabled = Boolean(state.process_running);
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

function renderExamples() {
  $("exampleChips").replaceChildren();
  for (const object of app.scene.objects) {
    for (const example of object.examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = example;
      button.onclick = () => {$("commandInput").value = example;};
      $("exampleChips").append(button);
    }
  }
}

async function selectScene(sceneId) {
  app.scene = sceneById(sceneId);
  if (!app.scene) throw new Error(`场景配置不存在：${sceneId}`);
  renderExamples();
  const layers = Object.fromEntries(
    app.scene.map.layers.map((layer) => [layer.id, layer.asset])
  );
  [app.images.rgb_pointcloud, app.images.occupancy] = await Promise.all([
    loadImage(layers.rgb_pointcloud),
    loadImage(layers.occupancy),
  ]);
  render();
}

async function poll() {
  try {
    app.state = await request("/api/state");
    if (app.state.scene_id && app.scene?.id !== app.state.scene_id) {
      $("sceneSelect").value = app.state.scene_id;
      await selectScene(app.state.scene_id);
    }
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
    $("cameraView").removeAttribute("src");
    $("cameraView").style.display = "none";
    $("cameraEmpty").style.display = "grid";
    app.state = await request("/api/command", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        scene_id: $("sceneSelect").value,
        command: $("commandInput").value.trim(),
      }),
    });
    render();
  } catch (error) {
    $("message").textContent = `指令未执行：${error.message}`;
  }
});

$("cancelButton").addEventListener("click", async () => {
  try {
    app.state = await request("/api/cancel", {method: "POST"});
    render();
  } catch (error) {
    $("message").textContent = `停止失败：${error.message}`;
  }
});

$("sceneSelect").addEventListener("change", async (event) => {
  try {
    await selectScene(event.target.value);
  } catch (error) {
    $("message").textContent = `场景加载失败：${error.message}`;
  }
});

async function initialize() {
  app.config = await request("/api/config");
  for (const scene of app.config.scenes) {
    const option = document.createElement("option");
    option.value = scene.id;
    option.textContent = scene.name;
    $("sceneSelect").append(option);
  }
  $("sceneSelect").value = app.config.default_scene_id;
  await selectScene(app.config.default_scene_id);
  await poll();
}

initialize().catch((error) => {
  $("message").textContent = `初始化失败：${error.message}`;
});
