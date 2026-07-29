#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/root/autodl-tmp"
PROJECT_DIR="${WORKSPACE_DIR}/MobileManiBench"
ENV_DIR="${WORKSPACE_DIR}/envs/mobilemanibench"
LINGBOT_ENV_DIR="${WORKSPACE_DIR}/envs/lingbot-map"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "MobileManiBench environment is missing: ${ENV_DIR}" >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR="${WORKSPACE_DIR}/.cache/pip"
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"

cd "${PROJECT_DIR}"

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${command_name}" in
    isaacsim)
        exec "${ENV_DIR}/bin/isaacsim" isaacsim.exp.full "$@"
        ;;
    smoke)
        exec "${ENV_DIR}/bin/python" unimanip/rsl_ppo/smoke_env.py \
            --task Isaac-G1-Robot-Direct-v0 \
            --config train_g1_robot_open_best_0.yaml \
            --type ycb --group ycb --index 0 --num_envs 1 "$@"
        ;;
    g1-d-smoke)
        exec "${ENV_DIR}/bin/python" scripts/g1_d_smoke.py \
            --usd "${WORKSPACE_DIR}/Assets/g1_d/g1_d.usd" "$@"
        ;;
    vln)
        exec "${ENV_DIR}/bin/python" scripts/g1_d_vln.py \
            --usd "${WORKSPACE_DIR}/Assets/g1_d/g1_d.usd" "$@"
        ;;
    simple-room-vln)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_simple_room_vln.py" \
            --allow-bootstrap "$@"
        ;;
    home-vln)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_simple_room_vln.py" \
            --scene-profile family-home \
            --allow-bootstrap \
            --output-dir "${WORKSPACE_DIR}/outputs/family_home_vln" \
            --command "我困了，请带我到卧室床边" "$@"
        ;;
    home-vln-formal)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_simple_room_vln.py" \
            --scene-profile family-home \
            --output-dir "${WORKSPACE_DIR}/outputs/family_home_vln" \
            --map "${WORKSPACE_DIR}/outputs/family_home_vln/lingbot_map/map.yaml" \
            --places "${WORKSPACE_DIR}/outputs/family_home_vln/places_formal.json" "$@"
        ;;
    home-survey)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_simple_room_vln.py" \
            --scene-profile family-home \
            --survey \
            --output-dir "${WORKSPACE_DIR}/outputs/family_home_vln" "$@"
        ;;
    home-web)
        exec python3 "${WORKSPACE_DIR}/scripts/serve_family_home_dashboard.py" "$@"
        ;;
    warehouse-survey)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_warehouse_vln.py" \
            --survey --allow-bootstrap "$@"
        ;;
    warehouse-vln)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_warehouse_vln.py" \
            --allow-bootstrap "$@"
        ;;
    warehouse-vln-formal)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_warehouse_vln.py" "$@"
        ;;
    warehouse-map)
        if [[ ! -x "${LINGBOT_ENV_DIR}/bin/python" ]]; then
            echo "LingBot-Map environment is missing: ${LINGBOT_ENV_DIR}" >&2
            exit 1
        fi
        export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/lingbot_semantic_nav/src:${WORKSPACE_DIR}/lingbot_semantic_nav/third_party/lingbot-map:${PYTHONPATH:-}"
        exec "${LINGBOT_ENV_DIR}/bin/python" \
            "${WORKSPACE_DIR}/scripts/build_warehouse_map.py" "$@"
        ;;
    warehouse-scene-audit)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/scripts/audit_mobile_scene.py" "$@"
        ;;
    hospital-survey)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_hospital_vln.py" \
            --survey "$@"
        ;;
    hospital-vln)
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_hospital_vln.py" "$@"
        ;;
    hospital-demo)
        if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
            echo "Hospital GUI needs a desktop display (DISPLAY or WAYLAND_DISPLAY)." >&2
            echo "Connect an AutoDL desktop/VNC session, then run this command there." >&2
            exit 3
        fi
        exec "${WORKSPACE_DIR}/isaacsim/python.sh" \
            "${WORKSPACE_DIR}/run_g1d_hospital_vln.py" \
            --viewport-mode chase --keep-open --command "请带我到候诊区" "$@"
        ;;
    hospital-web)
        exec python3 "${WORKSPACE_DIR}/scripts/serve_hospital_dashboard.py" "$@"
        ;;
    hospital-docking)
        exec python3 "${WORKSPACE_DIR}/scripts/build_hospital_docking.py" "$@"
        ;;
    hospital-object-docking)
        exec python3 "${WORKSPACE_DIR}/scripts/run_hospital_object_docking_demo.py" "$@"
        ;;
    hospital-object-web)
        exec python3 "${WORKSPACE_DIR}/scripts/serve_object_docking_dashboard.py" "$@"
        ;;
    agent)
        exec python3 "${WORKSPACE_DIR}/scripts/run_g1d_agent.py" "$@"
        ;;
    g1d-real-nav)
        if [[ ! -f /opt/ros/humble/setup.bash ]]; then
            echo "ROS 2 Humble is missing: /opt/ros/humble/setup.bash" >&2
            exit 1
        fi
        if [[ ! -f "${WORKSPACE_DIR}/lingbot_semantic_nav/ros2_ws/install/setup.bash" ]]; then
            echo "Build the ROS workspace before launching G1-D real navigation." >&2
            exit 1
        fi
        source /opt/ros/humble/setup.bash
        source "${WORKSPACE_DIR}/lingbot_semantic_nav/ros2_ws/install/setup.bash"
        exec ros2 launch lingbot_semantic_nav_ros g1d_real_nav2.launch.py \
            map:="${WORKSPACE_DIR}/outputs/warehouse_vln/lingbot_map/map.yaml" \
            places:="${WORKSPACE_DIR}/outputs/warehouse_vln/places_formal.json" \
            allow_hardware_output:=False "$@"
        ;;
    hospital-map)
        if [[ ! -x "${LINGBOT_ENV_DIR}/bin/python" ]]; then
            echo "LingBot-Map environment is missing: ${LINGBOT_ENV_DIR}" >&2
            exit 1
        fi
        exec "${LINGBOT_ENV_DIR}/bin/python" \
            "${WORKSPACE_DIR}/scripts/build_hospital_map.py" "$@"
        ;;
    doctor)
        exec "${ENV_DIR}/bin/python" scripts/mobilemanibench_doctor.py "$@"
        ;;
    convert-urdf)
        mkdir -p "${WORKSPACE_DIR}/Assets/g1_d"
        chmod -R u+rwX "${WORKSPACE_DIR}/Assets/g1_d"
        exec "${ENV_DIR}/bin/python" scripts/tools/convert_urdf.py \
            "${WORKSPACE_DIR}/g1_d_description/g1_d.urdf" \
            "${WORKSPACE_DIR}/Assets/g1_d/g1_d.usd" \
            --joint-stiffness 100.0 --joint-damping 10.0 --no-instanceable "$@"
        ;;
    python)
        exec "${ENV_DIR}/bin/python" "$@"
        ;;
    help|-h|--help)
        echo "Usage: ./mobilemanibench.sh {isaacsim|smoke|g1-d-smoke|vln|simple-room-vln|home-vln|home-vln-formal|home-survey|home-web|warehouse-survey|warehouse-map|warehouse-vln|warehouse-vln-formal|warehouse-scene-audit|hospital-survey|hospital-map|hospital-vln|hospital-demo|hospital-web|hospital-docking|hospital-object-docking|hospital-object-web|agent|g1d-real-nav|doctor|convert-urdf|python} [args...]"
        echo "  isaacsim     Launch the pinned MobileManiBench Isaac Sim GUI environment."
        echo "  smoke        Load one headless MobileManiBench G1/YCB environment."
        echo "  g1-d-smoke   Load and step the converted custom G1_D articulation."
        echo "  vln          Run the deterministic G1_D language-to-point navigation baseline."
        echo "  simple-room-vln  Navigate G1_D to a language goal in SimpleRoom (GUI by default)."
        echo "  home-vln     Navigate G1-D in the multi-zone family-home bootstrap scene."
        echo "  home-vln-formal Navigate with the future reviewed family-home LingBot map."
        echo "  home-survey  Collect G1-D RGB across bedroom/living/dining/kitchen zones."
        echo "  home-web     Serve the family-home live camera/map dashboard (port 6012)."
        echo "  warehouse-survey Record G1-D RGB in MobileManiBench's multi-shelf Warehouse."
        echo "  warehouse-map Build formal LingBot RGB-only, SAM3, occupancy, and place artifacts."
        echo "  warehouse-vln Navigate G1-D with the explicit collision bootstrap."
        echo "  warehouse-vln-formal Navigate G1-D with the reviewed formal occupancy/place bundle."
        echo "  warehouse-scene-audit Audit a USD scene's bounds, meshes, and colliders."
        echo "  hospital-survey  Drive G1_D through the Hospital lobby and record RGB/GIF."
        echo "  hospital-map     Run LingBot alignment/map building and render previews."
        echo "  hospital-vln     Navigate G1_D in Hospital using bootstrap or formal map artifacts."
        echo "  hospital-demo    Open the Hospital GUI with a chase camera and keep it open."
        echo "  hospital-web     Serve the TCP-only Hospital live dashboard (default port 6006)."
        echo "  hospital-docking Build isolated experimental multi-chair docking candidates."
        echo "  hospital-object-docking Run the isolated object-relative precision docking demo."
        echo "  hospital-object-web Serve unified semantic-region/object docking live UI (port 6009)."
        echo "  agent        Plan or execute a task through the existing VLN and future VLA."
        echo "  g1d-real-nav Launch fail-closed physical G1-D ROS 2/Nav2 (hardware output disabled)."
        echo "  doctor       Check Python, GPU, robot, USD, and official assets."
        echo "  convert-urdf Convert g1_d_description/g1_d.urdf to USD."
        echo "  python       Run a Python command inside the pinned environment."
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
