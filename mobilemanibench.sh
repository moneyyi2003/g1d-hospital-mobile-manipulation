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
        echo "Usage: ./mobilemanibench.sh {isaacsim|smoke|g1-d-smoke|vln|simple-room-vln|hospital-survey|hospital-map|hospital-vln|hospital-demo|hospital-web|hospital-docking|hospital-object-docking|hospital-object-web|agent|doctor|convert-urdf|python} [args...]"
        echo "  isaacsim     Launch the pinned MobileManiBench Isaac Sim GUI environment."
        echo "  smoke        Load one headless MobileManiBench G1/YCB environment."
        echo "  g1-d-smoke   Load and step the converted custom G1_D articulation."
        echo "  vln          Run the deterministic G1_D language-to-point navigation baseline."
        echo "  simple-room-vln  Navigate G1_D to a language goal in SimpleRoom (GUI by default)."
        echo "  hospital-survey  Drive G1_D through the Hospital lobby and record RGB/GIF."
        echo "  hospital-map     Run LingBot alignment/map building and render previews."
        echo "  hospital-vln     Navigate G1_D in Hospital using bootstrap or formal map artifacts."
        echo "  hospital-demo    Open the Hospital GUI with a chase camera and keep it open."
        echo "  hospital-web     Serve the TCP-only Hospital live dashboard (default port 6006)."
        echo "  hospital-docking Build isolated experimental multi-chair docking candidates."
        echo "  hospital-object-docking Run the isolated object-relative precision docking demo."
        echo "  hospital-object-web Serve unified semantic-region/object docking live UI (port 6009)."
        echo "  agent        Plan or execute a task through the existing VLN and future VLA."
        echo "  doctor       Check Python, GPU, robot, USD, and official assets."
        echo "  convert-urdf Convert g1_d_description/g1_d.urdf to USD."
        echo "  python       Run a Python command inside the pinned environment."
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
