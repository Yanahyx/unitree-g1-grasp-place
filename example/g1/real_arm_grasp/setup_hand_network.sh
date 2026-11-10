#!/usr/bin/env bash
set -euo pipefail

IFACE="eth1"
ROBOT_IP_CIDR="192.168.124.164/24"
HAND_IP="192.168.124.210"
HAND_PORT="6000"
RUN_PROBE=1
SUDO_PASSWORD="${SUDO_PASSWORD:-}"

usage() {
  cat <<'EOF'
Configure the temporary robot-side network address for the RH56E2 hand.

Defaults:
  iface           eth1
  robot ip/cidr   192.168.124.164/24
  hand endpoint   192.168.124.210:6000

Usage:
  bash example/g1/real_arm_grasp/setup_hand_network.sh [options]

Options:
  --iface NAME             Network interface connected to the hand subnet.
  --robot-ip-cidr CIDR     Temporary robot-side address to add.
  --hand-ip IP             RH56E2 hand IP.
  --hand-port PORT         RH56E2 hand Modbus TCP port.
  --password PASS          Sudo password, if passwordless sudo is unavailable.
  --no-probe               Only configure IP and TCP check; skip Modbus probe.
  -h, --help               Show this help.

Examples:
  bash example/g1/real_arm_grasp/setup_hand_network.sh
  read -rsp "Sudo password: " SUDO_PASSWORD; echo
  export SUDO_PASSWORD
  bash example/g1/real_arm_grasp/setup_hand_network.sh
  unset SUDO_PASSWORD
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface)
      IFACE="$2"
      shift 2
      ;;
    --robot-ip-cidr)
      ROBOT_IP_CIDR="$2"
      shift 2
      ;;
    --hand-ip)
      HAND_IP="$2"
      shift 2
      ;;
    --hand-port)
      HAND_PORT="$2"
      shift 2
      ;;
    --password)
      SUDO_PASSWORD="$2"
      shift 2
      ;;
    --no-probe)
      RUN_PROBE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

sudo_run() {
  if [[ -n "$SUDO_PASSWORD" ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "hand_network_iface: $IFACE"
echo "robot_temp_ip: $ROBOT_IP_CIDR"
echo "hand_endpoint: ${HAND_IP}:${HAND_PORT}"

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  echo "Interface not found: $IFACE" >&2
  exit 1
fi

if ip -4 addr show dev "$IFACE" | grep -q " ${ROBOT_IP_CIDR%/*}/"; then
  echo "temp_ip_status: already configured"
else
  echo "temp_ip_status: adding"
  sudo_run ip addr add "$ROBOT_IP_CIDR" dev "$IFACE"
fi

echo "route_to_hand:"
ip route get "$HAND_IP" || true

echo "tcp_check: ${HAND_IP}:${HAND_PORT}"
python3 -c 'import socket,sys; ip=sys.argv[1]; port=int(sys.argv[2]); s=socket.create_connection((ip, port), timeout=2.0); s.close(); print("TCP connected")' "$HAND_IP" "$HAND_PORT"

if [[ "$RUN_PROBE" -eq 1 ]]; then
  cd "$REPO_ROOT"
  echo "modbus_probe:"
  python3 example/g1/high_level/rh56e2_modbus_probe.py \
    --ip "$HAND_IP" \
    --port "$HAND_PORT"
fi
