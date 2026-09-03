#!/usr/bin/env bash
# Provisions the Streamlit frontend onto an already-running EC2 instance by
# copying the code over SSH, rather than having the instance clone it from
# GitHub.
#
# WHY NOT git clone (the original ec2_userdata.sh approach): the project
# repository is private, and a bare EC2 instance holds no GitHub
# credentials, so the clone failed at first boot with "could not read
# Username for 'https://github.com'". The obvious workarounds are worse
# than this one: a token or deploy key passed through user-data would be
# readable by anyone who can reach the instance metadata, and it is also
# echoed into the console log, while making the repository public would
# expose the dissertation's source before marking. Copying the code over
# the SSH key pair that already exists avoids both problems.
#
# Requires: the instance running, its key pair .pem locally, and
# deploy/deploy_lambda.sh already run (reads deploy/.function_url).
#
# Usage: ./deploy/provision_frontend.sh <public-ip> <path-to-key.pem>

set -euo pipefail

HOST="${1:?Usage: provision_frontend.sh <public-ip> <path-to-key.pem>}"
KEY="${2:?Usage: provision_frontend.sh <public-ip> <path-to-key.pem>}"
SSH_USER="ec2-user"

if [ ! -f "deploy/.function_url" ]; then
  echo "No deploy/.function_url found -- run deploy/deploy_lambda.sh first." >&2
  exit 1
fi
BACKEND_URL="$(cat deploy/.function_url)api/v1"

SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

echo "Waiting for SSH on ${HOST}..."
for _ in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" true 2>/dev/null; then break; fi
  sleep 5
done

echo "Copying frontend..."
# sudo rm, not plain rm: the running systemd service writes __pycache__
# as root, so a re-run by the SSH user alone fails with "Permission
# denied" trying to delete those files.
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" 'sudo rm -rf ~/frontend && mkdir -p ~/frontend'
# tar piped over ssh rather than rsync: Amazon Linux 2023 does not ship
# rsync, and rsync must exist on BOTH ends, so the first attempt failed
# with "rsync: command not found" on the remote side. tar and ssh are
# present everywhere and still support the excludes, which keep local
# build artefacts (and any stray .env) off the box.
# COPYFILE_DISABLE stops macOS tar adding ._* AppleDouble metadata files.
COPYFILE_DISABLE=1 tar czf - \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.env*' \
  -C frontend . \
  | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" 'tar xzf - -C ~/frontend'

echo "Installing and starting the service..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" \
  "BACKEND_URL='${BACKEND_URL}' bash -s" <<'REMOTE'
set -euxo pipefail

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$candidate")"
    break
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  sudo dnf install -y python3.12 python3.12-pip
  PYTHON_BIN="$(command -v python3.12)"
fi

sudo "$PYTHON_BIN" -m pip install --upgrade pip
sudo "$PYTHON_BIN" -m pip install -r "$HOME/frontend/requirements.txt"

# systemd rather than a backgrounded process: survives this SSH session
# closing, and restarts the app automatically if the instance reboots.
# Port 80 directly (unit runs as root) so no port number is needed in the URL.
sudo tee /etc/systemd/system/medreport-frontend.service >/dev/null <<EOF
[Unit]
Description=MedReport AI Streamlit frontend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$HOME/frontend
Environment=MEDREPORT_API_URL=${BACKEND_URL}
ExecStart=$PYTHON_BIN -m streamlit run main.py --server.port 80 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5
StandardOutput=append:/var/log/medreport-frontend.log
StandardError=append:/var/log/medreport-frontend.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now medreport-frontend.service
sleep 5
sudo systemctl --no-pager --lines=20 status medreport-frontend.service || true
REMOTE

echo ""
echo "Frontend provisioned. It should be live at: http://${HOST}"
