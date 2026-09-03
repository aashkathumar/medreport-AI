#!/usr/bin/env bash
# EC2 user-data script: installs and runs the Streamlit frontend on first
# boot. Runs the frontend directly on AWS (Amazon Linux 2023, free-tier
# t2.micro/t3.micro), rather than a third-party host, to stay inside the
# project's own AWS/GCP/Azure cloud-platform brief.
#
# The two placeholders below are substituted automatically by
# deploy/launch_ec2.sh -- this file itself never needs manual editing.
#
# Everything it does is logged to /var/log/medreport-userdata.log, and
# cloud-init's own copy lands in /var/log/cloud-init-output.log, so a
# failed boot can be diagnosed by SSHing in and reading those rather than
# guessing from the outside.

set -euxo pipefail
exec > >(tee -a /var/log/medreport-userdata.log) 2>&1

REPO_URL="REPLACE_WITH_YOUR_GITHUB_REPO_URL"   # e.g. https://github.com/<you>/medreport-AI.git
BACKEND_URL="REPLACE_WITH_LAMBDA_FUNCTION_URL/api/v1"  # from template.yaml's Outputs.FunctionUrl

# BUG FOUND on the first real launch, worth recording: the frontend lives
# only on the feature/developv4.2 branch -- `main` has no frontend/
# directory at all. A plain `git clone` takes the remote's default branch
# (main), so `cd frontend` then failed and set -e aborted the whole boot
# silently, leaving a running instance serving nothing. The branch is
# therefore pinned explicitly here.
REPO_BRANCH="feature/developv4.2"

dnf update -y
dnf install -y git

# Amazon Linux 2023 does not reliably carry python3.12 in its default
# repos (the version available moves with the AMI), so the interpreter is
# resolved at boot rather than hard-coded -- an unavailable package name
# would otherwise fail the whole script at install time.
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if dnf install -y "${candidate}" "${candidate}-pip" 2>/dev/null && command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${candidate}")"
    break
  fi
done
if [ -z "${PYTHON_BIN}" ]; then
  echo "FATAL: no usable python3 interpreter could be installed." >&2
  exit 1
fi
echo "Using interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

cd /home/ec2-user
rm -rf medreport-AI
git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" medreport-AI
chown -R ec2-user:ec2-user medreport-AI
cd medreport-AI/frontend

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

# Run under systemd rather than `nohup ... &`: cloud-init terminates the
# user-data script's process group when it finishes, which can take a
# backgrounded Streamlit with it, and a plain background process would not
# come back after an instance reboot either.
#
# Port 80 directly (the unit runs as root) so the app is reachable at the
# instance's plain public IP with no port number in the URL.
cat > /etc/systemd/system/medreport-frontend.service <<EOF
[Unit]
Description=MedReport AI Streamlit frontend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/ec2-user/medreport-AI/frontend
Environment=MEDREPORT_API_URL=${BACKEND_URL}
ExecStart=${PYTHON_BIN} -m streamlit run main.py --server.port 80 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5
StandardOutput=append:/var/log/medreport-frontend.log
StandardError=append:/var/log/medreport-frontend.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now medreport-frontend.service
systemctl --no-pager status medreport-frontend.service || true

echo "user-data finished at $(date -Is)"
