#!/usr/bin/env bash
# EC2 user-data script: installs and runs the Streamlit frontend on first
# boot. Runs the frontend directly on AWS (Amazon Linux 2023, free-tier
# t2.micro/t3.micro), rather than a third-party host, to stay inside the
# project's own AWS/GCP/Azure cloud-platform brief.
#
# Fill in the two placeholders below, then pass this file as --user-data
# to `aws ec2 run-instances` (see deploy/launch_ec2.sh).

set -euo pipefail

REPO_URL="REPLACE_WITH_YOUR_GITHUB_REPO_URL"   # e.g. https://github.com/<you>/medreport-AI.git
BACKEND_URL="REPLACE_WITH_LAMBDA_FUNCTION_URL/api/v1"  # from template.yaml's Outputs.FunctionUrl

dnf update -y
dnf install -y git python3.12 python3.12-pip

cd /home/ec2-user
git clone "${REPO_URL}" medreport-AI
cd medreport-AI/frontend

python3.12 -m pip install -r requirements.txt

# Runs on port 80 directly (root, via user-data) so the frontend is reachable
# at the instance's plain public IP/DNS with no port number needed.
export MEDREPORT_API_URL="${BACKEND_URL}"
nohup python3.12 -m streamlit run main.py \
  --server.port 80 --server.address 0.0.0.0 --server.headless true \
  > /var/log/medreport-frontend.log 2>&1 &
