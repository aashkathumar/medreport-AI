#!/usr/bin/env bash
# Launches a single free-tier EC2 instance (t3.micro) running the Streamlit
# frontend, using deploy/ec2_userdata.sh. Under the AWS Free account plan
# this draws from the same shared, structurally-cannot-charge credit --
# it cannot silently upgrade to a paid instance type or auto-bill.
#
# Requires: AWS CLI configured, an existing key pair (for SSH access if you
# ever need to debug on the box -- not required for the app to run), and
# deploy/deploy_lambda.sh already run (this script reads its saved
# deploy/.function_url automatically -- no manual editing needed).
#
# Usage: ./deploy/launch_ec2.sh <your-key-pair-name>

set -euo pipefail

REGION="eu-west-2"
KEY_NAME="${1:?Usage: launch_ec2.sh <key-pair-name>}"
INSTANCE_TYPE="t3.micro"  # Free-tier eligible under the Free account plan.
REPO_URL="https://github.com/aashkathumar/medreport-AI.git"

if [ ! -f "deploy/.function_url" ]; then
  echo "No deploy/.function_url found -- run deploy/deploy_lambda.sh first." >&2
  exit 1
fi
BACKEND_URL="$(cat deploy/.function_url)api/v1"

# Substitutes the real repo/backend URLs into a throwaway copy of the
# user-data script, so the checked-in ec2_userdata.sh itself never needs
# manual editing.
TMP_USERDATA="/tmp/medreport-userdata-$$.sh"
sed \
  -e "s|REPLACE_WITH_YOUR_GITHUB_REPO_URL|${REPO_URL}|" \
  -e "s|REPLACE_WITH_LAMBDA_FUNCTION_URL/api/v1|${BACKEND_URL}|" \
  deploy/ec2_userdata.sh > "$TMP_USERDATA"

# Amazon Linux 2023, looked up dynamically so this doesn't go stale.
AMI_ID=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text --region "${REGION}")

# Security group allowing HTTP (the app) and SSH (debugging only).
SG_ID=$(aws ec2 create-security-group --group-name medreport-frontend-sg \
  --description "MedReport AI frontend" --region "${REGION}" --query GroupId --output text \
  2>/dev/null || aws ec2 describe-security-groups --group-names medreport-frontend-sg \
  --region "${REGION}" --query "SecurityGroups[0].GroupId" --output text)

aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --protocol tcp --port 80 --cidr 0.0.0.0/0 --region "${REGION}" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "${REGION}" 2>/dev/null || true

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --key-name "${KEY_NAME}" \
  --security-group-ids "${SG_ID}" \
  --user-data "file://${TMP_USERDATA}" \
  --region "${REGION}" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=medreport-frontend}]' \
  --query "Instances[0].InstanceId" --output text)

rm -f "$TMP_USERDATA"
echo "Launched: ${INSTANCE_ID}"
echo "Waiting for a public IP..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo ""
echo "Instance running at: http://${PUBLIC_IP}"
echo "(Allow ~2-3 min after this for the user-data script to finish installing and starting Streamlit.)"
