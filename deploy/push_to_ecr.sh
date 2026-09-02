#!/usr/bin/env bash
# Builds the AWS Lambda container image (if not already built) and pushes it
# to ECR, ready for template.yaml's ImageUri.
#
# Requires: AWS CLI configured (`aws configure`) under the Free account
# plan, Docker running locally. Run from the project root.
#
# Usage: ./deploy/push_to_ecr.sh

set -euo pipefail

REGION="eu-west-2"
REPO_NAME="medreport-ai-backend-aws"
IMAGE_TAG="latest"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

echo "Account: ${ACCOUNT_ID}"
echo "Target:  ${ECR_URI}:${IMAGE_TAG}"

# Create the repository if it doesn't already exist (idempotent).
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

# Build fresh, so today's code/fixes are always what gets pushed.
docker build -f Dockerfile.aws -t "${REPO_NAME}:${IMAGE_TAG}" .

# Authenticate Docker to this account's ECR, then tag and push.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "Pushed. Put this in template.yaml's ImageUri field:"
echo "  ${ECR_URI}:${IMAGE_TAG}"
