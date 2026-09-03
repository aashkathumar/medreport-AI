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

# Authenticate Docker to this account's ECR before building, since the build
# below pushes directly rather than loading locally first.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Build fresh, so today's code/fixes are always what gets pushed.
#
# BUG FOUND during the first real deploy, worth recording: a plain
# `docker build` on current Docker Desktop produces an OCI image index
# (application/vnd.oci.image.index.v1+json) with provenance attestations
# attached. Lambda rejects that outright -- "The image manifest, config or
# layer media type for the source image ... is not supported" -- because it
# only accepts Docker Image Manifest V2 Schema 2. Hence the explicit
# buildx invocation: --provenance/--sbom off to stop the attestation
# manifest that forces an index, and oci-mediatypes=false to emit Docker
# media types. --platform is pinned to arm64 to match both the Apple
# Silicon build host (no emulation, so the build stays fast) and
# template.yaml's Architectures setting.
docker buildx build \
  --platform linux/arm64 \
  --provenance=false \
  --sbom=false \
  --output type=image,oci-mediatypes=false,push=true \
  -f Dockerfile.aws \
  -t "${ECR_URI}:${IMAGE_TAG}" \
  .

# Written so deploy_lambda.sh can pick it up automatically -- no manual
# copy-paste of the URI into template.yaml required.
mkdir -p deploy
echo "${ECR_URI}:${IMAGE_TAG}" > deploy/.image_uri

echo ""
echo "Pushed: ${ECR_URI}:${IMAGE_TAG}"
echo "Saved to deploy/.image_uri -- run deploy/deploy_lambda.sh next."
