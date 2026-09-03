#!/usr/bin/env bash
# Deploys the Lambda function from template.yaml, using the image URI
# push_to_ecr.sh already saved and the SSM parameters create_ssm_params.sh
# already created. Fully automated -- no manual template editing.
#
# Requires: AWS CLI configured, AWS SAM CLI installed (`pip install
# aws-sam-cli` if you don't have it -- it's a standard, free tool, not an
# AWS service, so this has no cost implication). Run from the project root,
# after push_to_ecr.sh and create_ssm_params.sh have both succeeded.
#
# Usage: ./deploy/deploy_lambda.sh

set -euo pipefail

REGION="eu-west-2"
STACK_NAME="medreport-ai-backend"

if ! command -v sam >/dev/null 2>&1; then
  echo "AWS SAM CLI is not installed. Install it with:"
  echo "  pip install aws-sam-cli"
  echo "(This is a local build/deploy tool, not a paid AWS service -- installing it costs nothing.)"
  exit 1
fi

if [ ! -f "deploy/.image_uri" ]; then
  echo "No deploy/.image_uri found -- run deploy/push_to_ecr.sh first." >&2
  exit 1
fi

IMAGE_URI=$(cat deploy/.image_uri)
echo "Deploying image: ${IMAGE_URI}"

# Substitutes the real image URI into a throwaway copy of the template, so
# the checked-in template.yaml itself never needs manual editing.
TMP_TEMPLATE="/tmp/medreport-template-$$.yaml"
sed "s|IMAGE_URI_PLACEHOLDER|${IMAGE_URI}|" template.yaml > "$TMP_TEMPLATE"

sam deploy \
  --template-file "$TMP_TEMPLATE" \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --no-confirm-changeset \
  --resolve-image-repos

rm -f "$TMP_TEMPLATE"

echo ""
echo "Deployed. Fetching the live Function URL..."
FUNCTION_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" \
  --output text)

echo "$FUNCTION_URL" > deploy/.function_url
echo "Backend live at: ${FUNCTION_URL}"
echo "Saved to deploy/.function_url -- deploy/launch_ec2.sh will pick this up automatically for the frontend."
