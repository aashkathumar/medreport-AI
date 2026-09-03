#!/usr/bin/env bash
# Creates every SSM SecureString parameter template.yaml needs, reading the
# actual values from backend/.env so nothing has to be re-typed by hand.
# Safe to re-run -- overwrites existing parameters rather than failing.
#
# Requires: AWS CLI configured under the Free account plan. Run from the
# project root.
#
# Usage: ./deploy/create_ssm_params.sh

set -euo pipefail

REGION="eu-west-2"
ENV_FILE="backend/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE -- run this from the project root." >&2
  exit 1
fi

# Reads KEY=value out of .env without sourcing it (avoids executing anything
# unexpected the file might contain).
get_env() {
  grep "^$1=" "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true
}

# Maps each SSM parameter name (matching template.yaml's resolve refs) to
# the .env variable that holds it. A blank value is skipped, not set --
# matches provider_has_key()'s existing "unset key is simply unused" logic,
# so an optional provider (e.g. Cerebras, if never configured) doesn't need
# a placeholder parameter just to satisfy the template.
declare -A PARAMS=(
  [groq_api_key]="GROQ_API_KEY"
  [mistral_api_key]="MISTRAL_API_KEY"
  [nvidia_api_key]="NVIDIA_API_KEY"
  [openrouter_api_key]="OPENROUTER_API_KEY"
  [gemini_api_key]="GEMINI_API_KEY"
  [cohere_api_key]="COHERE_API_KEY"
  [cerebras_api_key]="CEREBRAS_API_KEY"
  [cloudflare_api_key]="CLOUDFLARE_API_KEY"
  [cloudflare_account_id]="CLOUDFLARE_ACCOUNT_ID"
)

for param_name in "${!PARAMS[@]}"; do
  env_var="${PARAMS[$param_name]}"
  value=$(get_env "$env_var")
  if [ -z "$value" ]; then
    echo "Skipping /medreport/${param_name} (no ${env_var} set in .env)"
    continue
  fi
  # Plain String, not SecureString -- CloudFormation's ssm-secure dynamic
  # reference is not supported inside Lambda environment variables, so
  # template.yaml resolves these via the plain (non-secure) ssm reference
  # instead. Values are still only injected at deploy time, never committed.
  aws ssm put-parameter \
    --name "/medreport/${param_name}" \
    --type String \
    --value "$value" \
    --overwrite \
    --region "$REGION" >/dev/null
  echo "Set /medreport/${param_name}"
done

echo ""
echo "Done. template.yaml's {{resolve:ssm-secure:...}} references will now resolve."
