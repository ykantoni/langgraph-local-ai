#!/usr/bin/env bash
# Deploy OpenSearch for vector-store benchmarks via the official Helm chart.
#
# Requires: kubectl, helm
#
#   export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
#   ./deploy/opensearch/deploy.sh
#
# Optional:
#   OPENSEARCH_CHART_VERSION=3.1.0   # pin chart; omit for newest from repo
#   OPENSEARCH_NAMESPACE=opensearch-bench
#   OPENSEARCH_RELEASE=opensearch-bench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="${SCRIPT_DIR}/values.yaml"

NAMESPACE="${OPENSEARCH_NAMESPACE:-opensearch-bench}"
RELEASE="${OPENSEARCH_RELEASE:-opensearch-bench}"
REPO_NAME="opensearch"
REPO_URL="https://opensearch-project.github.io/helm-charts/"
CHART="${REPO_NAME}/opensearch"

if [[ -z "${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-}" ]]; then
  cat <<'EOF' >&2
OPENSEARCH_INITIAL_ADMIN_PASSWORD is required.

Example:
  export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
  ./deploy/opensearch/deploy.sh

Password rules (OpenSearch 2.12+): at least 8 characters with uppercase,
lowercase, digit, and special character.
EOF
  exit 1
fi

if ! command -v helm >/dev/null 2>&1; then
  echo "helm not found on PATH. Install Helm: https://helm.sh/docs/intro/install/" >&2
  exit 1
fi

resolve_chart_version() {
  if [[ -n "${OPENSEARCH_CHART_VERSION:-}" ]]; then
    echo "${OPENSEARCH_CHART_VERSION}"
    return
  fi
  helm search repo "${CHART}" --versions | awk 'NR==2 { print $2; exit }'
}

echo "Adding Helm repo ${REPO_NAME} ..."
helm repo add "${REPO_NAME}" "${REPO_URL}" >/dev/null 2>&1 || true
helm repo update "${REPO_NAME}" >/dev/null

CHART_VERSION="$(resolve_chart_version)"
echo "Using chart ${CHART} version ${CHART_VERSION}"

PASSWORD_VALUES="$(mktemp)"
trap 'rm -f "${PASSWORD_VALUES}"' EXIT

escape_yaml_double() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

PWD_ESCAPED="$(escape_yaml_double "${OPENSEARCH_INITIAL_ADMIN_PASSWORD}")"
cat > "${PASSWORD_VALUES}" <<EOF
extraEnvs:
  - name: OPENSEARCH_INITIAL_ADMIN_PASSWORD
    value: "${PWD_ESCAPED}"
EOF

helm upgrade --install "${RELEASE}" "${CHART}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --version "${CHART_VERSION}" \
  -f "${VALUES_FILE}" \
  -f "${PASSWORD_VALUES}" \
  --wait \
  --timeout 15m

echo
echo "Pods:"
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/instance=${RELEASE}"

echo
echo "Service (note 9200:<nodePort>):"
kubectl get svc -n "${NAMESPACE}" opensearch-bench

cat <<EOF

OpenSearch is deploying. When the pod is Ready, run benchmarks from your host:

  export OPENSEARCH_HOST='<NODE_INTERNAL_IP>'
  export OPENSEARCH_PORT='<NODE_PORT>'
  export OPENSEARCH_USER='admin'
  export OPENSEARCH_PASSWORD="\${OPENSEARCH_INITIAL_ADMIN_PASSWORD}"
  export OPENSEARCH_USE_SSL=1
  python benchmark.py

Or port-forward (matches benchmark_opensearch defaults):

  kubectl port-forward -n ${NAMESPACE} svc/opensearch-bench 9200:9200
  export OPENSEARCH_KUBECTL_PORT_FORWARD=1
  export OPENSEARCH_USER='admin'
  export OPENSEARCH_PASSWORD="\${OPENSEARCH_INITIAL_ADMIN_PASSWORD}"
  export OPENSEARCH_USE_SSL=1
  python benchmark.py

Uninstall:
  helm uninstall ${RELEASE} -n ${NAMESPACE}
EOF
