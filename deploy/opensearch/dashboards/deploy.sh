#!/usr/bin/env bash
# Deploy OpenSearch Dashboards via the official opensearch/opensearch-dashboards Helm chart.
# Connects to an existing OpenSearch cluster (default: https://10.1.139.63:9200).
#
#   export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
#   ./deploy/opensearch/dashboards/deploy.sh
#
# Optional:
#   OPENSEARCH_HOST=10.1.139.63
#   OPENSEARCH_PORT=9200
#   OPENSEARCH_USE_SSL=1
#   OPENSEARCH_DASHBOARDS_RELEASE=opensearch-dashboards
#   OPENSEARCH_DASHBOARDS_CHART_VERSION=   # pin chart; omit for newest
#   OPENSEARCH_DASHBOARDS_IMAGE_TAG=3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="${SCRIPT_DIR}/values.yaml"

NAMESPACE="${OPENSEARCH_DASHBOARDS_NAMESPACE:-opensearch-dashboards}"
RELEASE="${OPENSEARCH_DASHBOARDS_RELEASE:-opensearch-dashboards}"
SECRET_NAME="${OPENSEARCH_DASHBOARDS_SECRET:-opensearch-dashboards-credentials}"
HOST="${OPENSEARCH_HOST:-10.1.139.63}"
PORT="${OPENSEARCH_PORT:-9200}"
USER="${OPENSEARCH_DASHBOARDS_USER:-admin}"
IMAGE_TAG="${OPENSEARCH_DASHBOARDS_IMAGE_TAG:-3}"

HELM_REPO="opensearch"
HELM_REPO_URL="https://opensearch-project.github.io/helm-charts/"
CHART="${HELM_REPO}/opensearch-dashboards"

PASSWORD="${OPENSEARCH_DASHBOARDS_PASSWORD:-${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-${OPENSEARCH_PASSWORD:-}}}"

if [[ -z "${PASSWORD}" ]]; then
  cat <<'EOF' >&2
OpenSearch credentials required. Set one of:
  OPENSEARCH_DASHBOARDS_PASSWORD
  OPENSEARCH_INITIAL_ADMIN_PASSWORD
  OPENSEARCH_PASSWORD

Example:
  export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
  ./deploy/opensearch/dashboards/deploy.sh
EOF
  exit 1
fi

if [[ "${OPENSEARCH_USE_SSL:-1}" == "0" ]]; then
  SCHEME="http"
else
  SCHEME="https"
fi

for cmd in kubectl helm; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "${cmd} not found on PATH." >&2
    exit 1
  }
done

resolve_chart_version() {
  if [[ -n "${OPENSEARCH_DASHBOARDS_CHART_VERSION:-}" ]]; then
    echo "${OPENSEARCH_DASHBOARDS_CHART_VERSION}"
    return
  fi
  helm search repo "${CHART}" --versions | awk 'NR==2 { print $2; exit }'
}

echo "Adding Helm repo ${HELM_REPO} ..."
helm repo add "${HELM_REPO}" "${HELM_REPO_URL}" >/dev/null 2>&1 || true
helm repo update "${HELM_REPO}" >/dev/null

CHART_VERSION="$(resolve_chart_version)"
OPENSEARCH_HOSTS="${SCHEME}://${HOST}:${PORT}"

echo "Installing ${CHART} chart version ${CHART_VERSION} ..."
echo "  backend: ${OPENSEARCH_HOSTS} (user: ${USER})"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Chart maps secret key "cookie" -> COOKIE_PASS (32-char security plugin cookie secret).
COOKIE="${OPENSEARCH_DASHBOARDS_COOKIE:-}"
if [[ -z "${COOKIE}" ]]; then
  COOKIE="$(
    kubectl -n "${NAMESPACE}" get secret "${SECRET_NAME}" \
      -o jsonpath='{.data.cookie}' 2>/dev/null | base64 -d 2>/dev/null || true
  )"
fi
if [[ -z "${COOKIE}" ]]; then
  COOKIE="$(openssl rand -hex 16)"
fi

kubectl -n "${NAMESPACE}" create secret generic "${SECRET_NAME}" \
  --from-literal=username="${USER}" \
  --from-literal=password="${PASSWORD}" \
  --from-literal=cookie="${COOKIE}" \
  --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install "${RELEASE}" "${CHART}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --version "${CHART_VERSION}" \
  -f "${VALUES_FILE}" \
  --set "opensearchHosts=${OPENSEARCH_HOSTS}" \
  --set "opensearchAccount.secret=${SECRET_NAME}" \
  --set "image.tag=${IMAGE_TAG}" \
  --wait \
  --timeout 10m

echo
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/instance=${RELEASE}"
echo
kubectl get svc -n "${NAMESPACE}" -l "app.kubernetes.io/instance=${RELEASE}"

cat <<EOF

Open UI via NodePort (5601:<nodePort>) on a node IP, or port-forward:

  kubectl port-forward -n ${NAMESPACE} svc/${RELEASE} 5601:5601
  # http://localhost:5601  — login: ${USER} / <your password>

Tear down:

  helm uninstall ${RELEASE} -n ${NAMESPACE}
  kubectl delete secret ${SECRET_NAME} -n ${NAMESPACE}
  kubectl delete namespace ${NAMESPACE}
EOF
