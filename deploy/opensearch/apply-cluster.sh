#!/usr/bin/env bash
# Render cluster.yaml (substitute __OPENSEARCH_VERSION__) and apply to the cluster.
#
# Use this for configuration changes (e.g. CORS in additionalConfig). Do not apply
# cluster.yaml directly — it contains the __OPENSEARCH_VERSION__ placeholder.
#
#   export OPENSEARCH_VERSION=3   # optional; defaults to the running cluster version
#   ./deploy/opensearch/apply-cluster.sh
#
# Force a manual rolling pod restart if the operator does not restart nodes:
#
#   OPENSEARCH_ROLLOUT_RESTART=1 ./deploy/opensearch/apply-cluster.sh
#   ./deploy/opensearch/rollout-restart.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_TEMPLATE="${SCRIPT_DIR}/cluster.yaml"

NAMESPACE="${OPENSEARCH_NAMESPACE:-opensearch-bench}"
CLUSTER_NAME="${OPENSEARCH_CLUSTER_NAME:-opensearch-bench}"
OPENSEARCH_VERSION="${OPENSEARCH_VERSION:-}"

if [[ -z "${OPENSEARCH_VERSION}" ]]; then
  OPENSEARCH_VERSION="$(
    kubectl get opensearchcluster "${CLUSTER_NAME}" -n "${NAMESPACE}" \
      -o jsonpath='{.spec.general.version}' 2>/dev/null || true
  )"
  if [[ -z "${OPENSEARCH_VERSION}" || "${OPENSEARCH_VERSION}" == "__OPENSEARCH_VERSION__" ]]; then
    OPENSEARCH_VERSION="3"
  fi
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found on PATH." >&2
  exit 1
fi

echo "Applying OpenSearchCluster ${CLUSTER_NAME} (OpenSearch ${OPENSEARCH_VERSION}) ..."
sed "s/__OPENSEARCH_VERSION__/${OPENSEARCH_VERSION}/g" "${CLUSTER_TEMPLATE}" | kubectl apply -f -

echo
echo "OpenSearchCluster status:"
kubectl get opensearchcluster "${CLUSTER_NAME}" -n "${NAMESPACE}" 2>/dev/null || true

echo
echo "Pods:"
kubectl get pods -n "${NAMESPACE}" -l "opensearch.org/opensearch-cluster=${CLUSTER_NAME}" 2>/dev/null || true

echo
cat <<EOF
The operator should perform a rolling restart when additionalConfig changes.
If pods are unchanged after a few minutes, the cluster may not be GREEN yet.

Check operator logs:
  kubectl logs -n ${OPENSEARCH_OPERATOR_NAMESPACE:-opensearch-operator-system} \\
    -l app.kubernetes.io/name=opensearch-operator --tail=100

Force a rolling restart:
  ./deploy/opensearch/rollout-restart.sh
EOF

if [[ "${OPENSEARCH_ROLLOUT_RESTART:-}" == "1" ]]; then
  echo
  exec "${SCRIPT_DIR}/rollout-restart.sh"
fi
