#!/usr/bin/env bash
# Deploy OpenSearch for vector-store benchmarks via the OpenSearch Kubernetes Operator.
#
# Requires: kubectl, helm
#
#   export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
#   ./deploy/opensearch/deploy.sh
#
# Optional:
#   OPENSEARCH_VERSION=3              # OpenSearch image tag (default: 3 = latest 3.x)
#   OPENSEARCH_OPERATOR_VERSION=      # pin operator chart; omit for newest from repo
#   OPENSEARCH_NAMESPACE=opensearch-bench
#   OPENSEARCH_OPERATOR_NAMESPACE=opensearch-operator-system
#   OPENSEARCH_OPERATOR_RELEASE=opensearch-operator
#   OPENSEARCH_SKIP_OPERATOR=1        # skip operator install if already cluster-wide

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_TEMPLATE="${SCRIPT_DIR}/cluster.yaml"
OPERATOR_VALUES="${SCRIPT_DIR}/operator-values.yaml"

NAMESPACE="${OPENSEARCH_NAMESPACE:-opensearch-bench}"
OPERATOR_NS="${OPENSEARCH_OPERATOR_NAMESPACE:-opensearch-operator-system}"
OPERATOR_RELEASE="${OPENSEARCH_OPERATOR_RELEASE:-opensearch-operator}"
CLUSTER_NAME="${OPENSEARCH_CLUSTER_NAME:-opensearch-bench}"
ADMIN_SECRET="${OPENSEARCH_ADMIN_SECRET:-opensearch-bench-admin-credentials}"
OPENSEARCH_VERSION="${OPENSEARCH_VERSION:-3}"

OPERATOR_REPO="opensearch-operator"
OPERATOR_REPO_URL="https://opensearch-project.github.io/opensearch-k8s-operator/"
OPERATOR_CHART="${OPERATOR_REPO}/opensearch-operator"

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

for cmd in kubectl helm; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} not found on PATH." >&2
    exit 1
  fi
done

# shellcheck source=install-operator.sh
source "${SCRIPT_DIR}/install-operator.sh"

render_cluster_manifest() {
  sed "s/__OPENSEARCH_VERSION__/${OPENSEARCH_VERSION}/g" "${CLUSTER_TEMPLATE}"
}

apply_cluster() {
  echo "Creating namespace ${NAMESPACE} ..."
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

  echo "Creating admin credentials secret ${ADMIN_SECRET} ..."
  kubectl -n "${NAMESPACE}" create secret generic "${ADMIN_SECRET}" \
    --from-literal=username=admin \
    --from-literal=password="${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "Applying OpenSearchCluster ${CLUSTER_NAME} (OpenSearch ${OPENSEARCH_VERSION}) ..."
  render_cluster_manifest | kubectl apply -f -

  echo "Waiting for OpenSearch pods ..."
  kubectl wait --for=condition=ready pod \
    -n "${NAMESPACE}" \
    -l "opensearch.org/opensearch-cluster=${CLUSTER_NAME}" \
    --timeout=900s

  echo "Patching service ${CLUSTER_NAME} to NodePort ..."
  kubectl patch svc "${CLUSTER_NAME}" -n "${NAMESPACE}" --type=merge \
    -p '{"spec":{"type":"NodePort"}}'
}

install_opensearch_operator
apply_cluster

echo
echo "Pods:"
kubectl get pods -n "${NAMESPACE}" -l "opensearch.org/opensearch-cluster=${CLUSTER_NAME}"

echo
echo "Service (note http:<nodePort>):"
kubectl get svc -n "${NAMESPACE}" "${CLUSTER_NAME}"

cat <<EOF

OpenSearch cluster is ready. Run benchmarks from your host:

  export OPENSEARCH_HOST='<NODE_INTERNAL_IP>'
  export OPENSEARCH_PORT='<NODE_PORT>'
  export OPENSEARCH_USER='admin'
  export OPENSEARCH_PASSWORD="\${OPENSEARCH_INITIAL_ADMIN_PASSWORD}"
  export OPENSEARCH_USE_SSL=1
  python benchmark.py

Or port-forward (matches benchmark_opensearch defaults):

  kubectl port-forward -n ${NAMESPACE} svc/${CLUSTER_NAME} 9200:9200
  export OPENSEARCH_KUBECTL_PORT_FORWARD=1
  export OPENSEARCH_USER='admin'
  export OPENSEARCH_PASSWORD="\${OPENSEARCH_INITIAL_ADMIN_PASSWORD}"
  export OPENSEARCH_USE_SSL=1
  python benchmark.py

Tear down cluster (operator remains installed):

  kubectl delete opensearchcluster ${CLUSTER_NAME} -n ${NAMESPACE}
  kubectl delete secret ${ADMIN_SECRET} -n ${NAMESPACE}
  kubectl delete pvc -l opensearch.org/opensearch-cluster=${CLUSTER_NAME} -n ${NAMESPACE}

Uninstall operator:

  helm uninstall ${OPERATOR_RELEASE} -n ${OPERATOR_NS}
EOF
