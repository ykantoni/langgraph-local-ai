#!/usr/bin/env bash
# Manually rolling-restart OpenSearch pods (highest ordinal first).
#
# Use when additionalConfig changed but the operator did not restart nodes,
# often because the cluster is not GREEN or the upgrade reconciler is waiting.
#
#   ./deploy/opensearch/rollout-restart.sh

set -euo pipefail

NAMESPACE="${OPENSEARCH_NAMESPACE:-opensearch-bench}"
CLUSTER_NAME="${OPENSEARCH_CLUSTER_NAME:-opensearch-bench}"
TIMEOUT="${OPENSEARCH_ROLLOUT_TIMEOUT:-900s}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found on PATH." >&2
  exit 1
fi

mapfile -t PODS < <(
  kubectl get pods -n "${NAMESPACE}" \
    -l "opensearch.org/opensearch-cluster=${CLUSTER_NAME}" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.name \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
)

if [[ ${#PODS[@]} -eq 0 ]]; then
  echo "No running pods found for cluster ${CLUSTER_NAME} in ${NAMESPACE}." >&2
  exit 1
fi

echo "Rolling restart (${#PODS[@]} pod(s)), highest ordinal first:"
printf '  %s\n' "${PODS[@]}"

# Restart from highest ordinal to lowest.
for (( idx=${#PODS[@]}-1; idx>=0; idx-- )); do
  POD="${PODS[$idx]}"
  echo
  echo "Deleting pod/${POD} ..."
  kubectl delete pod "${POD}" -n "${NAMESPACE}" --wait=true --timeout="${TIMEOUT}"
  echo "Waiting for pod/${POD} to become Ready ..."
  kubectl wait --for=condition=ready "pod/${POD}" -n "${NAMESPACE}" --timeout="${TIMEOUT}"
done

echo
echo "Rolling restart complete."
kubectl get pods -n "${NAMESPACE}" -l "opensearch.org/opensearch-cluster=${CLUSTER_NAME}"
