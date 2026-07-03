#!/usr/bin/env bash
# Shared OpenSearch Kubernetes Operator install (Helm).
# Source from deploy scripts after setting OPERATOR_* variables.

install_opensearch_operator() {
  if [[ -n "${OPENSEARCH_SKIP_OPERATOR:-}" ]]; then
    echo "Skipping operator install (OPENSEARCH_SKIP_OPERATOR is set)."
    return
  fi

  echo "Adding Helm repo ${OPERATOR_REPO} ..."
  helm repo add "${OPERATOR_REPO}" "${OPERATOR_REPO_URL}" >/dev/null 2>&1 || true
  helm repo update "${OPERATOR_REPO}" >/dev/null

  local chart_version
  if [[ -n "${OPENSEARCH_OPERATOR_VERSION:-}" ]]; then
    chart_version="${OPENSEARCH_OPERATOR_VERSION}"
  else
    chart_version="$(helm search repo "${OPERATOR_CHART}" --versions | awk 'NR==2 { print $2; exit }')"
  fi
  echo "Installing operator ${OPERATOR_CHART} chart version ${chart_version} ..."

  local webhook_args=()
  if [[ "${OPENSEARCH_OPERATOR_WEBHOOKS:-}" == "1" ]]; then
    if kubectl get crd certificates.cert-manager.io >/dev/null 2>&1; then
      echo "Enabling operator admission webhooks (cert-manager detected)." >&2
      webhook_args=(--set webhook.enabled=true --set webhook.certManager.enabled=true)
    else
      echo "OPENSEARCH_OPERATOR_WEBHOOKS=1 but cert-manager is not installed." >&2
      exit 1
    fi
  elif [[ "${OPENSEARCH_OPERATOR_WEBHOOKS:-}" == "0" ]] || ! kubectl get crd certificates.cert-manager.io >/dev/null 2>&1; then
    echo "Disabling operator admission webhooks (cert-manager not required)." >&2
    webhook_args=(--set webhook.enabled=false)
  else
    echo "cert-manager detected; enabling operator admission webhooks." >&2
    webhook_args=(--set webhook.enabled=true --set webhook.certManager.enabled=true)
  fi

  helm upgrade --install "${OPERATOR_RELEASE}" "${OPERATOR_CHART}" \
    --namespace "${OPERATOR_NS}" \
    --create-namespace \
    --version "${chart_version}" \
    -f "${OPERATOR_VALUES}" \
    "${webhook_args[@]}" \
    --wait \
    --timeout 10m

  echo "Waiting for operator controller ..."
  kubectl wait --for=condition=available deployment \
    -l "app.kubernetes.io/name=opensearch-operator" \
    -n "${OPERATOR_NS}" \
    --timeout=300s
}
