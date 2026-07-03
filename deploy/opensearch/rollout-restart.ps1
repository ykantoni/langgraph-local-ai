# Manually rolling-restart OpenSearch pods (highest ordinal first).
#
# Use when additionalConfig changed but the operator did not restart nodes,
# often because the cluster is not GREEN or the upgrade reconciler is waiting.
#
#   .\deploy\opensearch\rollout-restart.ps1

$ErrorActionPreference = "Stop"

$Namespace = if ($env:OPENSEARCH_NAMESPACE) { $env:OPENSEARCH_NAMESPACE } else { "opensearch-bench" }
$ClusterName = if ($env:OPENSEARCH_CLUSTER_NAME) { $env:OPENSEARCH_CLUSTER_NAME } else { "opensearch-bench" }
$Timeout = if ($env:OPENSEARCH_ROLLOUT_TIMEOUT) { $env:OPENSEARCH_ROLLOUT_TIMEOUT } else { "900s" }

$pods = kubectl get pods -n $Namespace `
    -l "opensearch.org/opensearch-cluster=$ClusterName" `
    --field-selector=status.phase=Running `
    --sort-by=.metadata.name `
    -o jsonpath='{range .items[*]}{.metadata.name}{"`n"}{end}' 2>$null

if (-not $pods) {
    throw "No running pods found for cluster $ClusterName in $Namespace."
}

$podList = $pods.Trim().Split("`n") | Where-Object { $_ }
Write-Host "Rolling restart ($($podList.Count) pod(s)), highest ordinal first:"
$podList | ForEach-Object { Write-Host "  $_" }

for ($idx = $podList.Count - 1; $idx -ge 0; $idx--) {
    $pod = $podList[$idx]
    Write-Host ""
    Write-Host "Deleting pod/$pod ..."
    kubectl delete pod $pod -n $Namespace --wait=true --timeout=$Timeout
    Write-Host "Waiting for pod/$pod to become Ready ..."
    kubectl wait --for=condition=ready "pod/$pod" -n $Namespace --timeout=$Timeout
}

Write-Host ""
Write-Host "Rolling restart complete."
kubectl get pods -n $Namespace -l "opensearch.org/opensearch-cluster=$ClusterName"
