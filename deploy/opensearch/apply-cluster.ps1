# Render cluster.yaml (substitute __OPENSEARCH_VERSION__) and apply to the cluster.
#
# Use this for configuration changes (e.g. CORS in additionalConfig). Do not apply
# cluster.yaml directly — it contains the __OPENSEARCH_VERSION__ placeholder.
#
#   $env:OPENSEARCH_VERSION = "3"   # optional; defaults to the running cluster version
#   .\deploy\opensearch\apply-cluster.ps1
#
# Force a manual rolling pod restart if the operator does not restart nodes:
#
#   $env:OPENSEARCH_ROLLOUT_RESTART = "1"
#   .\deploy\opensearch\apply-cluster.ps1
#   .\deploy\opensearch\rollout-restart.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ClusterTemplate = Join-Path $ScriptDir "cluster.yaml"

$Namespace = if ($env:OPENSEARCH_NAMESPACE) { $env:OPENSEARCH_NAMESPACE } else { "opensearch-bench" }
$ClusterName = if ($env:OPENSEARCH_CLUSTER_NAME) { $env:OPENSEARCH_CLUSTER_NAME } else { "opensearch-bench" }
$OpenSearchVersion = $env:OPENSEARCH_VERSION

if (-not $OpenSearchVersion) {
    $OpenSearchVersion = kubectl get opensearchcluster $ClusterName -n $Namespace `
        -o jsonpath='{.spec.general.version}' 2>$null
    if (-not $OpenSearchVersion -or $OpenSearchVersion -eq "__OPENSEARCH_VERSION__") {
        $OpenSearchVersion = "3"
    }
}

$rendered = Join-Path $env:TEMP "opensearch-bench-cluster.yaml"
(Get-Content $ClusterTemplate -Raw).Replace("__OPENSEARCH_VERSION__", $OpenSearchVersion) |
    Set-Content -Path $rendered -Encoding utf8

Write-Host "Applying OpenSearchCluster $ClusterName (OpenSearch $OpenSearchVersion) ..."
kubectl apply -f $rendered

Write-Host ""
Write-Host "OpenSearchCluster status:"
kubectl get opensearchcluster $ClusterName -n $Namespace 2>$null

Write-Host ""
Write-Host "Pods:"
kubectl get pods -n $Namespace -l "opensearch.org/opensearch-cluster=$ClusterName" 2>$null

Write-Host ""
Write-Host @"
The operator should perform a rolling restart when additionalConfig changes.
If pods are unchanged after a few minutes, the cluster may not be GREEN yet.

Check operator logs:
  kubectl logs -n opensearch-operator-system `
    -l app.kubernetes.io/name=opensearch-operator --tail=100

Force a rolling restart:
  .\deploy\opensearch\rollout-restart.ps1
"@

if ($env:OPENSEARCH_ROLLOUT_RESTART -eq "1") {
    Write-Host ""
    & (Join-Path $ScriptDir "rollout-restart.ps1")
}
