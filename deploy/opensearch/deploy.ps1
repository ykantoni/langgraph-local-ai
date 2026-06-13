# Deploy OpenSearch for vector-store benchmarks via the official Helm chart.
#
# Requires: kubectl, helm
#
#   $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "Bench1pass!"
#   .\deploy\opensearch\deploy.ps1
#
# Optional:
#   $env:OPENSEARCH_CHART_VERSION = "3.1.0"   # pin chart; omit for newest from repo
#   $env:OPENSEARCH_NAMESPACE = "opensearch-bench"
#   $env:OPENSEARCH_RELEASE = "opensearch-bench"

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ValuesFile = Join-Path $ScriptDir "values.yaml"

$Namespace = if ($env:OPENSEARCH_NAMESPACE) { $env:OPENSEARCH_NAMESPACE } else { "opensearch-bench" }
$Release = if ($env:OPENSEARCH_RELEASE) { $env:OPENSEARCH_RELEASE } else { "opensearch-bench" }
$RepoName = "opensearch"
$RepoUrl = "https://opensearch-project.github.io/helm-charts/"
$Chart = "$RepoName/opensearch"

if (-not $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD) {
    throw @"
OPENSEARCH_INITIAL_ADMIN_PASSWORD is required.

Example:
  `$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = 'Bench1pass!'
  .\deploy\opensearch\deploy.ps1

Password rules (OpenSearch 2.12+): at least 8 characters with uppercase,
lowercase, digit, and special character.
"@
}

function Get-HelmExe {
    $cmd = Get-Command helm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "helm not found on PATH. Install Helm: https://helm.sh/docs/intro/install/"
}

function Get-LatestChartVersion {
    param([string]$HelmExe)
    $json = & $HelmExe search repo $Chart --versions -o json 2>$null | ConvertFrom-Json
    if (-not $json -or $json.Count -eq 0) {
        throw "Could not resolve chart version. Run: helm repo add $RepoName $RepoUrl ; helm repo update"
    }
    return $json[0].version
}

$Helm = Get-HelmExe

Write-Host "Adding Helm repo $RepoName ..."
& $Helm repo add $RepoName $RepoUrl 2>$null | Out-Null
& $Helm repo update $RepoName | Out-Null

$ChartVersion = $env:OPENSEARCH_CHART_VERSION
if (-not $ChartVersion) {
    $ChartVersion = Get-LatestChartVersion -HelmExe $Helm
}
Write-Host "Using chart $Chart version $ChartVersion"

$ExtraEnvsJson = @(
    @{
        name  = "OPENSEARCH_INITIAL_ADMIN_PASSWORD"
        value = $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
    }
) | ConvertTo-Json -Compress

$HelmArgs = @(
    "upgrade", "--install", $Release, $Chart,
    "--namespace", $Namespace,
    "--create-namespace",
    "--version", $ChartVersion,
    "-f", $ValuesFile,
    "--set-json", "extraEnvs=$ExtraEnvsJson",
    "--wait",
    "--timeout", "15m"
)

Write-Host "Installing OpenSearch release '$Release' in namespace '$Namespace' ..."
& $Helm @HelmArgs

Write-Host ""
Write-Host "Pods:"
kubectl get pods -n $Namespace -l "app.kubernetes.io/instance=$Release"

Write-Host ""
Write-Host "Service (note 9200:<nodePort>):"
kubectl get svc -n $Namespace opensearch-bench

Write-Host ""
Write-Host @"
OpenSearch is deploying. When the pod is Ready, run benchmarks from your host:

  `$env:OPENSEARCH_HOST = '<NODE_INTERNAL_IP>'
  `$env:OPENSEARCH_PORT = '<NODE_PORT>'
  `$env:OPENSEARCH_USER = 'admin'
  `$env:OPENSEARCH_PASSWORD = `$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
  `$env:OPENSEARCH_USE_SSL = '1'
  python benchmark.py

Or port-forward (matches benchmark_opensearch defaults):

  kubectl port-forward -n $Namespace svc/opensearch-bench 9200:9200
  `$env:OPENSEARCH_KUBECTL_PORT_FORWARD = '1'
  `$env:OPENSEARCH_USER = 'admin'
  `$env:OPENSEARCH_PASSWORD = `$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
  `$env:OPENSEARCH_USE_SSL = '1'
  python benchmark.py

Uninstall:
  helm uninstall $Release -n $Namespace
"@
