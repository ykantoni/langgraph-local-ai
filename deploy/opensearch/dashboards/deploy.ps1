# Deploy OpenSearch Dashboards via the official opensearch/opensearch-dashboards Helm chart.
#
#   $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "Bench1pass!"
#   .\deploy\opensearch\dashboards\deploy.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ValuesFile = Join-Path $ScriptDir "values.yaml"

$Namespace = if ($env:OPENSEARCH_DASHBOARDS_NAMESPACE) { $env:OPENSEARCH_DASHBOARDS_NAMESPACE } else { "opensearch-dashboards" }
$Release = if ($env:OPENSEARCH_DASHBOARDS_RELEASE) { $env:OPENSEARCH_DASHBOARDS_RELEASE } else { "opensearch-dashboards" }
$SecretName = if ($env:OPENSEARCH_DASHBOARDS_SECRET) { $env:OPENSEARCH_DASHBOARDS_SECRET } else { "opensearch-dashboards-credentials" }
$OpenSearchHost = if ($env:OPENSEARCH_HOST) { $env:OPENSEARCH_HOST } else { "10.1.139.63" }
$Port = if ($env:OPENSEARCH_PORT) { $env:OPENSEARCH_PORT } else { "9200" }
$User = if ($env:OPENSEARCH_DASHBOARDS_USER) { $env:OPENSEARCH_DASHBOARDS_USER } else { "admin" }
$ImageTag = if ($env:OPENSEARCH_DASHBOARDS_IMAGE_TAG) { $env:OPENSEARCH_DASHBOARDS_IMAGE_TAG } else { "3" }

$HelmRepo = "opensearch"
$HelmRepoUrl = "https://opensearch-project.github.io/helm-charts/"
$Chart = "$HelmRepo/opensearch-dashboards"

$Password = $env:OPENSEARCH_DASHBOARDS_PASSWORD
if (-not $Password) { $Password = $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD }
if (-not $Password) { $Password = $env:OPENSEARCH_PASSWORD }

if (-not $Password) {
    throw @"
OpenSearch credentials required. Set OPENSEARCH_DASHBOARDS_PASSWORD or OPENSEARCH_INITIAL_ADMIN_PASSWORD.
"@
}

$Scheme = if ($env:OPENSEARCH_USE_SSL -eq "0") { "http" } else { "https" }
$OpenSearchHosts = "${Scheme}://${OpenSearchHost}:${Port}"

function Get-Exe([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "$Name not found on PATH."
}

function Get-LatestChartVersion([string]$HelmExe) {
    $json = & $HelmExe search repo $Chart --versions -o json | ConvertFrom-Json
    if (-not $json -or $json.Count -eq 0) {
        throw "Could not resolve chart version from $HelmRepoUrl"
    }
    return $json[0].version
}

$Helm = Get-Exe "helm"
Get-Exe "kubectl" | Out-Null

& $Helm repo add $HelmRepo $HelmRepoUrl 2>$null | Out-Null
& $Helm repo update $HelmRepo | Out-Null

$ChartVersion = $env:OPENSEARCH_DASHBOARDS_CHART_VERSION
if (-not $ChartVersion) {
    $ChartVersion = Get-LatestChartVersion -HelmExe $Helm
}

Write-Host "Installing $Chart chart version $ChartVersion ..."
Write-Host "  backend: $OpenSearchHosts (user: $User)"

kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -

# Chart maps secret key "cookie" -> COOKIE_PASS (32-char security plugin cookie secret).
$Cookie = $env:OPENSEARCH_DASHBOARDS_COOKIE
if (-not $Cookie) {
    $existingCookie = kubectl -n $Namespace get secret $SecretName -o jsonpath='{.data.cookie}' 2>$null
    if ($existingCookie) {
        $Cookie = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($existingCookie))
    }
}
if (-not $Cookie) {
    $bytes = New-Object byte[] 16
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $Cookie = -join ($bytes | ForEach-Object { '{0:x2}' -f $_ })
}

kubectl -n $Namespace create secret generic $SecretName `
    --from-literal=username=$User `
    --from-literal=password=$Password `
    --from-literal=cookie=$Cookie `
    --dry-run=client -o yaml | kubectl apply -f -

& $Helm upgrade --install $Release $Chart `
    --namespace $Namespace `
    --create-namespace `
    --version $ChartVersion `
    -f $ValuesFile `
    --set "opensearchHosts=$OpenSearchHosts" `
    --set "opensearchAccount.secret=$SecretName" `
    --set "image.tag=$ImageTag" `
    --wait `
    --timeout 10m

Write-Host ""
kubectl get pods -n $Namespace -l "app.kubernetes.io/instance=$Release"
Write-Host ""
kubectl get svc -n $Namespace -l "app.kubernetes.io/instance=$Release"

Write-Host @"

Open UI via NodePort (5601:<nodePort>) on a node IP, or port-forward:

  kubectl port-forward -n $Namespace svc/$Release 5601:5601
  # http://localhost:5601  — login: $User / <your password>

Tear down:

  helm uninstall $Release -n $Namespace
  kubectl delete secret $SecretName -n $Namespace
  kubectl delete namespace $Namespace
"@
