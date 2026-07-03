# Deploy OpenSearch for vector-store benchmarks via the OpenSearch Kubernetes Operator.
#
# Requires: kubectl, helm
#
#   $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "Bench1pass!"
#   .\deploy\opensearch\deploy.ps1
#
# Optional:
#   $env:OPENSEARCH_VERSION = "3"
#   $env:OPENSEARCH_OPERATOR_VERSION = ""   # pin operator chart; omit for newest
#   $env:OPENSEARCH_NAMESPACE = "opensearch-bench"
#   $env:OPENSEARCH_SKIP_OPERATOR = "1"     # skip if operator already installed

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ClusterTemplate = Join-Path $ScriptDir "cluster.yaml"
$OperatorValues = Join-Path $ScriptDir "operator-values.yaml"

$Namespace = if ($env:OPENSEARCH_NAMESPACE) { $env:OPENSEARCH_NAMESPACE } else { "opensearch-bench" }
$OperatorNs = if ($env:OPENSEARCH_OPERATOR_NAMESPACE) { $env:OPENSEARCH_OPERATOR_NAMESPACE } else { "opensearch-operator-system" }
$OperatorRelease = if ($env:OPENSEARCH_OPERATOR_RELEASE) { $env:OPENSEARCH_OPERATOR_RELEASE } else { "opensearch-operator" }
$ClusterName = if ($env:OPENSEARCH_CLUSTER_NAME) { $env:OPENSEARCH_CLUSTER_NAME } else { "opensearch-bench" }
$AdminSecret = if ($env:OPENSEARCH_ADMIN_SECRET) { $env:OPENSEARCH_ADMIN_SECRET } else { "opensearch-bench-admin-credentials" }
$OpenSearchVersion = if ($env:OPENSEARCH_VERSION) { $env:OPENSEARCH_VERSION } else { "3" }

$OperatorRepo = "opensearch-operator"
$OperatorRepoUrl = "https://opensearch-project.github.io/opensearch-k8s-operator/"
$OperatorChart = "$OperatorRepo/opensearch-operator"

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

function Get-Exe([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "$Name not found on PATH."
}

function Get-LatestOperatorChartVersion([string]$HelmExe) {
    $json = & $HelmExe search repo $OperatorChart --versions -o json 2>$null | ConvertFrom-Json
    if (-not $json -or $json.Count -eq 0) {
        throw "Could not resolve operator chart version from $OperatorRepoUrl"
    }
    return $json[0].version
}

function Test-CertManagerInstalled {
    kubectl get crd certificates.cert-manager.io 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Get-OperatorWebhookHelmArgs {
    if ($env:OPENSEARCH_OPERATOR_WEBHOOKS -eq "1") {
        if (Test-CertManagerInstalled) {
            Write-Host "Enabling operator admission webhooks (cert-manager detected)."
            return @("--set", "webhook.enabled=true", "--set", "webhook.certManager.enabled=true")
        }
        throw "OPENSEARCH_OPERATOR_WEBHOOKS=1 but cert-manager is not installed. Install cert-manager or unset OPENSEARCH_OPERATOR_WEBHOOKS."
    }

    if ($env:OPENSEARCH_OPERATOR_WEBHOOKS -eq "0" -or -not (Test-CertManagerInstalled)) {
        Write-Host "Disabling operator admission webhooks (cert-manager not required)."
        return @("--set", "webhook.enabled=false")
    }

    Write-Host "cert-manager detected; enabling operator admission webhooks."
    return @("--set", "webhook.enabled=true", "--set", "webhook.certManager.enabled=true")
}

function Install-Operator([string]$HelmExe) {
    if ($env:OPENSEARCH_SKIP_OPERATOR) {
        Write-Host "Skipping operator install (OPENSEARCH_SKIP_OPERATOR is set)."
        return
    }

    Write-Host "Adding Helm repo $OperatorRepo ..."
    & $HelmExe repo add $OperatorRepo $OperatorRepoUrl 2>$null | Out-Null
    & $HelmExe repo update $OperatorRepo | Out-Null

    $chartVersion = $env:OPENSEARCH_OPERATOR_VERSION
    if (-not $chartVersion) {
        $chartVersion = Get-LatestOperatorChartVersion -HelmExe $HelmExe
    }
    Write-Host "Installing operator $OperatorChart chart version $chartVersion ..."

    $webhookArgs = Get-OperatorWebhookHelmArgs

    & $HelmExe upgrade --install $OperatorRelease $OperatorChart `
        --namespace $OperatorNs `
        --create-namespace `
        --version $chartVersion `
        -f $OperatorValues `
        @webhookArgs `
        --wait `
        --timeout 10m

    Write-Host "Waiting for operator controller ..."
    kubectl wait --for=condition=available deployment `
        -l "app.kubernetes.io/name=opensearch-operator" `
        -n $OperatorNs `
        --timeout=300s
}

function Apply-Cluster {
    $rendered = Join-Path $env:TEMP "opensearch-bench-cluster.yaml"
    (Get-Content $ClusterTemplate -Raw).Replace("__OPENSEARCH_VERSION__", $OpenSearchVersion) |
        Set-Content -Path $rendered -Encoding utf8

    Write-Host "Creating namespace $Namespace ..."
    kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -

    Write-Host "Creating admin credentials secret $AdminSecret ..."
    kubectl -n $Namespace create secret generic $AdminSecret `
        --from-literal=username=admin `
        --from-literal=password=$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD `
        --dry-run=client -o yaml | kubectl apply -f -

    Write-Host "Applying OpenSearchCluster $ClusterName (OpenSearch $OpenSearchVersion) ..."
    kubectl apply -f $rendered

    Write-Host "Waiting for OpenSearch pods ..."
    kubectl wait --for=condition=ready pod `
        -n $Namespace `
        -l "opensearch.org/opensearch-cluster=$ClusterName" `
        --timeout=900s

    Write-Host "Patching service $ClusterName to NodePort ..."
    kubectl patch svc $ClusterName -n $Namespace --type=merge `
        -p '{"spec":{"type":"NodePort"}}'
}

$Helm = Get-Exe "helm"
Get-Exe "kubectl" | Out-Null

Install-Operator -HelmExe $Helm
Apply-Cluster

Write-Host ""
Write-Host "Pods:"
kubectl get pods -n $Namespace -l "opensearch.org/opensearch-cluster=$ClusterName"

Write-Host ""
Write-Host "Service (note http:<nodePort>):"
kubectl get svc -n $Namespace $ClusterName

Write-Host ""
Write-Host @"
OpenSearch cluster is ready. Run benchmarks from your host:

  `$env:OPENSEARCH_HOST = '<NODE_INTERNAL_IP>'
  `$env:OPENSEARCH_PORT = '<NODE_PORT>'
  `$env:OPENSEARCH_USER = 'admin'
  `$env:OPENSEARCH_PASSWORD = `$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
  `$env:OPENSEARCH_USE_SSL = '1'
  python benchmark.py

Or port-forward (matches benchmark_opensearch defaults):

  kubectl port-forward -n $Namespace svc/$ClusterName 9200:9200
  `$env:OPENSEARCH_KUBECTL_PORT_FORWARD = '1'
  `$env:OPENSEARCH_USER = 'admin'
  `$env:OPENSEARCH_PASSWORD = `$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
  `$env:OPENSEARCH_USE_SSL = '1'
  python benchmark.py

Tear down cluster (operator remains installed):

  kubectl delete opensearchcluster $ClusterName -n $Namespace
  kubectl delete secret $AdminSecret -n $Namespace
  kubectl delete pvc -l opensearch.org/opensearch-cluster=$ClusterName -n $Namespace

Uninstall operator:

  helm uninstall $OperatorRelease -n $OperatorNs
"@
