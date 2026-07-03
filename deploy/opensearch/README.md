# OpenSearch on Kubernetes (Operator, NodePort)

Deploy [OpenSearch](https://opensearch.org/) for `benchmark.py` / `benchmark8.py` using the official [OpenSearch Kubernetes Operator](https://github.com/opensearch-project/opensearch-k8s-operator). The operator is installed with Helm; the cluster is an `OpenSearchCluster` custom resource.

HTTPS with operator-generated TLS, **NodePort** on service `opensearch-bench` (matches `benchmark_opensearch.py` port-forward defaults).

## Prerequisites

- `kubectl` and `helm` on your PATH
- Cluster with enough memory for a **3-node** OpenSearch cluster (~6–9 GiB; operator does not support single-node)
- **cert-manager is optional** — if not installed, deploy scripts disable operator admission webhooks automatically
- **Admin password** in the environment (required for OpenSearch 2.12+):

  - At least 8 characters
  - Uppercase, lowercase, digit, and special character

## Deploy

**PowerShell (Windows):**

```powershell
$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "Bench1pass!"
.\deploy\opensearch\deploy.ps1
```

**Bash (Linux / WSL / cluster jump host):**

```bash
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!' # 123QWEasd#%¤
chmod +x deploy/opensearch/deploy.sh
./deploy/opensearch/deploy.sh
kubectl -n opensearch-bench expose service opensearch-bench --name=opensearch-external --type=NodePort
```

The script:

1. Installs the OpenSearch Operator via Helm (latest chart from repo unless pinned)
2. Creates namespace `opensearch-bench` and admin credentials secret
3. Applies `OpenSearchCluster` `opensearch-bench` (OpenSearch **3.x** by default)
4. Waits for pods and patches the service to `NodePort`

**Skip operator install** if it is already cluster-wide:

```powershell
$env:OPENSEARCH_SKIP_OPERATOR = "1"
```

```bash
export OPENSEARCH_SKIP_OPERATOR=1
```

**Pin versions** (optional):

```powershell
$env:OPENSEARCH_VERSION = "3.1.0"
$env:OPENSEARCH_OPERATOR_VERSION = "3.0.3"
```

```bash
export OPENSEARCH_VERSION=3.1.0
export OPENSEARCH_OPERATOR_VERSION=3.0.3
```

Check readiness and NodePort:

```powershell
kubectl get pods -n opensearch-bench -l opensearch.org/opensearch-cluster=opensearch-bench
kubectl get nodes -o wide
kubectl get svc opensearch-bench -n opensearch-bench
```

Note the node **INTERNAL-IP** and `9200:<nodePort>/TCP` (e.g. `9200:31234/TCP` → port **31234**).

## Run benchmark from your machine

```powershell
$env:OPENSEARCH_HOST = "<NODE_IP>"
$env:OPENSEARCH_PORT = "<NODE_PORT>"
$env:OPENSEARCH_USER = "admin"
$env:OPENSEARCH_PASSWORD = $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
$env:OPENSEARCH_USE_SSL = "1"
python benchmark.py
```

**Port-forward** (localhost):

```powershell
kubectl port-forward -n opensearch-bench svc/opensearch-bench 9200:9200
$env:OPENSEARCH_KUBECTL_PORT_FORWARD = "1"
$env:OPENSEARCH_USER = "admin"
$env:OPENSEARCH_PASSWORD = $env:OPENSEARCH_INITIAL_ADMIN_PASSWORD
$env:OPENSEARCH_USE_SSL = "1"
python benchmark.py
```

## Verify

```bash
kubectl exec -n opensearch-bench opensearch-bench-masters-0 -- \
  curl -sk -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" https://localhost:9200
```

curl --insecure -XGET -u 'admin:admin' 'https://2.2.60.94:9200/_cat/nodes?v'

## Tear down

**Cluster only** (operator stays installed):

```powershell
kubectl delete opensearchcluster opensearch-bench -n opensearch-bench
kubectl delete secret opensearch-bench-admin-credentials -n opensearch-bench
kubectl delete pvc -l opensearch.org/opensearch-cluster=opensearch-bench -n opensearch-bench
```

**Operator** (cluster-wide):

```powershell
helm uninstall opensearch-operator -n opensearch-operator-system
```

## Configuration

| File | Purpose |
|------|---------|
| `cluster.yaml` | `OpenSearchCluster` manifest template (`__OPENSEARCH_VERSION__` substituted at deploy) |
| `apply-cluster.sh` / `apply-cluster.ps1` | Apply config changes to an existing cluster (renders version placeholder) |
| `rollout-restart.sh` / `rollout-restart.ps1` | Manual rolling pod restart when the operator does not restart nodes |
| `operator-values.yaml` | Operator Helm defaults (webhooks off without cert-manager) |
| `deploy.ps1` | Windows deploy script |
| `deploy.sh` | Linux deploy script |
| `install-operator.sh` | Shared operator Helm install |
| `dashboards/` | Helm Dashboards → external OpenSearch ([`dashboards/README.md`](dashboards/README.md)) |

For **Dashboards only** (external OpenSearch at `10.1.139.63:9200`), see [`dashboards/README.md`](dashboards/README.md).

Edit `cluster.yaml` to tune:

- `nodePools[].diskSize` / `resources` / `jvm` — heap and storage
- `nodePools[].nodeSelector` — pin to node `titan` (see `deploy/chroma/chroma.yaml`)
- `dashboards.enable: true` — optional Dashboards on the same OpenSearchCluster (or use `dashboards/` Helm chart for external OpenSearch)

The admin password is **never** stored in `cluster.yaml`. It is passed via `OPENSEARCH_INITIAL_ADMIN_PASSWORD` into secret `opensearch-bench-admin-credentials`, referenced by `adminCredentialsSecret` and `OPENSEARCH_INITIAL_ADMIN_PASSWORD` on node pods.

## Update cluster configuration (e.g. CORS)

Do **not** run `kubectl apply -f deploy/opensearch/cluster.yaml` directly. The template contains `__OPENSEARCH_VERSION__`, which must be substituted before apply.

**Apply rendered manifest:**

```bash
chmod +x deploy/opensearch/apply-cluster.sh deploy/opensearch/rollout-restart.sh
./deploy/opensearch/apply-cluster.sh
```

```powershell
.\deploy\opensearch\apply-cluster.ps1
```

The operator should detect `additionalConfig` changes and perform a rolling restart. It waits for the cluster to return to **GREEN** between each pod; if the cluster is **YELLOW**, the restart can appear stuck and old pods keep running.

**Force a rolling restart** (after apply, or if pods did not recycle within a few minutes):

```bash
./deploy/opensearch/rollout-restart.sh
```

```powershell
.\deploy\opensearch\rollout-restart.ps1
```

Or apply and restart in one step:

```bash
OPENSEARCH_ROLLOUT_RESTART=1 ./deploy/opensearch/apply-cluster.sh
```

**Verify CORS settings inside a pod:**

```bash
kubectl exec -n opensearch-bench opensearch-bench-masters-0 -- \
  grep -E '^http\.cors\.' /usr/share/opensearch/config/opensearch.yml
```

**Check operator logs** if restart never starts:

```bash
kubectl logs -n opensearch-operator-system \
  -l app.kubernetes.io/name=opensearch-operator --tail=100
```

Look for messages such as `RollingRestart` or `Cluster is not ready for next pod to restart`.

## Troubleshooting

**`no matches for kind "Certificate" in version "cert-manager.io/v1"`**

The operator chart defaults to cert-manager for webhook TLS. Re-run deploy with the updated script (it disables webhooks when cert-manager is absent), or install cert-manager first.

If a failed install left a partial release:

```bash
helm uninstall opensearch-operator -n opensearch-operator-system
./deploy/opensearch/deploy.sh
```

Force webhooks on only when cert-manager is installed:

```bash
export OPENSEARCH_OPERATOR_WEBHOOKS=1
```

## Notes

- Operator minimum: **3** nodes with `cluster_manager` role (no single-node).
- Default `OPENSEARCH_VERSION=3` tracks the latest OpenSearch 3.x image tag.
- k-NN is included in the official image; benchmarks create HNSW indices via the REST API.
- Lab use only: operator-generated demo TLS. For production, supply custom `securityConfig` and certificates.
