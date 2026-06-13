# OpenSearch on Kubernetes (Helm, NodePort)

Deploy [OpenSearch](https://opensearch.org/) for `benchmark.py` / `benchmark8.py` using the official [`opensearch/opensearch`](https://github.com/opensearch-project/helm-charts) Helm chart. The chart version is resolved from the repo at deploy time (newest unless pinned).

Single-node cluster, HTTPS with the demo security configuration, and a **NodePort** service named `opensearch-bench` (matches `benchmark_opensearch.py` port-forward defaults).

## Prerequisites

- `kubectl` and `helm` on your PATH
- Cluster with enough memory for one OpenSearch pod (~2–4 GiB)
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
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
chmod +x deploy/opensearch/deploy.sh
./deploy/opensearch/deploy.sh
```

**Pin a specific chart version** (optional):

```powershell
$env:OPENSEARCH_CHART_VERSION = "3.1.0"
```

```bash
export OPENSEARCH_CHART_VERSION=3.1.0
```

Wait for the pod to become Ready:

```powershell
kubectl wait --for=condition=ready pod -n opensearch-bench -l app.kubernetes.io/instance=opensearch-bench --timeout=600s
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

Example:

```powershell
$env:OPENSEARCH_HOST = "172.19.73.182"
$env:OPENSEARCH_PORT = "31234"
$env:OPENSEARCH_USER = "admin"
$env:OPENSEARCH_PASSWORD = "Bench1pass!"
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

In-cluster (from the OpenSearch pod):

```bash
kubectl exec -n opensearch-bench opensearch-bench-0 -- \
  curl -sk -u "admin:${OPENSEARCH_INITIAL_ADMIN_PASSWORD}" https://localhost:9200
```

## Tear down

```powershell
helm uninstall opensearch-bench -n opensearch-bench
kubectl delete namespace opensearch-bench   # optional; removes PVC if reclaim policy allows
```

## Configuration

| File | Purpose |
|------|---------|
| `values.yaml` | Single-node Helm values (NodePort, persistence, JVM heap) |
| `deploy.ps1` | Windows deploy script; reads `OPENSEARCH_INITIAL_ADMIN_PASSWORD` |
| `deploy.sh` | Linux deploy script; same env contract |

Edit `values.yaml` to tune:

- `persistence.storageClass` — e.g. `local-storage` (see `deploy/chroma/chroma.yaml`)
- `nodeSelector` — pin to node `titan` or your storage node
- `service.nodePort` — fixed NodePort if your cluster requires it
- `opensearchJavaOpts` / `resources` — for larger benchmark datasets

The admin password is **never** written to `values.yaml`; only `OPENSEARCH_INITIAL_ADMIN_PASSWORD` at install time.

## Notes

- Default chart deploys OpenSearch **3.x** when using the current `opensearch` Helm repo (`main` branch).
- k-NN is included in the official image; benchmarks create HNSW indices via the REST API.
- Lab use only: demo TLS certificates and security config. For production, supply custom `securityConfig` and disable demo settings.
