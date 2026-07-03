# OpenSearch Dashboards (Helm)

Deploy [OpenSearch Dashboards](https://opensearch.org/docs/latest/install-and-configure/install-dashboards/index/) using the official [`opensearch/opensearch-dashboards`](https://github.com/opensearch-project/helm-charts) Helm chart, connected to an **existing** OpenSearch cluster.

Default backend: **`https://10.1.139.63:9200`**.

## Prerequisites

- `kubectl` and `helm`
- OpenSearch admin credentials

## Deploy

```bash
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
chmod +x deploy/opensearch/dashboards/deploy.sh
./deploy/opensearch/dashboards/deploy.sh
```

```powershell
$env:OPENSEARCH_INITIAL_ADMIN_PASSWORD = "Bench1pass!"
.\deploy\opensearch\dashboards\deploy.ps1
```

If you previously deployed the operator-based Dashboards CR, remove it first:

```bash
kubectl delete opensearchcluster opensearch-dashboards -n opensearch-dashboards 2>/dev/null || true
```

## Configuration (env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENSEARCH_HOST` | `10.1.139.63` | External OpenSearch IP or hostname |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `OPENSEARCH_USE_SSL` | `1` | `1` = `https`, `0` = `http` |
| `OPENSEARCH_DASHBOARDS_USER` | `admin` | Backend + UI login user |
| `OPENSEARCH_DASHBOARDS_PASSWORD` | — | Falls back to `OPENSEARCH_INITIAL_ADMIN_PASSWORD` |
| `OPENSEARCH_DASHBOARDS_COOKIE` | — | 32-char security cookie secret; auto-generated if omitted |
| `OPENSEARCH_DASHBOARDS_RELEASE` | `opensearch-dashboards` | Helm release name |
| `OPENSEARCH_DASHBOARDS_CHART_VERSION` | — | Pin chart version (omit for newest) |
| `OPENSEARCH_DASHBOARDS_IMAGE_TAG` | `3` | Dashboards image tag |

Example — HTTP backend:

```bash
export OPENSEARCH_HOST=10.1.139.63
export OPENSEARCH_PORT=9200
export OPENSEARCH_USE_SSL=0
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='Bench1pass!'
./deploy/opensearch/dashboards/deploy.sh
```

## Access UI

```bash
kubectl get nodes -o wide
kubectl get svc -n opensearch-dashboards -l app.kubernetes.io/instance=opensearch-dashboards
```

```bash
kubectl port-forward -n opensearch-dashboards svc/opensearch-dashboards 5601:5601
```

Open [http://localhost:5601](http://localhost:5601) and sign in with `admin` and your OpenSearch password.

## Verify

```bash
helm list -n opensearch-dashboards
kubectl logs -n opensearch-dashboards -l app.kubernetes.io/instance=opensearch-dashboards --tail=50
```

## Tear down

```bash
helm uninstall opensearch-dashboards -n opensearch-dashboards
kubectl delete namespace opensearch-dashboards
```

## Files

| File | Purpose |
|------|---------|
| `values.yaml` | Base Helm values (NodePort, SSL verify off, resources) |
| `deploy.sh` / `deploy.ps1` | Helm install + credentials secret |

## Notes

- The credentials secret must include `username`, `password`, and `cookie` (used as `COOKIE_PASS` by the chart).
- `opensearch.ssl.verificationMode: none` in `values.yaml` supports self-signed TLS on the backend.
- Match `OPENSEARCH_DASHBOARDS_IMAGE_TAG` to your OpenSearch major version (e.g. `3` for OpenSearch 3.x).
- UI is HTTP on port **5601** inside the cluster (lab setup).
