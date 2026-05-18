# pgvector on Kubernetes (NodePort)

Deploy Postgres with the [pgvector](https://github.com/pgvector/pgvector) extension, backed by a PVC (uses the cluster **default StorageClass** when none is set on the PVC) and exposed with **`type: NodePort`**.

## Prerequisites

- Default `StorageClass` (PVC binds without `storageClassName`)
- MetalLB is **not** required for this manifest

## Deploy (from repo root on any host with `kubectl`)

```bash
kubectl apply -f deploy/pgvector/postgres.yaml
```

**PowerShell** (from repo root on Windows):

```powershell
kubectl apply -f .\deploy\pgvector\postgres.yaml
```

Wait for the Deployment and PVC:

```bash
kubectl wait --for=condition=available deployment/pgvector-bench -n pgvector-bench --timeout=300s
kubectl get nodes -o wide
kubectl get svc pgvector-bench -n pgvector-bench
```

Note the node **INTERNAL-IP** and the Service line `5432:<nodePort>/TCP` (e.g. `5432:32381/TCP` → use port **32381**).

## Connect from your machine

**NodePort on the node IP** (works from Windows when Kubernetes runs in WSL2):

```powershell
$env:PGVECTOR_DSN="postgresql://postgres:postgres@<NODE_IP>:<NODE_PORT>/postgres"
python scripts/pg_connectivity_check.py
```

Example: node `172.19.73.182`, NodePort `32381`:

```powershell
$env:PGVECTOR_DSN="postgresql://postgres:postgres@172.19.73.182:32381/postgres"
```

**In-cluster** (ClusterIP DNS, port 5432):

```bash
kubectl run pg-test --rm -it --restart=Never -n pgvector-bench \
  --image=postgres:16 --env="PGPASSWORD=postgres" \
  -- psql -h pgvector-bench -U postgres -d postgres -c "SELECT 1;"
```

**port-forward** (localhost):

```bash
kubectl port-forward -n pgvector-bench svc/pgvector-bench 5432:5432
# PGVECTOR_DSN=postgresql://postgres:postgres@127.0.0.1:5432/postgres
```

Default credentials are in the Secret `pgvector-bench-credentials` (`postgres` / `postgres` / `postgres`). **Change for anything beyond a lab.**

## Tear down

```bash
kubectl delete -f deploy/pgvector/postgres.yaml
```

## Troubleshooting

### `chmod` / `initdb: ... Operation not permitted` on `/var/lib/postgresql/data`

Do **not** set container `runAsUser` / `runAsGroup` on the main postgres container. The image entrypoint must run as root briefly to fix PVC permissions, then drops to `postgres`. Pod-level `fsGroup: 999` is sufficient.

### Service type change (LoadBalancer → NodePort)

If you previously applied a LoadBalancer Service, delete it before re-applying or run:

```bash
kubectl delete svc pgvector-bench -n pgvector-bench
kubectl apply -f deploy/pgvector/postgres.yaml
```

### `fsGroupChangePolicy` rejected on old clusters

Requires Kubernetes **≥ 1.23**. Remove that line from the Deployment `securityContext` if needed.

## Files

| File | Purpose |
|------|---------|
| `postgres.yaml` | Namespace, Secret, PVC, Deployment, NodePort Service |
