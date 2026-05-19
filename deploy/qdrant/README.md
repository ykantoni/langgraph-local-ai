# Qdrant on Kubernetes (NodePort)

Deploy [Qdrant](https://qdrant.tech/) for `benchmark.py`, with a PVC on the cluster **default StorageClass** and **`type: NodePort`** HTTP/gRPC access.

## Deploy

```powershell
kubectl apply -f .\deploy\qdrant\qdrant.yaml
kubectl wait --for=condition=available deployment/qdrant-bench -n qdrant-bench --timeout=300s
kubectl get nodes -o wide
kubectl get svc qdrant-bench -n qdrant-bench
```

Note the node **INTERNAL-IP** and the `6333:<nodePort>/TCP` line (e.g. `6333:31234/TCP` → use port **31234**).

## Run benchmark from your machine

```powershell
$env:QDRANT_URL="http://<NODE_IP>:<NODE_PORT>"
python benchmark.py
```

Example:

```powershell
$env:QDRANT_URL="http://172.19.73.182:31234"
python benchmark.py
```

**port-forward** (localhost):

```powershell
kubectl port-forward -n qdrant-bench svc/qdrant-bench 6333:6333
$env:QDRANT_URL="http://127.0.0.1:6333"
python benchmark.py
```

## Verify

```powershell
curl http://<NODE_IP>:<NODE_PORT>/readyz
```

In-cluster:

```bash
kubectl run qdrant-test --rm -it --restart=Never -n qdrant-bench \
  --image=curlimages/curl -- curl -sf http://qdrant-bench:6333/readyz
```

## Tear down

```powershell
kubectl delete -f .\deploy\qdrant\qdrant.yaml
```

## Notes

- Image `qdrant/qdrant:v1.17.0` aligns with `qdrant-client==1.17.0` in `requirements.txt`.
- For a fixed node + `local-storage` PV (like `deploy/pgvector/postgres.yaml`), bind a PV/PVC before applying or edit the PVC `storageClassName` / `volumeName`.
- Lab deploy: no API key. For production, set `QDRANT__SERVICE__API_KEY` on the Deployment and pass the key in `QdrantClient` / `QDRANT_URL` as required by your client version.

## Files

| File | Purpose |
|------|---------|
| `qdrant.yaml` | Namespace, PVC, Deployment, NodePort Service |
