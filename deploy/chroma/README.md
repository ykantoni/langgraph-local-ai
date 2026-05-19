# Chroma server on Kubernetes (NodePort)

Deploy the [Chroma](https://www.trychroma.com/) HTTP server for `benchmark.py`, with data on a **local-storage** PV (node `titan`, path `/mnt/kubehome/chroma`) and **`type: NodePort`** on port **8000**.

## Deploy

```powershell
kubectl apply -f .\deploy\chroma\chroma.yaml
kubectl wait --for=condition=available deployment/chroma-bench -n chroma-bench --timeout=300s
kubectl get nodes -o wide
kubectl get svc chroma-bench -n chroma-bench
```

Note the node **INTERNAL-IP** and the `8000:<nodePort>/TCP` line (e.g. `8000:31234/TCP` → use port **31234**).

## Run benchmark from your machine

Set host/port so `benchmark.py` uses `chromadb.HttpClient` instead of local `PersistentClient`:

```powershell
$env:CHROMA_HOST="<NODE_IP>"
$env:CHROMA_PORT="<NODE_PORT>"
python benchmark.py
```

Example:

```powershell
$env:CHROMA_HOST="172.19.73.182"
$env:CHROMA_PORT="31234"
python benchmark.py
```

**port-forward** (localhost):

```powershell
kubectl port-forward -n chroma-bench svc/chroma-bench 8000:8000
$env:CHROMA_HOST="127.0.0.1"
$env:CHROMA_PORT="8000"
python benchmark.py
```

## Verify

```powershell
curl http://<NODE_IP>:<NODE_PORT>/api/v2/heartbeat
```

In-cluster:

```bash
kubectl run chroma-test --rm -it --restart=Never -n chroma-bench \
  --image=curlimages/curl -- curl -sf http://chroma-bench:8000/api/v2/heartbeat
```

## Tear down

```powershell
kubectl delete -f .\deploy\chroma\chroma.yaml
```

## Notes

- Image `chromadb/chroma:1.5.9` matches `chromadb==1.5.9` in `requirements.txt`.
- Without `CHROMA_HOST`, the benchmark uses an embedded `PersistentClient` under `./chroma_data` (no cluster required).
- Ensure `/mnt/kubehome/chroma` exists on node `titan`, or edit the PV `local.path` / `nodeAffinity` for your cluster.

## Files

| File | Purpose |
|------|---------|
| `chroma.yaml` | Namespace, PV, PVC, Deployment, NodePort Service |
