# OpenSearch utilities

Scripts for loading data into OpenSearch from this repo.

## `load_json_opensearch.py`

Loads a JSON file into OpenSearch. The **index name** is taken from the file name (without extension), and **field mappings** are inferred by scanning JSON keys and values.

### Prerequisites

- Python 3.10+
- Dependencies installed: `pip install -r requirements.txt` (`opensearch-py` is required)
- A reachable OpenSearch cluster (local, Kubernetes, Aiven, etc.)

### JSON format

The file root must be either:

- a **single object** (indexed as one document), or
- an **array of objects** (each object becomes one document)

Example `users.json`:

```json
[
  {
    "id": 1,
    "name": "Alice",
    "active": true,
    "score": 9.5,
    "created_at": "2024-01-15T10:00:00Z"
  },
  {
    "id": 2,
    "name": "Bob",
    "active": false,
    "score": 8,
    "created_at": "2024-02-01"
  }
]
```

This file would create index `users` with inferred types such as `id: long`, `name: keyword`, `active: boolean`, `score: double`, `created_at: date`.

### Connection settings

Set credentials via environment variables (recommended) or CLI flags.

| Variable | CLI flag | Default | Description |
|----------|----------|---------|-------------|
| `OPENSEARCH_URL` | `--url` | — | Full URL, e.g. `https://host:9200` (overrides host/port/ssl when set) |
| `OPENSEARCH_HOST` | `--host` | `localhost` | Hostname |
| `OPENSEARCH_PORT` | `--port` | `9200` | Port |
| `OPENSEARCH_USERNAME` or `OPENSEARCH_USER` | `--user` | `admin` | HTTP basic auth user |
| `OPENSEARCH_PASSWORD` | `--password` | `admin` | HTTP basic auth password |
| `OPENSEARCH_USE_SSL` | `--use-ssl` / `--no-use-ssl` | `true` | Use HTTPS |
| `OPENSEARCH_VERIFY_CERTS` | `--verify-certs` / `--no-verify-certs` | `false` | Verify TLS certificates |

If `OPENSEARCH_URL` includes `https://`, TLS is enabled automatically.

### Usage

Run from the repository root.

**PowerShell:**

```powershell
$env:OPENSEARCH_URL = "https://your-host:9200"
$env:OPENSEARCH_USERNAME = "admin"
$env:OPENSEARCH_PASSWORD = "your-password"

python opensearch/load_json_opensearch.py path\to\users.json
```

**Bash:**

```bash
export OPENSEARCH_URL="https://your-host:9200"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="your-password"

python opensearch/load_json_opensearch.py path/to/users.json
```

**Recreate an existing index:**

```powershell
python opensearch/load_json_opensearch.py users.json --delete-index
```

**Override the index name:**

```powershell
python opensearch/load_json_opensearch.py users.json --index my_users
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `json_file` | — | Path to the JSON file (required) |
| `--index` | file stem | Override index name |
| `--delete-index` | off | Delete and recreate the index if it already exists |
| `--chunk-size` | `500` | Bulk indexing batch size |
| `--request-timeout` | `120` | Bulk request timeout (seconds) |
| `--timeout` | `60` | Client connection timeout (seconds) |

### Type inference

The script scans all records and infers OpenSearch field types:

| Detected value | OpenSearch type |
|----------------|-----------------|
| `true` / `false` | `boolean` |
| integer | `long` |
| float, or mixed int/float | `double` |
| ISO date string (`2024-01-15`, `2024-01-15T10:00:00Z`, etc.) | `date` |
| short string (≤ 256 chars) | `keyword` |
| long string (> 256 chars) | `text` |
| nested object | `object` |
| mixed or incompatible types | `keyword` |

`null` values are ignored during inference. If a field is only `null` across all records, it may be omitted from the mapping.

### Index naming

The index name is derived from the file stem, lowercased and sanitized for OpenSearch:

- `users.json` → `users`
- `My Data.json` → `my_data`

Invalid characters are replaced with `_`.

### Behavior notes

- If the index **already exists** and `--delete-index` is not set, the script **skips index creation** but still bulk-indexes documents into that index.
- Document IDs are assigned sequentially (`0`, `1`, `2`, …).
- After indexing, the script refreshes the index and prints the document count.
- Certificate verification is **off** by default (`verify_certs=false`), which is convenient for self-signed or managed clusters during local testing. Use `--verify-certs` in production when you have proper CA trust configured.

### Example output

```
Connected to OpenSearch 3.0.0
Index: users
Records: 2
Fields: active, created_at, id, name, score
Creating index: users
Indexed: 2
Document count: 2
```

### Verify in OpenSearch

```bash
curl -k -u admin:your-password "https://your-host:9200/users/_search?pretty"
```

Or use OpenSearch Dashboards Dev Tools:

```
GET users/_search
GET users/_mapping
```
