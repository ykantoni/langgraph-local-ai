# OpenSearch getting started

Scripts for loading data into OpenSearch from this repo.

## `load_json_opensearch.py`

Loads JSON into OpenSearch with inferred field mappings. Supports:

- **Single file** — one JSON file becomes one index (name defaults to file stem)
- **Join and embed** — two JSON files joined on a shared key; lookup records are nested inside base documents before indexing

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

This file would create index `users` with inferred types such as `id: long`, `name: text` (+ `.keyword` subfield), `active: boolean`, `score: double`, `created_at: date`.

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
| `LOG_LEVEL` | `--log-level` | `INFO` | Logging verbosity |

If `OPENSEARCH_URL` includes `https://`, TLS is enabled automatically.

### Usage — single file

Run from the repository root.

**PowerShell:**

```powershell
$env:OPENSEARCH_URL = "https://your-host:9200"
$env:OPENSEARCH_USERNAME = "admin"
$env:OPENSEARCH_PASSWORD = "your-password"

python gettingstarted/opensearch/load_json_opensearch.py path\to\users.json
```

**Bash:**

```bash
export OPENSEARCH_URL="https://your-host:9200"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="your-password"

python gettingstarted/opensearch/load_json_opensearch.py path/to/users.json
```

**Recreate an existing index:**

```powershell
python gettingstarted/opensearch/load_json_opensearch.py users.json --delete-index
```

**Override the index name:**

```powershell
python gettingstarted/opensearch/load_json_opensearch.py users.json --index my_users
```

### Usage — join and embed two JSON files

Use this to denormalize related data before indexing — for example, embed customer details into each order document.

**Example data**

`orders.json` (base / left file):

```json
[
  { "order_id": 1, "customer_id": 10, "total": 99.5 },
  { "order_id": 2, "customer_id": 20, "total": 12.0 }
]
```

`customers.json` (lookup / right file):

```json
[
  { "customer_id": 10, "name": "Alice", "tier": "gold", "active": true },
  { "customer_id": 20, "name": "Bob", "tier": "silver", "active": false }
]
```

**PowerShell:**

```powershell
python gettingstarted/opensearch/load_json_opensearch.py orders.json `
  --join-file customers.json `
  --join-on customer_id `
  --embed-as customer `
  --index orders_enriched `
  --delete-index
```

**Bash:**

```bash
python gettingstarted/opensearch/load_json_opensearch.py orders.json \
  --join-file customers.json \
  --join-on customer_id \
  --embed-as customer \
  --index orders_enriched \
  --delete-index
```

**Resulting document shape:**

```json
{
  "order_id": 1,
  "customer_id": 10,
  "total": 99.5,
  "customer": {
    "customer_id": 10,
    "name": "Alice",
    "tier": "gold",
    "active": true
  }
}
```

**Join options**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--join-file` | yes (join mode) | — | Lookup JSON file whose records are embedded |
| `--join-on` | yes (join mode) | — | Field name present in both files (e.g. `customer_id`) |
| `--embed-as` | no | join file stem | Nested object field name in the indexed document (`customers.json` → `customers`) |
| `--index` | yes (join mode) | file stem (single-file mode) | Target index name |

**Join behavior**

- The **first positional argument** is the base file; each base record becomes one OpenSearch document.
- Lookup records are matched on `--join-on`. Join keys are compared as strings (`10` matches `"10"`).
- The full lookup record is nested under `--embed-as`.
- Base records with no matching lookup row are still indexed (without the embedded field).
- Duplicate keys in the lookup file: **last record wins** (logged as a warning).

**Embedded field mappings**

Nested embedded objects get the same typed mappings as top-level fields — not just a generic `object`. For the example above:

```json
{
    "customer": {
      "type": "nested",
      "properties": {
      "customer_id": { "type": "long" },
      "name": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "tier": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "active": { "type": "boolean" }
    }
  }
}
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `json_file` | — | Primary JSON file (required) |
| `--index` | file stem | Target index name (required with `--join-file`) |
| `--delete-index` | off | Delete and recreate the index if it already exists |
| `--join-file` | — | Secondary JSON file for join/embed mode |
| `--join-on` | — | Join field name in both files |
| `--embed-as` | join file stem | Nested field name for embedded lookup record |
| `--chunk-size` | `500` | Bulk indexing batch size |
| `--request-timeout` | `120` | Bulk request timeout (seconds) |
| `--timeout` | `60` | Client connection timeout (seconds) |
| `--log-level` | `INFO` | Logging level |

### Type inference

The script scans all records (including nested embedded objects) and infers OpenSearch field types:

| Detected value | OpenSearch type |
|----------------|-----------------|
| `true` / `false` | `boolean` |
| integer | `long` |
| float, or mixed int/float | `double` |
| ISO date string (`2024-01-15`, `2024-01-15T10:00:00Z`, etc.) | `date` |
| short string (≤ 256 chars) | `text` with `fields.keyword` (`keyword`) |
| long string (> 256 chars) | `text` with `fields.keyword` (`keyword`, `ignore_above: 256`) |
| nested object | `nested` with typed `properties` |
| array | typed by contents, or `text` + `keyword` subfield if mixed |
| mixed or incompatible scalar types | `text` with `fields.keyword` (`keyword`) |

`null` values are ignored during inference. If a field is only `null` across all records, it may be omitted from the mapping.

### Index naming

The index name is derived from the file stem, lowercased and sanitized for OpenSearch:

- `users.json` → `users`
- `My Data.json` → `my_data`

Invalid characters are replaced with `_`. In join mode, use `--index` to set the target index explicitly.

### Behavior notes

- If the index **already exists** and `--delete-index` is not set, the script **skips index creation** but still bulk-indexes documents into that index.
- Document IDs are assigned sequentially (`0`, `1`, `2`, …).
- After indexing, the script refreshes the index and logs the document count.
- Progress is written through Python **logging**. Use `--log-level DEBUG` for inferred mappings and connection details.
- Certificate verification is **off** by default (`verify_certs=false`), which is convenient for self-signed or managed clusters during local testing. Use `--verify-certs` in production when you have proper CA trust configured.

### Example output

```
2026-07-03 19:00:01 INFO main: connected opensearch_version=3.0.0
2026-07-03 19:00:01 INFO main: index=orders_enriched records=2
2026-07-03 19:00:01 INFO main: fields=customer, customer_id, order_id, total
2026-07-03 19:00:01 INFO join_records: base=2 lookup=2 embedded=2 missing_match=0 missing_key=0
2026-07-03 19:00:01 INFO create_index: creating index=orders_enriched
2026-07-03 19:00:02 INFO bulk_index: done indexed=2 document_count=2
```

### Verify in OpenSearch

```bash
curl -k -u admin:your-password "https://your-host:9200/orders_enriched/_search?pretty"
```

Or use OpenSearch Dashboards / Dejavu Dev Tools:

```
GET orders_enriched/_search
GET orders_enriched/_mapping
```

### Related scripts

- `load_tlc_taxi_opensearch.py` — load NYC TLC parquet data into OpenSearch
- Cluster deployment: [`deploy/opensearch/README.md`](../../deploy/opensearch/README.md)
