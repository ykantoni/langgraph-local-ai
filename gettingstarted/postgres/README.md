# PostgreSQL utilities

Scripts for loading data into PostgreSQL from this repo.

## `load_json_postgres.py`

Loads a JSON file into PostgreSQL. The **table name** is taken from the file name (without extension), and **column types** are inferred by scanning JSON keys and values.

### Prerequisites

- Python 3.10+
- Dependencies installed: `pip install -r requirements.txt` (`psycopg` is required)
- A reachable PostgreSQL server (local, Docker, Kubernetes, managed cloud, etc.)

### JSON format

The file root must be either:

- a **single object** (inserted as one row), or
- an **array of objects** (each object becomes one row)

Example `users.json`:

```json
[
  {
    "id": 1,
    "name": "Alice",
    "active": true,
    "score": 9.5,
    "created_at": "2024-01-15T10:00:00Z",
    "tags": ["admin", "beta"],
    "meta": {"role": "editor"}
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

This file would create table `users` with inferred columns such as `id: BIGINT`, `name: TEXT`, `active: BOOLEAN`, `score: DOUBLE PRECISION`, `created_at: TIMESTAMPTZ`, `tags: JSONB`, `meta: JSONB`.

### Connection settings

Set credentials via environment variables (recommended) or CLI flags.

| Variable | CLI flag | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL`, `POSTGRES_URL`, or `PGVECTOR_DSN` | `--dsn` | — | Full connection string, e.g. `postgresql://user:pass@host:5432/dbname` |
| `POSTGRES_HOST` | `--host` | `localhost` | Hostname |
| `POSTGRES_PORT` | `--port` | `5432` | Port |
| `POSTGRES_USER` | `--user` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `--password` | `postgres` | Database password |
| `POSTGRES_DB` | `--database` | `postgres` | Database name |

If `--dsn` or a URL env var is set, individual host/port/user/password/database flags are ignored.

### Usage

Run from the repository root.

**PowerShell:**

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

python postgres/load_json_postgres.py path\to\users.json
```

**Bash:**

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"

python postgres/load_json_postgres.py path/to/users.json
```

**Recreate an existing table:**

```powershell
python postgres/load_json_postgres.py users.json --drop-table
```

**Override the table name:**

```powershell
python postgres/load_json_postgres.py users.json --table my_users
```

**Verbose logging:**

```powershell
python postgres/load_json_postgres.py users.json --log-level DEBUG
```

Or set `LOG_LEVEL=DEBUG`.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `json_file` | — | Path to the JSON file (required) |
| `--table` | file stem | Override table name |
| `--drop-table` | off | Drop and recreate the table if it already exists |
| `--dsn` | — | PostgreSQL connection string |
| `--host` | `localhost` | Hostname (used when no DSN is set) |
| `--port` | `5432` | Port (used when no DSN is set) |
| `--user` | `postgres` | User (used when no DSN is set) |
| `--password` | `postgres` | Password (used when no DSN is set) |
| `--database` | `postgres` | Database name (used when no DSN is set) |
| `--chunk-size` | `500` | Insert batch size |
| `--connect-timeout` | `30` | Connection timeout (seconds) |
| `--log-level` | `INFO` | Logging level (`LOG_LEVEL` env or `INFO`) |

### Type inference

The script scans all records and infers PostgreSQL column types:

| Detected value | PostgreSQL type |
|----------------|-----------------|
| `true` / `false` | `BOOLEAN` |
| integer | `BIGINT` |
| float, or mixed int/float | `DOUBLE PRECISION` |
| ISO date string (`2024-01-15`, `2024-01-15T10:00:00Z`, etc.) | `TIMESTAMPTZ` |
| short string (≤ 256 chars) | `TEXT` |
| long string (> 256 chars) | `TEXT` |
| nested object | `JSONB` |
| array | `JSONB` |
| mixed or incompatible scalar types | `TEXT` |

`null` values are ignored during inference. If a field is only `null` across all records, it may be omitted from the table schema.

### Table and column naming

Names are derived from the file stem and JSON keys, lowercased and sanitized:

- `users.json` → table `users`
- `My Data.json` → table `my_data`
- JSON key `User ID` → column `user_id`

Invalid characters are replaced with `_`. Names that start with a digit are prefixed (`1users.json` → `t_1users`).

### Behavior notes

- Tables are created in the **`public`** schema.
- If the table **already exists** and `--drop-table` is not set, the script **skips table creation** but still inserts rows.
- Nested objects and arrays are stored as **`JSONB`**.
- Progress and results are written through Python **logging** (not `print`). Use `--log-level INFO` for normal output or `DEBUG` for inferred columns and connection details.
- Missing keys in a row are inserted as `NULL`.

### Example output

```
2026-07-02 09:45:01 INFO main: loading json_file=C:\data\users.json
2026-07-02 09:45:01 INFO main: connected postgres_version=16.2.0
2026-07-02 09:45:01 INFO main: table=users records=2
2026-07-02 09:45:01 INFO main: columns=active:BOOLEAN, created_at:TIMESTAMPTZ, id:BIGINT, meta:JSONB, name:TEXT, score:DOUBLE PRECISION, tags:JSONB
2026-07-02 09:45:01 INFO create_table: creating table=users columns=7
2026-07-02 09:45:01 INFO bulk_insert: start table=users records=2 chunk_size=500
2026-07-02 09:45:01 INFO bulk_insert: done inserted=2 row_count=2
```

### Verify in PostgreSQL

```sql
SELECT * FROM users;
\d users
```

From the shell:

```bash
psql "$DATABASE_URL" -c "SELECT * FROM users;"
psql "$DATABASE_URL" -c "\d users"
```

**PowerShell:**

```powershell
psql $env:DATABASE_URL -c "SELECT * FROM users;"
```

### Related scripts

- OpenSearch equivalent: [`opensearch/load_json_opensearch.py`](../opensearch/load_json_opensearch.py)
- pgvector benchmarks in this repo use `PGVECTOR_DSN` for PostgreSQL connections (`benchmark.py`, `benchmark8.py`)
