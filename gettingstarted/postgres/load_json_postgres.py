#!/usr/bin/env python3
"""Load a JSON file into PostgreSQL. Table name = file stem; columns inferred from values."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg import sql
from psycopg.types.json import Json

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
SQL_NAME_RE = re.compile(r"[^a-z0-9_]+")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def resolve_connection(args: argparse.Namespace) -> str:
    dsn = (
        (args.dsn or "").strip()
        or os.getenv("DATABASE_URL", "").strip()
        or os.getenv("POSTGRES_URL", "").strip()
        or os.getenv("PGVECTOR_DSN", "").strip()
    )
    if dsn:
        return dsn

    host = args.host or os.getenv("POSTGRES_HOST", "localhost")
    port = args.port or int(os.getenv("POSTGRES_PORT", "5432"))
    user = args.user or os.getenv("POSTGRES_USER", "postgres")
    password = args.password or os.getenv("POSTGRES_PASSWORD", "postgres")
    database = args.database or os.getenv("POSTGRES_DB", "postgres")

    logger.debug("resolve_connection: %s@%s:%s/%s", user, host, port, database)
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def sql_name_from_file(path: Path) -> str:
    name = SQL_NAME_RE.sub("_", path.stem.lower()).strip("_")
    if not name:
        raise ValueError(f"Cannot derive table name from file: {path}")
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def sql_column_name(key: str) -> str:
    name = SQL_NAME_RE.sub("_", key.lower()).strip("_")
    if not name:
        raise ValueError(f"Invalid JSON key for column name: {key!r}")
    if name[0].isdigit():
        name = f"f_{name}"
    return name


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not data:
            return []
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON array must contain objects")
        return data

    raise ValueError("JSON root must be an object or an array of objects")


def _looks_like_date(value: str) -> bool:
    return bool(ISO_DATE_RE.match(value.strip()))


def _value_kind(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        if _looks_like_date(value):
            return "date"
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        kinds = {_value_kind(item) for item in value}
        kinds.discard(None)
        if not kinds:
            return None
        if len(kinds) == 1:
            return kinds.pop()
        return "keyword"
    return "keyword"


def _merge_kinds(kinds: set[str]) -> str:
    kinds.discard(None)  # type: ignore[arg-type]
    if not kinds:
        return "keyword"
    if kinds == {"boolean"}:
        return "boolean"
    if kinds == {"long"}:
        return "long"
    if kinds <= {"long", "double"}:
        return "double"
    if kinds == {"date"}:
        return "date"
    if kinds == {"object"}:
        return "object"
    if kinds == {"string"}:
        return "keyword"
    if kinds <= {"string", "date"}:
        return "date" if "date" in kinds else "keyword"
    return "keyword"


def _pg_type(merged: str, *, has_json_container: bool, string_samples: list[str]) -> str:
    if has_json_container or merged == "object":
        return "JSONB"
    if merged == "boolean":
        return "BOOLEAN"
    if merged == "long":
        return "BIGINT"
    if merged == "double":
        return "DOUBLE PRECISION"
    if merged == "date":
        return "TIMESTAMPTZ"
    if any(len(value) > 256 for value in string_samples):
        return "TEXT"
    return "TEXT"


def infer_columns(records: list[dict[str, Any]]) -> dict[str, str]:
    """Return mapping of original JSON keys to PostgreSQL column types."""
    field_kinds: dict[str, set[str]] = {}
    string_samples: dict[str, list[str]] = {}
    json_containers: set[str] = set()

    for record in records:
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                json_containers.add(key)
                continue
            kind = _value_kind(value)
            if kind is None:
                continue
            field_kinds.setdefault(key, set()).add(kind)
            if kind == "string" and isinstance(value, str):
                string_samples.setdefault(key, []).append(value)

    columns: dict[str, str] = {}
    keys = set(field_kinds) | json_containers
    for key in sorted(keys):
        merged = _merge_kinds(field_kinds.get(key, set()))
        columns[key] = _pg_type(
            merged,
            has_json_container=key in json_containers,
            string_samples=string_samples.get(key, []),
        )

    return columns


def table_exists(conn: psycopg.Connection, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        row = cur.fetchone()
        return bool(row and row[0])


def create_table(
    conn: psycopg.Connection,
    table_name: str,
    columns: dict[str, str],
    *,
    drop_existing: bool,
) -> None:
    exists = table_exists(conn, table_name)
    if exists and drop_existing:
        logger.info("create_table: dropping existing table=%s", table_name)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP TABLE {}").format(sql.Identifier(table_name))
            )
        conn.commit()
        exists = False

    if exists:
        logger.info("create_table: table already exists=%s", table_name)
        return

    if not columns:
        raise ValueError("No columns inferred from JSON records")

    column_defs = [
        sql.SQL("{} {}").format(sql.Identifier(sql_column_name(key)), sql.SQL(pg_type))
        for key, pg_type in columns.items()
    ]
    create_stmt = sql.SQL("CREATE TABLE {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(column_defs),
    )

    logger.info("create_table: creating table=%s columns=%d", table_name, len(columns))
    logger.debug("create_table: ddl=%s", create_stmt.as_string(conn))
    with conn.cursor() as cur:
        cur.execute(create_stmt)
    conn.commit()


def _prepare_value(value: Any, pg_type: str) -> Any:
    if value is None:
        return None
    if pg_type == "JSONB":
        return Json(value)
    return value


def bulk_insert(
    conn: psycopg.Connection,
    table_name: str,
    records: list[dict[str, Any]],
    columns: dict[str, str],
    *,
    chunk_size: int,
) -> int:
    if not columns:
        return 0

    json_keys = list(columns.keys())
    sql_columns = [sql_column_name(key) for key in json_keys]
    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(sql_columns))
    insert_stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, sql_columns)),
        placeholders,
    )

    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(records), chunk_size):
            batch = records[start : start + chunk_size]
            rows = [
                tuple(
                    _prepare_value(record.get(key), columns[key])
                    for key in json_keys
                )
                for record in batch
            ]
            cur.executemany(insert_stmt, rows)
            inserted += len(batch)
            logger.debug("bulk_insert: inserted batch rows=%d total=%d", len(batch), inserted)

    conn.commit()
    return inserted


def row_count(conn: psycopg.Connection, table_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSON file into PostgreSQL with inferred column types."
    )
    parser.add_argument("json_file", type=Path, help="Path to a JSON file")
    parser.add_argument("--table", help="Override table name (default: file stem)")
    parser.add_argument("--drop-table", action="store_true")

    parser.add_argument("--dsn", default="")
    parser.add_argument("--host", default=os.getenv("POSTGRES_HOST"))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--connect-timeout", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: LOG_LEVEL env or INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    json_path = args.json_file.resolve()
    if not json_path.is_file():
        raise SystemExit(f"File not found: {json_path}")

    logger.info("main: loading json_file=%s", json_path)
    records = load_records(json_path)
    if not records:
        raise SystemExit("JSON file contains no records")

    table_name = args.table or sql_name_from_file(json_path)
    columns = infer_columns(records)
    logger.debug("main: inferred columns=%s", columns)

    dsn = resolve_connection(args)
    parsed = urlparse(dsn)
    logger.debug(
        "main: connecting user=%s host=%s port=%s db=%s",
        parsed.username or "",
        parsed.hostname or "",
        parsed.port or "",
        (parsed.path or "").lstrip("/"),
    )

    with psycopg.connect(dsn, connect_timeout=args.connect_timeout) as conn:
        version = conn.info.server_version
        major = version // 10000
        minor = (version % 10000) // 100
        patch = version % 100
        logger.info("main: connected postgres_version=%d.%d.%d", major, minor, patch)
        logger.info("main: table=%s records=%d", table_name, len(records))
        logger.info(
            "main: columns=%s",
            ", ".join(f"{sql_column_name(key)}:{pg_type}" for key, pg_type in columns.items()),
        )

        create_table(conn, table_name, columns, drop_existing=args.drop_table)

        logger.info(
            "bulk_insert: start table=%s records=%d chunk_size=%d",
            table_name,
            len(records),
            args.chunk_size,
        )
        inserted = bulk_insert(
            conn,
            table_name,
            records,
            columns,
            chunk_size=args.chunk_size,
        )
        count = row_count(conn, table_name)
        logger.info("bulk_insert: done inserted=%d row_count=%d", inserted, count)


if __name__ == "__main__":
    main()
