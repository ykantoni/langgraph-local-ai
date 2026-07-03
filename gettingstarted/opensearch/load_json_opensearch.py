#!/usr/bin/env python3
"""Load a JSON file into OpenSearch. Index name = file stem; mappings inferred from values."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from opensearchpy import OpenSearch, helpers

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
INDEX_NAME_RE = re.compile(r"[^a-z0-9._-]+")


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


def resolve_connection(args: argparse.Namespace) -> dict[str, Any]:
    url = (args.url or os.getenv("OPENSEARCH_URL", "")).strip()
    host = args.host or os.getenv("OPENSEARCH_HOST", "localhost")
    port = args.port or int(os.getenv("OPENSEARCH_PORT", "9200"))
    use_ssl = args.use_ssl if args.use_ssl is not None else env_bool("OPENSEARCH_USE_SSL", True)
    user = (
        args.user
        or os.getenv("OPENSEARCH_USERNAME", "")
        or os.getenv("OPENSEARCH_USER", "")
        or "admin"
    )
    password = args.password or os.getenv("OPENSEARCH_PASSWORD", "admin")

    logger.info("resolve_connection: host by os.getenv is %s", os.getenv("OPENSEARCH_HOST"))
    logger.info("resolve_connection: %s@%s:%s", user, host, str(port))

    if url:
        parsed = urlparse(url)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        if parsed.scheme:
            use_ssl = parsed.scheme == "https"
        if parsed.username:
            user = parsed.username
        if parsed.password:
            password = parsed.password

    return {
        "host": host,
        "port": port,
        "use_ssl": use_ssl,
        "user": user,
        "password": password,
        "verify_certs": args.verify_certs,
        "timeout": args.timeout,
    }


def create_client(cfg: dict[str, Any]) -> OpenSearch:
    http_auth = (cfg["user"], cfg["password"]) if cfg["user"] and cfg["password"] else None
    return OpenSearch(
        hosts=[{"host": cfg["host"], "port": cfg["port"]}],
        http_auth=http_auth,
        use_ssl=cfg["use_ssl"],
        verify_certs=cfg["verify_certs"],
        ssl_show_warn=cfg["verify_certs"],
        timeout=cfg["timeout"],
        max_retries=3,
        retry_on_timeout=True,
    )


def index_name_from_file(path: Path) -> str:
    name = INDEX_NAME_RE.sub("_", path.stem.lower()).strip("._-")
    if not name:
        raise ValueError(f"Cannot derive index name from file: {path}")
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


def _string_mapping(values: list[str]) -> dict[str, str]:
    if any(len(value) > 256 for value in values):
        return {"type": "text"}
    return {"type": "keyword"}


def infer_mappings(records: list[dict[str, Any]]) -> dict[str, Any]:
    field_kinds: dict[str, set[str]] = {}
    string_samples: dict[str, list[str]] = {}

    for record in records:
        for key, value in record.items():
            kind = _value_kind(value)
            if kind is None:
                continue
            field_kinds.setdefault(key, set()).add(kind)
            if kind == "string" and isinstance(value, str):
                string_samples.setdefault(key, []).append(value)

    properties: dict[str, Any] = {}
    for key, kinds in field_kinds.items():
        merged = _merge_kinds(kinds)
        if merged == "boolean":
            properties[key] = {"type": "boolean"}
        elif merged == "long":
            properties[key] = {"type": "long"}
        elif merged == "double":
            properties[key] = {"type": "double"}
        elif merged == "date":
            properties[key] = {"type": "date"}
        elif merged == "object":
            properties[key] = {"type": "object", "enabled": True}
        elif merged == "keyword":
            properties[key] = {"type": "keyword"}
        else:
            properties[key] = _string_mapping(string_samples.get(key, []))

    return {"properties": properties}


def create_index(
    client: OpenSearch,
    index_name: str,
    mappings: dict[str, Any],
    *,
    delete_existing: bool,
) -> None:
    exists = client.indices.exists(index=index_name)
    if exists and delete_existing:
        logger.info("create_index: deleting existing index=%s", index_name)
        client.indices.delete(index=index_name)
        exists = False

    if exists:
        logger.info("create_index: index already exists=%s", index_name)
        return

    logger.info("create_index: creating index=%s", index_name)
    client.indices.create(
        index=index_name,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": mappings,
        },
    )


def bulk_index(
    client: OpenSearch,
    index_name: str,
    records: list[dict[str, Any]],
    *,
    chunk_size: int,
    request_timeout: int,
) -> tuple[int, list[Any]]:
    actions = (
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": str(i),
            "_source": record,
        }
        for i, record in enumerate(records)
    )
    return helpers.bulk(
        client,
        actions,
        chunk_size=chunk_size,
        request_timeout=request_timeout,
        raise_on_error=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSON file into OpenSearch with inferred mappings."
    )
    parser.add_argument("json_file", type=Path, help="Path to a JSON file")
    parser.add_argument("--index", help="Override index name (default: file stem)")
    parser.add_argument("--delete-index", action="store_true")

    parser.add_argument("--url", default=os.getenv("OPENSEARCH_URL", ""))
    parser.add_argument("--host", default=os.getenv("OPENSEARCH_HOST"))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--use-ssl",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--verify-certs",
        action=argparse.BooleanOptionalAction,
        default=env_bool("OPENSEARCH_VERIFY_CERTS", False),
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--request-timeout", type=int, default=120)
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

    index_name = args.index or index_name_from_file(json_path)
    mappings = infer_mappings(records)
    logger.debug("main: inferred mappings=%s", mappings)

    cfg = resolve_connection(args)
    logger.debug(
        "main: connecting host=%s port=%s use_ssl=%s user=%s",
        cfg["host"],
        cfg["port"],
        cfg["use_ssl"],
        cfg["user"],
    )
    client = create_client(cfg)

    version = client.info().get("version", {}).get("number", "unknown")
    logger.info("main: connected opensearch_version=%s", version)
    logger.info("main: index=%s records=%d", index_name, len(records))
    logger.info(
        "main: fields=%s",
        ", ".join(sorted(mappings["properties"])),
    )

    create_index(client, index_name, mappings, delete_existing=args.delete_index)

    logger.info(
        "bulk_index: start index=%s records=%d chunk_size=%d",
        index_name,
        len(records),
        args.chunk_size,
    )
    ok_count, errors = bulk_index(
        client,
        index_name,
        records,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
    )

    client.indices.refresh(index=index_name)
    count = client.count(index=index_name)["count"]

    logger.info("bulk_index: done indexed=%d document_count=%d", ok_count, count)
    if errors:
        logger.warning("bulk_index: %d error(s); showing first %d", len(errors), min(5, len(errors)))
        for error in errors[:5]:
            logger.warning("bulk_index: %s", error)


if __name__ == "__main__":
    main()
