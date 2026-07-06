#!/usr/bin/env python3
"""Load JSON into OpenSearch. Single file or join two files and embed lookup records."""

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


def _join_key(value: Any) -> str:
    return str(value)


def join_records(
    base_records: list[dict[str, Any]],
    lookup_records: list[dict[str, Any]],
    *,
    join_on: str,
    embed_as: str,
) -> list[dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    duplicate_keys = 0
    skipped_lookup = 0

    for record in lookup_records:
        key_value = record.get(join_on)
        if key_value is None:
            skipped_lookup += 1
            continue
        norm_key = _join_key(key_value)
        if norm_key in lookup:
            duplicate_keys += 1
        lookup[norm_key] = record

    if duplicate_keys:
        logger.warning(
            "join_records: %d duplicate %r keys in lookup file; last record wins",
            duplicate_keys,
            join_on,
        )
    if skipped_lookup:
        logger.warning(
            "join_records: skipped %d lookup records with missing %r",
            skipped_lookup,
            join_on,
        )

    joined: list[dict[str, Any]] = []
    missing_matches = 0
    missing_keys = 0
    embedded_count = 0

    for record in base_records:
        doc = dict(record)
        key_value = record.get(join_on)
        if key_value is None:
            missing_keys += 1
            joined.append(doc)
            continue

        match = lookup.get(_join_key(key_value))
        if match is None:
            missing_matches += 1
            joined.append(doc)
            continue

        doc[embed_as] = dict(match)
        embedded_count += 1
        joined.append(doc)

    logger.info(
        "join_records: base=%d lookup=%d embedded=%d missing_match=%d missing_key=%d",
        len(base_records),
        len(lookup_records),
        embedded_count,
        missing_matches,
        missing_keys,
    )
    return joined


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


def _text_keyword_mapping(*, ignore_above: int = 256) -> dict[str, Any]:
    return {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword",
                "ignore_above": ignore_above,
            },
        },
    }


def _string_mapping(values: list[str]) -> dict[str, Any]:
    if any(len(value) > 256 for value in values):
        return _text_keyword_mapping(ignore_above=256)
    return _text_keyword_mapping()


def _mapping_for_merged_kind(merged: str, string_samples: list[str]) -> dict[str, Any]:
    if merged == "boolean":
        return {"type": "boolean"}
    if merged == "long":
        return {"type": "long"}
    if merged == "double":
        return {"type": "double"}
    if merged == "date":
        return {"type": "date"}
    if merged == "keyword":
        return _text_keyword_mapping()
    return _string_mapping(string_samples)


def _infer_field_mapping(values: list[Any]) -> dict[str, Any]:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return _text_keyword_mapping()

    dict_values = [value for value in non_null if isinstance(value, dict)]
    kinds: set[str] = set()
    string_samples: list[str] = []

    for value in non_null:
        if isinstance(value, dict):
            kinds.add("object")
            continue
        kind = _value_kind(value)
        if kind is None:
            continue
        kinds.add(kind)
        if kind == "string" and isinstance(value, str):
            string_samples.append(value)

    merged = _merge_kinds(kinds)
    if merged == "object" and dict_values:
        return {
            "type": "nested",
            "properties": _infer_properties(dict_values),
        }

    return _mapping_for_merged_kind(merged, string_samples)


def _infer_properties(objects: list[dict[str, Any]]) -> dict[str, Any]:
    field_values: dict[str, list[Any]] = {}
    for obj in objects:
        for key, value in obj.items():
            field_values.setdefault(key, []).append(value)

    return {
        key: _infer_field_mapping(values)
        for key, values in field_values.items()
    }


def infer_mappings(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"properties": _infer_properties(records)}


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
        description="Load JSON into OpenSearch with inferred mappings."
    )
    parser.add_argument("json_file", type=Path, help="Primary JSON file (base records)")
    parser.add_argument(
        "--index",
        help="Target index name (required with --join-file; default: primary file stem)",
    )
    parser.add_argument("--delete-index", action="store_true")

    join_group = parser.add_argument_group("join and embed")
    join_group.add_argument(
        "--join-file",
        type=Path,
        help="Secondary JSON file whose records are embedded into base records",
    )
    join_group.add_argument(
        "--join-on",
        help="Field name used to join base and lookup records (e.g. customer_id)",
    )
    join_group.add_argument(
        "--embed-as",
        help="Nested field name for embedded lookup record (default: join file stem)",
    )

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


def resolve_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    json_path = args.json_file.resolve()
    if not json_path.is_file():
        raise SystemExit(f"File not found: {json_path}")

    logger.info("resolve_records: loading base json_file=%s", json_path)
    records = load_records(json_path)
    if not records:
        raise SystemExit("Primary JSON file contains no records")

    if args.join_file:
        if not args.join_on:
            raise SystemExit("--join-on is required when --join-file is set")
        if not args.index:
            raise SystemExit("--index is required when --join-file is set")

        join_path = args.join_file.resolve()
        if not join_path.is_file():
            raise SystemExit(f"Join file not found: {join_path}")

        embed_as = args.embed_as or index_name_from_file(join_path)
        logger.info(
            "resolve_records: joining join_file=%s join_on=%r embed_as=%r",
            join_path,
            args.join_on,
            embed_as,
        )
        lookup_records = load_records(join_path)
        if not lookup_records:
            raise SystemExit("Join JSON file contains no records")

        records = join_records(
            records,
            lookup_records,
            join_on=args.join_on,
            embed_as=embed_as,
        )
        index_name = args.index
    else:
        index_name = args.index or index_name_from_file(json_path)

    return records, index_name


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    records, index_name = resolve_records(args)
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
