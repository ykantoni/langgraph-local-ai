#!/usr/bin/env python3

import argparse
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm


DEFAULT_DATASET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"Dataset already exists: {target}")
        return

    print(f"Downloading dataset:\n  {url}\n  -> {target}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(target, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="download",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))


def normalize_value(value: Any) -> Any:
    """
    Convert pandas/numpy values to JSON-safe Python values.
    OpenSearch documents must be serializable to JSON.
    """
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def dataframe_to_bulk_actions(
    df: pd.DataFrame,
    index_name: str,
    id_start: int,
) -> Iterable[Dict[str, Any]]:
    columns = list(df.columns)

    for offset, row in enumerate(df.itertuples(index=False, name=None)):
        doc = {
            col: normalize_value(value)
            for col, value in zip(columns, row)
        }

        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": str(id_start + offset),
            "_source": doc,
        }


def create_client(args: argparse.Namespace) -> OpenSearch:
    http_auth = None
    if args.user and args.password:
        http_auth = (args.user, args.password)

    return OpenSearch(
        hosts=[{"host": args.host, "port": args.port}],
        http_auth=http_auth,
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        ssl_show_warn=args.verify_certs,
        timeout=args.timeout,
        max_retries=3,
        retry_on_timeout=True,
    )


def create_index(client: OpenSearch, index_name: str, delete_existing: bool) -> None:
    exists = client.indices.exists(index=index_name)

    if exists and delete_existing:
        print(f"Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)
        exists = False

    if exists:
        print(f"Index already exists: {index_name}")
        return

    print(f"Creating index: {index_name}")

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        "mappings": {
            "properties": {
                "VendorID": {"type": "integer"},
                "tpep_pickup_datetime": {"type": "date"},
                "tpep_dropoff_datetime": {"type": "date"},
                "passenger_count": {"type": "float"},
                "trip_distance": {"type": "float"},
                "RatecodeID": {"type": "float"},
                "store_and_fwd_flag": {"type": "keyword"},
                "PULocationID": {"type": "integer"},
                "DOLocationID": {"type": "integer"},
                "payment_type": {"type": "integer"},
                "fare_amount": {"type": "float"},
                "extra": {"type": "float"},
                "mta_tax": {"type": "float"},
                "tip_amount": {"type": "float"},
                "tolls_amount": {"type": "float"},
                "improvement_surcharge": {"type": "float"},
                "total_amount": {"type": "float"},
                "congestion_surcharge": {"type": "float"},
                "Airport_fee": {"type": "float"},
            }
        },
    }

    client.indices.create(index=index_name, body=body)


def finalize_index(client: OpenSearch, index_name: str) -> None:
    print("Restoring refresh interval and refreshing index...")

    client.indices.put_settings(
        index=index_name,
        body={
            "index": {
                "refresh_interval": "1s"
            }
        },
    )

    client.indices.refresh(index=index_name)


def load_parquet_to_opensearch(
    client: OpenSearch,
    parquet_path: Path,
    index_name: str,
    max_rows: int,
    read_batch_rows: int,
    bulk_chunk_size: int,
    request_timeout: int,
) -> None:
    parquet_file = pq.ParquetFile(parquet_path)

    attempted = 0
    succeeded = 0
    first_errors = []

    print(f"Loading up to {max_rows:,} rows into index '{index_name}'")

    with tqdm(total=max_rows, unit="rows", desc="index") as progress:
        for batch in parquet_file.iter_batches(batch_size=read_batch_rows):
            if attempted >= max_rows:
                break

            df = batch.to_pandas()

            remaining = max_rows - attempted
            if len(df) > remaining:
                df = df.iloc[:remaining]

            actions = dataframe_to_bulk_actions(
                df=df,
                index_name=index_name,
                id_start=attempted,
            )

            ok_count, errors = helpers.bulk(
                client,
                actions,
                chunk_size=bulk_chunk_size,
                request_timeout=request_timeout,
                max_retries=3,
                initial_backoff=2,
                max_backoff=30,
                raise_on_error=False,
            )

            attempted += len(df)
            succeeded += ok_count
            progress.update(len(df))

            if errors and len(first_errors) < 5:
                first_errors.extend(errors[:5])

    print()
    print(f"Attempted rows: {attempted:,}")
    print(f"Successfully indexed: {succeeded:,}")

    if first_errors:
        print()
        print("First indexing errors:")
        for error in first_errors[:5]:
            print(error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ~1M NYC TLC Yellow Taxi rows into OpenSearch."
    )

    parser.add_argument("--host", default=os.getenv("OPENSEARCH_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))

    parser.add_argument("--user", default=os.getenv("OPENSEARCH_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("OPENSEARCH_PASSWORD", "admin"))

    parser.add_argument(
        "--use-ssl",
        action=argparse.BooleanOptionalAction,
        default=env_bool("OPENSEARCH_USE_SSL", True),
    )
    parser.add_argument(
        "--verify-certs",
        action=argparse.BooleanOptionalAction,
        default=env_bool("OPENSEARCH_VERIFY_CERTS", False),
    )

    parser.add_argument("--index", default="nyc-yellow-taxi-1m")
    parser.add_argument("--delete-index", action="store_true")

    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-file", default="yellow_tripdata_2024-01.parquet")

    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--read-batch-rows", type=int, default=50_000)
    parser.add_argument("--bulk-chunk-size", type=int, default=5_000)

    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=int, default=120)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_file)

    download_file(args.dataset_url, dataset_path)

    client = create_client(args)

    info = client.info()
    print(f"Connected to OpenSearch: {info.get('version', {}).get('number')}")

    create_index(
        client=client,
        index_name=args.index,
        delete_existing=args.delete_index,
    )

    load_parquet_to_opensearch(
        client=client,
        parquet_path=dataset_path,
        index_name=args.index,
        max_rows=args.max_rows,
        read_batch_rows=args.read_batch_rows,
        bulk_chunk_size=args.bulk_chunk_size,
        request_timeout=args.request_timeout,
    )

    finalize_index(client, args.index)

    count = client.count(index=args.index)
    print(f"OpenSearch _count result: {count['count']:,}")


if __name__ == "__main__":
    main()