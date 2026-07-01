"""
Command-line EntroGD compression wrapper.

This wrapper keeps EntroGD-specific selection in this repository while using
the local GreedyGD repository as the canonical GD backend.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


DTYPE_MAP = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}


DEFAULT_GREEDYGD_ROOT = (
    Path(__file__).resolve().parents[1] / "aaron-gd-implementation"
)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    schema_json = Path(args.schema_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    greedygd_root = Path(args.greedygd_root).resolve()
    sys.path.insert(0, str(greedygd_root))

    from gd.config import LOG_FORMAT
    from gd.GD import GD
    from gd.data_loader import load_dataset

    entro_select = load_local_entro_select().entro_select

    schema = load_schema_json(schema_json)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = output_dir / f"out_{timestamp}.log"
    setup_logging(log_path, LOG_FORMAT)
    logger = logging.getLogger()

    n_rows = count_csv_rows(input_csv)
    bytes_per_row = sum(np.dtype(schema[c]["type"]).itemsize for c in schema)
    size_uncompressed = n_rows * bytes_per_row

    logger.info(f"Total rows:    {n_rows:,}")
    logger.info(f"Bytes per row: {bytes_per_row}")
    logger.info("Starting compression...")

    started = perf_counter()
    GD.compress(
        data=str(input_csv),
        schema=schema,
        has_header=True,
        out_path=str(output_dir),
        export_counts=True,
        export_synopsis=True,
        gd_version="gd-entro",
        config_sample_size=args.training_size,
        selection_fn=entro_select,
        selection_kwargs={
            "entropy": None,
            "tau": args.tau,
        },
    )
    runtime_s = perf_counter() - started
    logger.info(f"gd-entro compression finished in {runtime_s} seconds.")

    output_stem = input_csv.stem
    compressed_path = output_dir / f"{output_stem}.gd"
    bases_path = output_dir / f"{output_stem}_bases.gd"

    if args.check_decompression:
        logger.info("Checking that decompression is lossless...")
        data = pd.concat(load_dataset(str(input_csv), schema))
        decompressed_data = GD.decompress(filepath=str(compressed_path))
        logger.info(f"Lossless compression: {data.equals(decompressed_data)}")
        if not data.equals(decompressed_data):
            logger.error(f"Difference:\n{data.compare(decompressed_data)}")

    size_compressed = os.path.getsize(compressed_path)
    size_bases = os.path.getsize(bases_path)
    cr = size_compressed / size_uncompressed
    cr_bases = size_bases / size_uncompressed

    logger.info(f"Uncompressed size:            {size_uncompressed} bytes")
    logger.info(f"Compressed size:              {size_compressed} bytes")
    logger.info(f"Compression ratio:            {cr:.3f}")
    logger.info(f"Compression ratio (inverse):  {1 / cr:.3f}")

    metadata_path = output_dir / f"out_{timestamp}_gd_metadata.txt"
    write_metadata(
        metadata_path=metadata_path,
        input_csv=input_csv,
        greedygd_root=greedygd_root,
        training_size=args.training_size,
        tau=args.tau,
        runtime_s=runtime_s,
        size_uncompressed=size_uncompressed,
        size_compressed=size_compressed,
        size_bases=size_bases,
        cr=cr,
        cr_bases=cr_bases,
    )
    logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--schema-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--greedygd-root",
        type=Path,
        default=DEFAULT_GREEDYGD_ROOT,
        help="Path to the local GreedyGD repository used as the GD backend.",
    )
    parser.add_argument("--tau", type=int, default=10)
    parser.add_argument("--training-size", type=int, default=1_000_000)
    parser.add_argument("--check-decompression", action="store_true")
    return parser.parse_args()


def setup_logging(log_path: Path, log_format: str) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[file_handler, console_handler],
    )


def load_schema_json(path: Path) -> dict:
    schema = json.loads(path.read_text())
    for column_info in schema.values():
        column_info["type"] = DTYPE_MAP[column_info["type"]]
    return schema


def load_local_entro_select():
    module_path = Path(__file__).resolve().parent / "gd" / "entro_select.py"
    spec = importlib.util.spec_from_file_location(
        "entrogd_local_entro_select",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_csv_rows(path: Path) -> int:
    with path.open("r") as handle:
        return sum(1 for _ in handle) - 1


def write_metadata(
    metadata_path: Path,
    input_csv: Path,
    greedygd_root: Path,
    training_size: int,
    tau: int,
    runtime_s: float,
    size_uncompressed: int,
    size_compressed: int,
    size_bases: int,
    cr: float,
    cr_bases: float,
) -> None:
    with metadata_path.open("w") as handle:
        handle.write("------------- Parameters -------------\n")
        handle.write(f"DATA                      {input_csv}\n")
        handle.write("GD_VERSION                gd-entro\n")
        handle.write(f"GREEDYGD_ROOT             {greedygd_root}\n")
        handle.write(f"GD_TRAINING_SIZE          {training_size}\n")
        handle.write(f"TAU                       {tau}\n")

        handle.write("\n------------- Runtime -------------\n")
        handle.write(f"Compression (total)              {runtime_s:.3f} s\n")

        handle.write("\n------------- Storage -------------\n")
        handle.write(f"Uncompressed size:        {size_uncompressed} (bytes)\n")
        handle.write(f"Compressed size (total):  {size_compressed} (bytes)\n")
        handle.write(f"Compressed size (bases):  {size_bases} (bytes)\n")
        handle.write(f"CR:                       {cr:.3f}\n")
        handle.write(f"CR (inverse):             {1 / cr:.3f}\n")
        handle.write(f"CR (bases):               {cr_bases:.3f}\n")
        handle.write(f"CR (bases, inverse):      {1 / cr_bases:.3f}\n")


if __name__ == "__main__":
    main()
