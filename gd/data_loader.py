import logging
import os
from math import ceil

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(filepath, schema, chunk_size=int(1e7), has_header=True, sep=","):
    """
    Loads a dataset as a list of non-overlapping and exhaustive chunks as DataFrames.

    Args:
        filepath (str): Path to dataset.
        schema (dict): Dataset schema.
        chunk_size (int): Number of rows per chunk.
        has_header (bool): Whether the data file contains column headers.
        sep (str): File delimiter.
    """
    logger.debug(f"Loading data from: {filepath}...")
    ext = os.path.splitext(filepath)[1]

    # CSV
    if ext in (".csv", ".txt"):
        logger.debug("Counting total rows of filepath...")
        f = open(filepath, "r")
        n = sum(1 for _ in f) - int(has_header)
        f.close()
        n_chunks = int(ceil(n / chunk_size))
        logger.debug(f"Total rows: {n}")
        logger.debug(f"Total chunks: {n_chunks} (chunk size={chunk_size})")

        data_chunks = []
        with pd.read_csv(
            filepath,
            sep=sep,
            header=0 if has_header else None,
            na_values=[""],
            chunksize=chunk_size,
            dtype={col_name: schema[col_name]["type"] for col_name in schema}
        ) as reader:
            for i, chunk in enumerate(reader):
                logger.debug(f"Loading chunk {i+1}/{n_chunks}")
                data_chunks.append(chunk)

    # Numpy
    elif ext == ".npy":
        np.load(filepath, mmap_mode="r")
        data = np.load(filepath, mmap_mode="r")
        if data.shape[0] > chunk_size:
            raise NotImplementedError("Cannot load numpy files in chunks.")
        else:
            data_chunks = [pd.DataFrame(np.load(filepath))]
            for i in range(data_chunks.shape[1]):
                data_chunks[i] = data_chunks[i].astype(schema["types"][i])

    # Other
    else:
        raise NotImplementedError(f"Loading of {ext} files not yet supported.")

    return data_chunks
