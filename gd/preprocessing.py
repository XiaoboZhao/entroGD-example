import logging
from datetime import date, timedelta, time

import numpy as np
import pandas as pd

from gd.config import UINT_LOOKUP

logger = logging.getLogger(__name__)

NON_PREPROCESSING = ["gd-infocom", "gd-glean", "gd-d"]
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
TIME_FORMAT_MICRO = "%H:%M:%S.%f0"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIMESTAMP_FORMAT_MICRO = "%Y-%m-%dT%H:%M:%S.%f"


def time_to_float(time):
    """Converts a time string to seconds past midnight as an integer."""
    total_seconds = (
        time.second
        + time.minute * 60
        + time.hour * 3600
        + time.microsecond * 1e-6
    )
    return total_seconds


def float_to_time(s, fmt):
    """Converts seconds past midnight into a time as a string."""
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    microseconds = round((seconds % 1) * (10 ** 6))
    seconds = int(seconds)
    return time(int(hours), int(minutes), seconds, microseconds).strftime(fmt)


def preprocess_chunk(chunk, gd_version, schema):
    """
    Preprocess a single chunk of data. Encoding categorical values, minimum
    subtraction and type selection are postponed until all chunks are processed.

    Args:
        chunk (DataFrame): Data chunk.
        gd_version (str): GD compressor version.
        schema (dict): Column data types and precisions.

    Returns:
        DataFrame: Processed data chunk.
        dict: Categorical value map.
        list: NA value indicators.
        list: NA value indices.
    """
    n, d = chunk.shape

    # GD versions that do not include pre-processing
    if gd_version in NON_PREPROCESSING:
        for c in chunk.columns:
            buff_type = UINT_LOOKUP[chunk[c].dtype.itemsize * 8]
            chunk[c] = np.frombuffer(chunk[c].to_numpy().tobytes(), dtype=buff_type)
            chunk[c] = chunk[c]
        categorical_maps = {}
        na_indicators = np.zeros(d)
        na_idxs = []

    # Other GD versions
    else:
        categorical_maps = {}
        na_indicators = []
        na_idxs = {}
        for i, c in enumerate(chunk.columns):
            col_dtype = schema[c]["type"]
            scaling_factor = 10 ** schema[c]["precision"]

            # Check for missing values and encode them with a placeholder value
            na_idxs[c] = np.where(chunk[c].isna())[0]
            na_indicators.append(any(na_idxs[c]))
            if na_indicators[i]:  # temporary encoding with dummy value
                if col_dtype == "categorical":
                    na_placeholder = "nan"
                else:
                    na_placeholder = chunk.loc[chunk[c].first_valid_index(), c]
                chunk.iloc[na_idxs[c], c] = na_placeholder

            # Type-specific transformations
            if col_dtype in ["id"]:
                chunk[c] = np.uint64(chunk[c])
            elif col_dtype in ["date"]:
                # Convert to days since epoch
                chunk[c] = pd.to_datetime(chunk[c], format=DATE_FORMAT)
                chunk[c] = (chunk[c] - pd.Timestamp(1970, 1, 1)).dt.days
            elif col_dtype in ["timestamp"]:
                # Convert to seconds since epoch then scale based on precision
                chunk[c] = pd.to_datetime(chunk[c], format=TIMESTAMP_FORMAT)
                chunk[c] = (chunk[c] - pd.Timestamp(1970, 1, 1)).dt.total_seconds()
                chunk[c] = np.int64((chunk[c] * scaling_factor).round())
            elif col_dtype in ["timestamp_micro"]:
                # Convert to seconds since epoch then scale based on precision
                chunk[c] = pd.to_datetime(chunk[c], format=TIMESTAMP_FORMAT_MICRO)
                chunk[c] = (chunk[c] - pd.Timestamp(1970, 1, 1)).dt.total_seconds()
                chunk[c] = np.int64((chunk[c] * scaling_factor).round())
            elif col_dtype in ["time"]:
                # Convert to seconds since midnight then scale based on precision
                chunk[c] = pd.to_datetime(chunk[c], format=TIME_FORMAT)
                chunk[c] = (
                    chunk[c].dt.second
                    + chunk[c].dt.minute * 60
                    + chunk[c].dt.hour * 3600
                    + chunk[c].dt.microsecond * 1e-6
                )
                chunk[c] = np.int64((chunk[c] * scaling_factor).round())
            elif col_dtype in ["time_micro"]:
                # Convert to seconds since midnight then scale based on precision
                chunk[c] = pd.to_datetime(chunk[c], format=TIME_FORMAT_MICRO)
                chunk[c] = (
                    chunk[c].dt.second
                    + chunk[c].dt.minute * 60
                    + chunk[c].dt.hour * 3600
                    + chunk[c].dt.microsecond * 1e-6
                )
                chunk[c] = np.uint64((chunk[c] * scaling_factor).round())
            elif col_dtype == "categorical":
                # Idea: encode each categorical value with a unique integer. For
                # better compression, the categories are sorted from most to least
                # frequent so that smaller integers are assigned to the most common
                # categories and there is less variation in higher order bits.
                map = chunk[c].groupby(chunk[c], sort=False, dropna=False).count()
                categorical_maps[c] = map.to_dict()
            elif col_dtype in [np.float32, np.float64]:
                # Idea: Scale floats up to integers, which typically results in more
                # constant bits.
                chunk[c] = np.int64((chunk[c].astype(float) * scaling_factor).round())
            elif col_dtype in [np.int8, np.int16, np.int32, np.int64]:
                chunk[c] = np.int64(chunk[c])
            elif col_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
                chunk[c] = np.uint64(chunk[c])
            else:  # e.g. string, text
                raise ValueError(f"Invalid data type: {schema[c]['type']}")

    return chunk, categorical_maps, na_indicators, na_idxs


def preprocess(data_chunks, gd_version, schema=None, minimums=None, data_range=None):
    """
    Apply pre-processing to the data data_chunks.

    Args:
        data_chunks (list): List of DataFrames containing the data in data_chunks.
        gd_version (str): GD compressor version.
        schema (dict, optional): Data schema.
        minimums (list, optional): Minimum values in each column.
        range (list, optional): Maximum dynamic range that should be accommodated for
            each column. This is used to set the UINT data type.

    Returns:
        list: List of data data_chunks.
    """
    # TODO derive data types automatically
    if schema is None:
        raise NotImplementedError()
    n_chunks = len(data_chunks)

    # Preprocess each chunk
    categorical_maps = [None] * n_chunks
    na_indicators = [None] * n_chunks
    na_idxs = [None] * n_chunks
    for i in range(n_chunks):
        logger.debug(f"Preprocessing chunk {i+1}/{n_chunks}")
        (
            data_chunks[i],
            categorical_maps[i],
            na_indicators[i],
            na_idxs[i],
        ) = preprocess_chunk(data_chunks[i], gd_version, schema)
    # NOTE: I removed multiprocessing here as it was causing memory issues
    # with multiprocessing.Pool() as pool:
    #     data_chunks = pool.starmap(
    #         GD.Preprocessing, zip(data_chunks, repeat(data_types))
    #     )

    logger.debug("Consolidating preprocessing across data_chunks...")

    # Consolidate missing value indicators
    na_indicators = np.all(na_indicators, axis=0)

    # Consolidate and apply categorical value maps
    categorical_maps_all = {}
    for map_dict in categorical_maps:
        for col, map in map_dict.items():
            if col not in categorical_maps_all:
                categorical_maps_all[col] = {}
            for val, count in map.items():
                try:
                    categorical_maps_all[col][val] += count
                except KeyError:
                    categorical_maps_all[col][val] = count
    for col, map in categorical_maps_all.items():
        map = pd.Series(map).sort_values(ascending=False)
        map = {v: c for c, v in enumerate(map.index)}
        for c in range(n_chunks):
            data_chunks[c][col] = data_chunks[c][col].map(map)
        categorical_maps_all[col] = list(map.keys())

    # Minimum value subtraction
    # Subtract lowest value and store using the smallest suitable uint type
    # Reserve one bit for indicating missing/NaN values if present.
    minimum_values_all = []
    for i, c in enumerate(data_chunks[0].columns):
        # Subtract global minimum
        if minimums:
            min_value = minimums[c]
        else:
            min_value = min([chunk[c].min() for chunk in data_chunks])
        for ii in range(n_chunks):
            data_chunks[ii][c] = data_chunks[ii][c] - min_value
        minimum_values_all.append(np.int64(min_value))

        # Compute max value for later use before converting missing values
        # NOTE: This is shifted one bit up if there are missing values to reserve
        # the maximum value to indicate them.
        if data_range is None:
            max_value = max([chunk[c].max() for chunk in data_chunks])
            max_value = max_value << na_indicators[i]
        else:
            max_value = data_range[c]

        # Map missing/NaN values to the maximum value for the data type
        if na_indicators[i] and (schema[c]["type"] != "categorical"):
            for ii in range(n_chunks):
                data_chunks[ii].iloc[na_idxs[ii][c], c] = np.iinfo(
                    data_chunks[ii][c].dtype
                ).max

        # Cast to smallest suitable uint data type, reserving one bit for indicating
        # missing/NaN values if present
        if max_value < (2**8):
            dtype = np.uint8
        elif max_value < (2**16):
            dtype = np.uint16
        elif max_value < (2**32):
            dtype = np.uint32
        else:
            dtype = np.uint64
        for ii in range(n_chunks):
            data_chunks[ii][c] = data_chunks[ii][c].astype(dtype)

    return data_chunks, minimum_values_all, categorical_maps_all, na_indicators


def preprocess_revert(
    data,
    schema,
    minimum_values,
    categorical_maps,
    na_indicators,
    column_names,
):
    assert len(schema) == data.shape[1]

    for i in range(len(column_names)):
        col_dtype = schema[i]["type"]
        scaling_factor = 10 ** schema[i]["precision"].astype(np.uint64)

        # Temporary encoding of missing/NaN with dummy value
        if na_indicators[i]:
            na_idxs = np.where(data[i] == data[i].max())[0]
            data.iloc[na_idxs, i] = data[i].min()

        # Convert to larger type and add minimum value
        if col_dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
            data[i] = data[i].astype(np.int64) + minimum_values[i]
        else:
            data[i] = data[i].astype(np.uint64) + np.uint64(minimum_values[i])

        # Undo type-specific preprocessing
        if col_dtype in ["id"]:
            pass
        elif col_dtype in ["date"]:
            data[i] = data[i].apply(
                lambda x: (date(1970, 1, 1) + timedelta(days=x)).isoformat()
            )
        elif col_dtype in ["timestamp"]:
            data[i] = data[i].astype(float) / scaling_factor
            data[i] = pd.to_datetime(data[i], unit="s")
            data[i] = data[i].dt.strftime(TIMESTAMP_FORMAT)
        elif col_dtype in ["timestamp_micro"]:
            data[i] = data[i].astype(float) / scaling_factor
            data[i] = pd.to_datetime(data[i], unit="s")
            data[i] = data[i].dt.strftime(TIMESTAMP_FORMAT_MICRO)
        elif col_dtype in ["time"]:
            data[i] = data[i].astype(float) / scaling_factor
            data[i] = data[i].apply(lambda x: float_to_time(x, TIME_FORMAT))
        elif col_dtype in ["time_micro"]:
            data[i] = data[i].astype(float) / scaling_factor
            data[i] = data[i].apply(lambda x: float_to_time(x, TIME_FORMAT_MICRO))
        elif col_dtype in ["categorical"]:
            map = {i: s for i, s in enumerate(categorical_maps[i])}
            data[i] = data[i].map(map)
        elif col_dtype in [np.float32, np.float64]:
            data[i] = (data[i] / scaling_factor).astype(col_dtype)
        elif col_dtype in [np.int8, np.int16, np.int32, np.int64]:
            data[i] = data[i].astype(col_dtype)
        elif col_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            data[i] = data[i].astype(col_dtype)
        else:  # e.g. string, text
            raise ValueError(f"Invalid data type: {schema[i]['type']}")

        # Restore missing/NaN values
        if na_indicators[i]:
            data.iloc[na_idxs, i] = pd.NA

    data.columns = column_names
    return data
