import logging

import numpy as np
import pandas as pd

from gd.config import INT_LOOKUP, FLOAT_LOOKUP, UINT_LOOKUP, BUFFER_LOOKUP


logger = logging.getLogger(__name__)


def hamming(a, b):
    """Calculated the hamming distance between two integers."""
    x = a ^ b
    dist = 0
    while x > 0:
        dist += x & type(x)(1)
        x = x >> type(x)(1)
    return dist


def cast_float_to_int(data, m):
    """
    If the dataset contains floating point data, this function will reinterpret the bits
    as integers and return the result.

    Args:
        data (array): Dataset to be processed.
        m (int): Bit depth.

    Returns:
        array: Dataset after reinterpretation
        bool: Indicator for whether dataset contained floating point data
    """
    float_indicator = False
    if type(data.item(0)) == float:
        logger.debug("Re-interpreting floating point data as integers.")
        float_indicator = True
        data_int = data.astype(FLOAT_LOOKUP[m])
        data_int = np.frombuffer(data_int.tobytes(), dtype=INT_LOOKUP[m])
        data_int = data_int.reshape(data.shape)
    else:
        data_int = data
    return data_int, float_indicator


def unpack_data(data, m):
    # TODO no longer necessary? replace with unpack_bits
    """
    Unpack an array of data into a binary array. For floating point data, there is also
    an intermediate reinterpretation of the underlying binary data as integer format.

    Args:
        data (array): Array to unpack.
        m (int): Bit depth. For integer data, this can be any positive integer. For
            floating point data, it can be either 32 or 64

    Returns:
        array: Unpacked data
        bool: Indicator for whether original data is floating point
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if type(data.item(0)) == float:
        data, float_indicator = cast_float_to_int(data, m)
    else:
        float_indicator = False
    data_unpack = np.ascontiguousarray(data, dtype=">u8")  # big endian
    data_unpack = data_unpack.view(">u1")  # split into individual bytes
    data_unpack = np.unpackbits(data_unpack).reshape((*data.shape, -1))  # unpack
    data_unpack = data_unpack[..., -m:].reshape((data.shape[0], -1))  # only m last bits
    return data_unpack, float_indicator


def get_compressed_size(n, d, n_b, dev_bits, m, counts=True):
    """
    Returns the estimated compressed data size for given compression parameters.

    Args:
        n (int): Number of samples.
        d (int): Dimensionality.
        n_b (int): Number of bases.
        dev_bits (list): List of bit-backed mask of deviation bits.
        m (list): Number of bits per value.
        counts (bool, optional): Whether to include base counts. Defaults to True.

    Returns:
        int: Estimated compressed data size in bytes.
        int: Estimated size of compressed data required for analytics in bytes.
    """
    len_d = np.sum(hamming(a, type(a)(0)) for a in dev_bits)
    len_b = np.sum(m) - len_d
    len_bc = get_min_bit_length(n)  # overestimated due to not knowing max base count
    len_id = get_min_bit_length(n_b)

    size_bases = n_b * len_b
    size_base_counts = counts * n_b * len_bc
    size_deviations = n * (len_d.astype(np.uint64) + len_id)  # ignores Elias Gamma encoding

    size_dev_bits = np.sum(m)
    size_delta_initial = 0  # TODO ignored for now, requires passing more info
    size_categories = 0  # TODO ignored for now, requires passing more info
    size_params = 16 * d + 16 + size_dev_bits + size_delta_initial + size_categories

    total_bits = size_bases + size_base_counts + size_deviations + size_params
    total_bits_analytics = total_bits - size_deviations
    size_compressed = np.ceil(total_bits / 8).astype(int)  # bytes
    size_analytics = np.ceil(total_bits_analytics / 8).astype(int)  # bytes
    return size_compressed, size_analytics


def get_nonzero_bit_idices(data, invert=False):
    """
    Returns a list containing the indices of non-zero bits in the given data. Data
    should be a list or 1-dimensional numpy array containing integer type data. The
    indices increment across elements in data. For example, the second bit in the third
    element will have index 2m + 1 (the first bit in the first element has index 0).
    Note that the high significance bits come first.

    Args:
        data (list): Data to identify non-zero indices.
        invert (bool, optional): Whether to invert the result and return the zero
            indices instead. Defaults to False.

    Raises:
        ValueError: Raised if data does not contain numpy types.

    Returns:
        list: List of non-zero bit indices in data.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, ndmin=1)
    try:
        m = data[0].nbytes * 8
    except AttributeError:
        raise ValueError("Data must be a numpy type.")

    # Get indices of non-zero bits
    d = len(data)
    mask = []
    powers_of_two = 2 ** np.arange(m, dtype=UINT_LOOKUP[m])[::-1]
    for j in range(d):
        mask.append((data[j] & powers_of_two).astype(bool))
    mask = np.concatenate(mask)
    if invert:
        return np.arange(d * m)[np.invert(mask)]
    else:
        return np.arange(d * m)[mask]


def get_min_bit_length(max_value):
    """Returns the minimum number of bits required per value to represent some data
    given the maximum value in the data."""
    return int(max(np.ceil(np.log2(1 + max_value)), 1))


def sample_from_chunks(chunks, sample_size, seed=100):
    """Returns a sample uniformly sampled from a collection of data chunks of
    potentially uneven size with uniform sampling ratio."""

    chunk_sizes = [chunk.shape[0] for chunk in chunks]
    total_rows = np.sum(chunk_sizes)

    # No sampling
    if (sample_size is None) or (sample_size >= total_rows):
        return pd.concat(chunks, axis=0, ignore_index=True)

    # Samples per chunk
    sample_ratio = sample_size / total_rows
    samples_per_chunk = [int(sample_ratio * s) for s in chunk_sizes]
    if np.sum(samples_per_chunk) < sample_size:
        diff = sample_size - np.sum(samples_per_chunk)
        samples_per_chunk = [s + (i < diff) for i, s in enumerate(samples_per_chunk)]

    # Build sample
    chunk_samples = []
    for i, chunk in enumerate(chunks):
        chunk_samples.append(
            chunk.sample(int(samples_per_chunk[i]), replace=False, random_state=seed)
        )

    return pd.concat(chunk_samples, axis=0, ignore_index=True)


def unpack_bits(data):
    # TODO docstring
    # must be a... dataframe (shape must work even if there is a single column)
    # for each column, unpack, convert to binary
    # combine columns
    # same assumptions about the input as for greedy_select

    if isinstance(data, np.ndarray) and (len(data.shape) == 1):
        data = np.expand_dims(data, axis=1)

    if isinstance(data, (list, pd.Series)):
        n = len(data)
        d = 1
    else:
        n, d = data.shape

    cols = []
    for i in range(d):
        # Convert to big endian
        if isinstance(data, pd.DataFrame):
            fmt = BUFFER_LOOKUP[str(data.dtypes.iloc[i])]
            col = np.ascontiguousarray(data.iloc[:, i], dtype=fmt)
        elif isinstance(data, pd.Series):
            fmt = BUFFER_LOOKUP[str(data.dtype)]
            col = np.ascontiguousarray(data, dtype=fmt)
        elif isinstance(data, list):
            fmt = BUFFER_LOOKUP[str(data[0].dtype)]
            col = np.ascontiguousarray(data, dtype=fmt)
        elif isinstance(data, np.ndarray):
            fmt = BUFFER_LOOKUP[str(data.dtype)]
            col = np.ascontiguousarray(data[:, i], dtype=fmt)
        else:
            raise RuntimeError(f"Invalid data type: {type(data)}")
        col = col.view(">u1")  # split into individual bytes
        col = np.unpackbits(col).reshape((n, -1))  # unpack
        col = col.astype(bool)  # boolean data type
        cols.append(col)
    data_unpacked = np.hstack(cols)
    return data_unpacked


def pack_bits(data, m):
    # TODO docstring
    # convert a binary array to an array of uint values with precision specified
    # per column by m
    # data must be a numpy binary array
    cols = []
    # output = pd.DataFrame()
    if not isinstance(m, list):
        m = [m]
    for i in range(len(m)):
        start_idx = np.sum(m[:i]).astype(int)
        end_idx = np.sum(m[: i + 1])
        col = np.frombuffer(
            np.packbits(data[:, start_idx:end_idx]).tobytes(),
            dtype=f">u{int(m[i]/8)}",
        )
        cols.append(pd.Series(col).astype(UINT_LOOKUP[m[i]]))
    output = pd.concat(cols, axis=1)
    return output


def get_base_bit_indices(dev_bits, drop_trailing_bits=0):
    """Returns the (global) indices of base bits."""
    index_mask = []
    for a in dev_bits:
        fmt = BUFFER_LOOKUP[str(a.dtype)]
        arr = np.ascontiguousarray([a], dtype=fmt)  # big endian
        sub_mask = np.invert(np.unpackbits(arr.view(">u1")).astype(bool))
        last_idx = np.argmin(sub_mask) - 1
        for i in range(min(drop_trailing_bits, last_idx)):
            sub_mask[last_idx - i] = False
        index_mask += list(sub_mask)
    return np.arange(len(index_mask))[index_mask]


def get_deviation_bit_indices(dev_bits, drop_trailing_bits=0):
    """Returns the (global) indices of deviation bits."""
    index_mask = []
    for a in dev_bits:
        fmt = BUFFER_LOOKUP[str(a.dtype)]
        arr = np.ascontiguousarray([a], dtype=fmt)  # big endian
        sub_mask = np.unpackbits(arr.view(">u1")).astype(bool)
        for i in range(min(drop_trailing_bits, np.argmax(sub_mask))):
            sub_mask[-(i + 1)] = False
        index_mask += list(sub_mask)
    return np.arange(len(index_mask))[index_mask]
