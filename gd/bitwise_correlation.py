"""
Functions for configuring default GD and analytics-tailored GD using correlation bit
ordering. Notes:
 - Always assume c = 1 (no sample concatenation)
 - Always restrict sign and exponent bits
"""

import logging

import numpy as np
import pandas as pd

from gd.config import N_EXPONENT_BITS_LOOKUP, UINT_LOOKUP
from gd.BaseLeaves import BaseLeaves
from gd.common import get_compressed_size, unpack_bits, unpack_data

logger = logging.getLogger(__name__)


def get_bit_ordering(data, bit_restrictions=False):
    """
    Returns bit indices ordered according to maximum inter-bit correlations. All bits
    with zero variance (i.e. that are always the same) are put first, followed by the
    remaining bits in order of decreasing inter-bit correlation.

    Args:
        data (DataFrame): Data to determine bit ordering for.
        bit_restrictions (bool, optional): Whether to restrict sign and exponent (for
            floating point data) bits to the base. Defaults to False.

    Raises:
        RuntimeError: Raised if bit ordering length does not match sample length.

    Returns:
        list: Ordering of bit indices.
        int: Number of restricted bits.
    """
    # Setup
    m = [dtype.itemsize * 8 for dtype in data.dtypes]  # number of bits for each value
    float_indicator = any([dtype in (np.float32, np.float64) for dtype in data.dtypes])
    logger.debug("Dataset shape: {:s}".format(str(data.shape)))
    logger.debug(f"Bits per value: {m}")
    bit_ordering = []
    if float_indicator:
        data_unpack, _ = unpack_data(data, max(m))
    else:
        data_unpack = unpack_bits(data)
    logger.debug("Dataset unpacked shape: {:s}".format(str(data_unpack.shape)))

    # Get restricted bit indices
    n_restricted_bits = 0
    if bit_restrictions and any([dt in (np.float32, np.float64) for dt in data.dtypes]):
        d = data.shape[1]
        n_restricted_bits = 1 + N_EXPONENT_BITS_LOOKUP[max(m)] * float_indicator
        restricted_bit_idxs = np.tile(np.arange(n_restricted_bits), d)
        restricted_bit_idxs += np.repeat(np.arange(d) * m, n_restricted_bits)
        bit_ordering = restricted_bit_idxs.tolist()
        logger.debug(
            "Restricted bits total: %d, indices: %s", len(bit_ordering), bit_ordering
        )

    # Add zero variance bits
    all_zero_bits = ~np.bitwise_or.reduce(data_unpack, dtype=bool)
    all_one_bits = np.bitwise_and.reduce(data_unpack, dtype=bool)
    zero_var_bits = all_zero_bits | all_one_bits
    zero_var_bit_idxs = np.arange(data_unpack.shape[1])[zero_var_bits]
    bit_ordering += [idx for idx in zero_var_bit_idxs if idx not in bit_ordering]
    logger.debug(
        "Restricted & zero total: %d, indices: %s", len(bit_ordering), bit_ordering
    )

    # Order remaining bits by correlation
    positive_var_idxs = np.arange(data_unpack.shape[1])[~zero_var_bits]
    bit_corr = np.corrcoef(data_unpack[:, positive_var_idxs], rowvar=False)
    bit_corr_max = np.max(np.abs(bit_corr) - np.eye(bit_corr.shape[0]), axis=0)
    bit_corr_order = np.argsort(bit_corr_max)[::-1]
    pos_var_idxs_ordered = positive_var_idxs[bit_corr_order]
    bit_ordering += [idx for idx in pos_var_idxs_ordered if idx not in bit_ordering]
    try:
        assert len(bit_ordering) == data_unpack.shape[1]
    except AssertionError:
        raise RuntimeError(
            "Length of bit ordering (%d) does not match sample length (%d).",
            len(bit_ordering),
            data_unpack.shape[1],
        )
    logger.debug("Final bit ordereing: {:s}".format(str(bit_ordering)))

    return bit_ordering, n_restricted_bits * data.shape[1]


def _count_bases(data, index, n_bits):
    """
    Recursively count the number of unique bases in the data for every possible base
    length using a depth first search. Each bit/column starting from the first bit is
    checked to see if there are one unique value or two values.

    If one, then a vector for counting unique values is incremented by 1 at that bit
    and the data is then passed on to the next iteration and the next bit is checked.

    If two, then the counting vector is incremented by 2 at that bit and the data is
    split based on the value of the relevant bit before being passed on to the next
    iteration to check the next bit.

    In this way, a tree-like structure (depth first search) is developed.

    Args:
        data (array): Data to determine number of bases for.
        index (int): Bit index to consider for splitting.
        n_bits (int): Total number of bits per sample.

    Raises:
        IndexError: Index is outside the array bounds.

    Returns:
        list: Count of unique bases at each base length.
    """
    # Check index is within length of data
    if index >= n_bits:
        raise IndexError("index ({:,d}) > data size ({:,d})".format(index, n_bits))

    # If only one data point, return 1s from index bit
    if data.shape[0] == 1:
        return (np.arange(n_bits) >= index).astype(int)

    # Check if there are one or two unique values and expand search
    zero_idxs = data[:, index] == 0  # indices where bit is zero
    unique_counts = np.zeros(n_bits, dtype=np.int32)  # pre-allocate vector
    if len(set(zero_idxs)) == 1:  # one unique value
        unique_counts[index] = 1
        if index + 1 >= n_bits:  # terminate if reached last bit
            return unique_counts
        else:
            unique_counts += _count_bases(data, index + 1, n_bits)
    else:  # two unique values, split data
        unique_counts[index] = 2
        if index + 1 >= n_bits:  # terminate if reached last bit
            return unique_counts
        else:
            unique_counts += _count_bases(data[zero_idxs], index + 1, n_bits)
            unique_counts += _count_bases(data[np.invert(zero_idxs)], index + 1, n_bits)

    return unique_counts


def get_num_bases(data, bit_order=None):
    """
    Returns the number of bases for each base length given a bit ordering.

    Args:
        data (DataFrame): Data to determine number of bases for.
        bit_order (list, optional): How to order bits before processing.

    Returns:
        list: Count of unique bases at each base length.
    """
    if not all(data.dtypes == np.uint8):
        data = unpack_bits(data)
    if bit_order:
        data = data[:, bit_order]  # sort bits based on given order
    n_bits = data.shape[1]
    return _count_bases(data, index=0, n_bits=n_bits)


def get_dev_bits_infocom(
    data,
    counts=True,
    exploration=0.1,
    training_size=None,
    k=None,
):
    """
    Select deviation bits using the version of GD from Rasmus' INFOCOM paper.

    Args:
        data (array): Data to be compressed.
        counts (bool): Include base counts in data size. Defaults True.
        exploration (float, optional): Amount of exploration beyond local minima.
        training_size (int): Number of samples from data to use for configuration.
        k (unused): Unused, only for API compatability.

    Returns:
        list: Deviation bits (in bit packed form)
        int: Estimated compressed data size
        int: Estimated number of bases
    """
    # Downsample for configuration
    if training_size:
        data = data[np.random.choice(data.shape[0], training_size, replace=False)]

    # Get compressed data size (cost) at each base length
    m = max([d.itemsize for d in data.dtypes]) * 8  # number of bits per value
    bit_ordering, _ = get_bit_ordering(data, bit_restrictions=False)
    data_unpack, _ = unpack_data(data, m)
    data_unpack = data_unpack[:, bit_ordering]

    # Classical deduplication case (all bits in base)
    # NOTE: pd.DataFrame.drop_duplicates is significantly faster than np.unique
    # because it does not sort the data.
    n, d = data.shape
    data_repacked = np.packbits(data_unpack, axis=1)
    best_n_b = pd.DataFrame(data_repacked).drop_duplicates().shape[0]
    best_dev_bits = np.array([0] * d, dtype=UINT_LOOKUP[m])
    best_cost, _ = get_compressed_size(n, d, best_n_b, best_dev_bits, [m] * d, counts)

    # Iteratively shift bits to the deviation until a local minima
    dev_bits_tmp = np.array([0] * d, dtype=UINT_LOOKUP[m])
    for i in range(1, len(bit_ordering) + 1):
        len_b = len(bit_ordering) - i
        data_repacked = np.packbits(data_unpack[:, :len_b], axis=1)
        n_b = pd.DataFrame(data_repacked).drop_duplicates().shape[0]
        idx = bit_ordering[-i]  # index of bit shifted to deviation
        jj = idx // m
        ii = idx % m
        bit_value = UINT_LOOKUP[m](1 << int(m - ii - 1))
        dev_bits_tmp[jj] = dev_bits_tmp[jj] ^ bit_value
        cost, _ = get_compressed_size(n, d, n_b, dev_bits_tmp, [m] * d, counts=counts)
        if (best_cost is None) or (cost < best_cost):  # update best cost
            best_cost = cost
            best_n_b = n_b
            best_dev_bits = dev_bits_tmp.copy()
        if (best_cost is not None) and cost > (1 + exploration) * best_cost:
            break
        logger.debug("len_d: %3d, len_b: %3d, n_b: %d, cost: %d", i, len_b, n_b, cost)
    return best_dev_bits, best_cost, best_n_b


def get_dev_bits_infocom_plus(
    data,
    counts=True,
    exploration=0.1,
    training_size=None,
    k=None,
):
    """
    Select deviation bits using the version of GD from Rasmus' INFOCOM paper with a
    number of changes:
        - Start with zero bits in the base and gradually add bits,
          instead of starting with all bits and gradually removing bits
        - Use the BaseLeaves for counting bases
        - Terminate when the cost exceeds the global best cost by an exploration factor

    Args:
        data (array): Data to be compressed.
        counts (bool): Include base counts in data size. Defaults True.
        exploration (float, optional): Amount of exploration beyond local minima.
        training_size (int): Number of samples from data to use for configuration.
        k (unused): Unused, only for API compatability.

    Returns:
        list: Deviation bits (in bit packed form)
        int: Estimated compressed data size
        int: Estimated number of bases
    """
    # Downsample for configuration
    if training_size:
        data = data[np.random.choice(data.shape[0], training_size, replace=False)]

    # Get compressed data size (cost) at each base length
    bit_ordering, _ = get_bit_ordering(data, bit_restrictions=False)
    data_unpack = unpack_bits(data)
    data_unpack_bool = data_unpack.astype(bool)

    # Iteratively shift bits to the base until a local minima
    m = [d.itemsize * 8 for d in data.dtypes]  # number of bits per value
    end_idxs = np.cumsum(m)
    best_cost = None
    best_n_b = None
    best_dev_bits = [UINT_LOOKUP[mi](2**mi - 1) for mi in m]
    dev_bits_tmp = best_dev_bits.copy()
    base_tree = BaseLeaves()
    for i, idx in enumerate(bit_ordering):
        leaf_nodes = base_tree.expand(data_unpack_bool[:, idx])
        n_b = len(leaf_nodes)
        jj = np.argmax(idx < end_idxs)
        ii = idx - (end_idxs[jj] - m[jj])
        bit_value = UINT_LOOKUP[m[jj]](1 << int(m[jj] - ii - 1))
        dev_bits_tmp[jj] = dev_bits_tmp[jj] ^ bit_value
        n, d = data.shape
        cost, _ = get_compressed_size(n, d, n_b, dev_bits_tmp, m, counts=counts)
        if (best_cost is None) or (cost < best_cost):  # update best cost
            best_cost = cost
            best_n_b = n_b
            best_dev_bits = dev_bits_tmp.copy()
        if (best_cost is not None) and cost > (1 + exploration) * best_cost:
            break
        logger.debug("len_b: %3d, n_b: %d, idx: %d, cost: %d", i + 1, n_b, idx, cost)

    return best_dev_bits, best_cost, best_n_b


def get_dev_bits_gdd(
    data,
    counts=True,
    bit_restrictions=True,
    training_size=None,
    k=None,
):
    """
    Select deviation bits using the "default" correlation bit order version of GD as was
    used in the 2021 GLOBECOM paper.

    Args:
        data (array): Data to be compressed.
        counts (bool): Include base counts in data size. Defaults True.
        bit_restrictions (bool, optional): Whether to restrict sign and exponent (for
            floating point data) bits to the base. Defaults to False.
        training_size (int): Number of samples from data to use for configuration.
        k (unused): Unused, only for API compatability.

    Returns:
        list: Deviation bits (in bit packed form)
        int: Estimated compressed data size
        int: Estimated number of bases
    """
    # Downsample for configuration
    if training_size:
        data = data[np.random.choice(data.shape[0], training_size, replace=False)]

    # Get compressed data size (cost) at each base length
    bit_ordering, n_restricted_bits = get_bit_ordering(data, bit_restrictions)
    n_b_all = get_num_bases(data, bit_order=bit_ordering)
    costs = []
    m = [dtype.itemsize * 8 for dtype in data.dtypes]
    dev_bits_tmp = [UINT_LOOKUP[mi](2**mi - 1) for mi in m]
    end_idxs = np.cumsum(m)
    for n_b, idx in zip(n_b_all, bit_ordering):
        j = np.argmax(idx < end_idxs)
        i = idx - (end_idxs[j] - m[j])
        bit_value = UINT_LOOKUP[m[j]](1 << int(m[j] - i - 1))
        dev_bits_tmp[j] = dev_bits_tmp[j] ^ bit_value
        costs.append(
            get_compressed_size(*data.shape, n_b, dev_bits_tmp, m, counts=counts)[0]
        )
    summary = np.vstack((np.arange(len(costs)) + 1, n_b_all, costs)).transpose()
    msg = "Base length, number of bases and estimated compressed data size:\n{:s}"
    logger.debug(msg.format(str(summary)))

    # Select base length at minimum compressed data size subject to requirement for all
    # restricted bits to be in the base
    cost_idx = np.argmin(costs[n_restricted_bits:]) + n_restricted_bits
    cost = costs[cost_idx]
    n_b = n_b_all[cost_idx]
    dev_bit_idxs = bit_ordering[cost_idx + 1 :]

    # Convert selected deviation bits to uint32 format
    logger.debug("Deviation bit indices:\n{:s}".format(str(sorted(dev_bit_idxs))))
    dev_bits = [UINT_LOOKUP[mi](0) for mi in m]
    for idx in dev_bit_idxs:
        j = np.argmax(idx < end_idxs)
        i = idx - (end_idxs[j] - m[j])
        dev_bits[j] += UINT_LOOKUP[m[j]](1 << int(m[j] - i - 1))

    return dev_bits, cost, n_b


def get_dev_bits_glean_plus(
    data,
    counts=True,
    exploration=0.1,
    k=5,
    training_size=None,
):
    """
    Select deviation bits using the version of GD from the GLEAN paper (analytics-
    tailored version). Note that this version, unlike others, restricts the sign and
    exponent bits to the base prior to selecting additional bits based on correlation.

    Args:
        data (array): Data to be compressed.
        counts (bool, optional): Include base counts in data size. Defaults to True.
        exploration (float, optional): Amount of exploration beyond local minima.
        k (int, optional): Number of clusters in the dataset. Defaults to 5.
        training_size (int): Number of samples from data to use for configuration.

    Returns:
        list: Deviation bits (in bit packed form)
        int: Estimated compressed data size
        int: Estimated number of bases
    """
    # Check that k is an integer
    if k is None:
        k = 5  # reset to defailt value

    # Downsample for configuration
    if training_size:
        data = data[np.random.choice(data.shape[0], training_size, replace=False)]

    # Get compressed data size (cost) at each base length
    bit_ordering, _ = get_bit_ordering(data, bit_restrictions=True)
    data_unpack = unpack_bits(data)
    d = data.shape[1]
    n_b_min = (6.54 * (d < 6) + 15.50 * ((d > 5) & (d < 11)) + 36.97 * (d > 10)) * d * k
    logger.debug("Minimum number of bases: {:,.1f}".format(n_b_min))

    # Iteratively shift bits to the base until a local minima
    m = [dtype.itemsize * 8 for dtype in data.dtypes]
    end_idxs = np.cumsum(m)
    best_cost = None
    best_n_b = None
    best_dev_bits = [UINT_LOOKUP[mi](2**mi - 1) for mi in m]
    dev_bits_tmp = best_dev_bits.copy()
    base_tree = BaseLeaves()
    for i, idx in enumerate(bit_ordering):
        leaf_nodes = base_tree.expand(data_unpack[:, idx])
        n_b = len(leaf_nodes)
        jj = np.argmax(idx < end_idxs)
        ii = idx - (end_idxs[jj] - m[jj])
        bit_value = UINT_LOOKUP[m[jj]](1 << int(m[jj] - ii - 1))
        dev_bits_tmp[jj] = dev_bits_tmp[jj] ^ bit_value
        cost, _ = get_compressed_size(*data.shape, n_b, dev_bits_tmp, m, counts=counts)
        if ((best_cost is None) or (cost < best_cost)) and (n_b > n_b_min):
            best_cost = cost
            best_n_b = n_b
            best_dev_bits = dev_bits_tmp.copy()
        if (best_cost is not None) and cost > (1 + exploration) * best_cost:
            break
        logger.debug("len_b: %3d, n_b: %d, idx: %d, cost: %d", i + 1, n_b, idx, cost)

    # If no suitable solution eas found, then allocate all bits to the base
    if best_cost is None:
        return dev_bits_tmp, cost, n_b
    else:
        return best_dev_bits, best_cost, best_n_b
