import logging

import numpy as np

from gd.BaseLeaves import BaseLeaves, BaseLeafNode
from gd.common import get_compressed_size, hamming, unpack_bits, UINT_LOOKUP

logger = logging.getLogger(__name__)


def _log_transformation_function(dev_bits, m, n_b, size_compressed, size_uncompressed):
    dev_bits_str = " ".join([f"{a:0{mi}b}" for a, mi in zip(dev_bits, m)])
    logger.debug(
        (
            "Transformation function:\n"
            f"  len_d:            {sum(hamming(a, type(a)(0)) for a in dev_bits):.0f}\n"
            f"  n_b:              {n_b:d}\n"
            f"  size (comp):      {size_compressed:.0f} (bytes)\n"
            f"  size (original):  {size_uncompressed:.0f} (bytes)\n"
            f"  cr:               {size_uncompressed / size_compressed:.3f}\n"
            f"  deviation bits:   {dev_bits_str}"
        ),
    )


def greedy_select(data, include_base_counts=True, alpha=0.1, L=0.02):
    """
    Estimate the optimal transformation function for a given dataset by greedily
    adding bits to the base. First, all zero-variance bits are added to the base, as
    well as all sign and exponent bits for floating point numbers. Next, the most
    significant deviation bits in each dimension are analysed and the one that
    results in the smallest compressed data size is added to the base.

    Args:
        data (array): Numpy integer array.
        include_base_counts (bool): Include base counts in data size. Defaults True.
        alpha (float): Amount of exploration beyond local optima.
        L (float, optional): Parameter to control magnitude of the effect of
            adjusting estimated costs in favour of attributes which have not had
            many bits added to the base. Defaults to 0.02.

    Returns:
        dev_bits (list): List of integers where the bit values indicate which bits
            of attribute i are assigned to (1) the deviation or (0) the base.
        cost (float): Estimated compressed data size in bytes.
        n_b (int): Estimated number of bases.
    """

    # assume input is a DataFrame
    # assume all columns are integer
    # possibly different precision and signed/unsigned
    # assume any downsampling has already been done

    n, d = data.shape
    m = [dtype.itemsize * 8 for dtype in data.dtypes]  # bits per column
    size_uncompressed = sum([n * mi for mi in m]) / 8  # total size, in bytes
    base_tree = BaseLeaves()
    dev_bits = [UINT_LOOKUP[mi](2**mi - 1) for mi in m]
    data_unpack = unpack_bits(data)  # unpacked bit values

    

    # Identify all constant bits, zero out these bits in the deviation bits vector and
    # add a node to the base tree with all sample indices.
    logger.debug("Adding constant bits to the base.")
    for i in range(d):
        all_zero_bits = ~np.bitwise_or.reduce(data.iloc[:, i], dtype=UINT_LOOKUP[m[i]])
        all_one_bits = np.bitwise_and.reduce(data.iloc[:, i], dtype=UINT_LOOKUP[m[i]])
        constant_bits = all_zero_bits | all_one_bits
        dev_bits[i] = dev_bits[i] ^ constant_bits
    base_tree.nodes.append(BaseLeafNode(np.arange(n)))
    logger.debug("Deviation bits after removing constant bits:")
    logger.debug(" ".join([f"{a:0{mi}b}" for a, mi in zip(dev_bits, m)]))

    # Initial transformation function
    global_best_n_b = len(base_tree.nodes)
    global_best_dev_bits = dev_bits.copy()
    global_best_size_compressed, _ = get_compressed_size(
        n, d, global_best_n_b, global_best_dev_bits, m
    )
    _log_transformation_function(
        dev_bits, m, global_best_n_b, global_best_size_compressed, size_uncompressed
    )

    # Iterate through the most significant remaining bits from each attribute and
    # evaluate the cost of adding them individually to the base. Identify the one
    # with the lowest cost and add this to the base. Repeat until termination.
    dev_bits_initial = dev_bits.copy()
    current_costs = [0] * d
    current_size_compressed = [None] * d
    current_bin_idxs = [None] * d
    current_bin_idxs_glob = [None] * d
    current_n_b = [None] * d
    while any(dev_bits):
        size_compressed = None
        column = None
        updated = [False] * d  # indicates if cost for each column has been updated

        # Calculate minimum costs based on previous n_b values
        for i in range(d):
            if not dev_bits[i]:
                updated[i] = True
                current_costs[i] = np.inf
                continue
            current_bin_idxs[i] = int(np.log2(dev_bits[i]))  # bit index
            current_bin_idxs_glob[i] = sum(m[: i + 1]) - 1 - current_bin_idxs[i]
            if current_n_b[i] is None:  # skip on first iteration
                continue
            dev_bits_i = dev_bits.copy()
            dev_bits_i[i] = dev_bits_i[i] ^ UINT_LOOKUP[m[i]](1 << current_bin_idxs[i])
            current_size_compressed[i], _ = get_compressed_size(
                n, d, current_n_b[i], dev_bits_i, m, counts=include_base_counts
            )
            adjustment_i = 1 - L * (dev_bits[i] / dev_bits_initial[i]) ** 2
            current_costs[i] = adjustment_i * current_size_compressed[i]

        # Select lowest cost dimension
        while not all(updated):
            i = np.nanargmin(current_costs)

            # Skip already updated columns. Break look if an already updated column now
            # has the lowest cost
            if updated[i]:
                if current_costs[i] == min(current_costs):
                    break
                continue

            # Get current cost for column i
            current_n_b[i] = base_tree.count_new_bases(
                data_unpack[:, current_bin_idxs_glob[i]]
            )
            dev_bits_i = dev_bits.copy()
            dev_bits_i[i] = dev_bits_i[i] ^ UINT_LOOKUP[m[i]](1 << current_bin_idxs[i])
            current_size_compressed[i], _ = get_compressed_size(
                n, d, current_n_b[i], dev_bits_i, m, counts=include_base_counts
            )
            adjustment_i = 1 - L * (dev_bits[i] / dev_bits_initial[i]) ** 2
            current_costs[i] = adjustment_i * current_size_compressed[i]
            updated[i] = True
            logger.debug(
                f"Bit {current_bin_idxs[i]:3d} from dimension {i:2d}, "
                f"n_b: {current_n_b[i]}, "
                f"estimated size: {current_size_compressed[i]:.0f}, "
                f"adjusted cost: {current_costs[i]:.0f}"
            )

            # Break if cost is lowest
            if current_costs[i] == min(current_costs):
                break

        # Select lowest cost column
        column = np.argmin(current_costs)
        size_compressed = current_size_compressed[column]
        b = current_bin_idxs[column]
        b_global = current_bin_idxs_glob[column]

        # Check for termination, if not then add new nodes to the tree and update
        # deviation bits.
        if size_compressed > (1 + alpha) * global_best_size_compressed:
            logger.debug("New size exceeds exploration limit, terminating.")
            break
        logger.debug(f"Selected bit {b} from column {column} (size: {size_compressed})")
        base_tree.expand(data_unpack[:, b_global])
        dev_bits[column] = dev_bits[column] ^ UINT_LOOKUP[m[column]](1 << b)

        # Update global optimum
        if size_compressed < global_best_size_compressed:
            global_best_n_b = len(base_tree.nodes)
            global_best_dev_bits = dev_bits.copy()
            global_best_size_compressed = size_compressed
            logger.debug(f"New global best cost: {global_best_size_compressed}")

    # Update final deviation bits and base tree results
    base_tree.nodes = base_tree.nodes[:global_best_n_b]
    dev_bits = global_best_dev_bits
    _log_transformation_function(
        dev_bits, m, global_best_n_b, global_best_size_compressed, size_uncompressed
    )

    return dev_bits, global_best_size_compressed, global_best_n_b
