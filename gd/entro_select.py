import logging

import numpy as np

from gd.BaseLeaves import BaseLeaves, BaseLeafNode
from gd.common import get_compressed_size, hamming, unpack_bits, UINT_LOOKUP

logger = logging.getLogger(__name__)


def calculate_column_entropy(bits_array):
    """Calculate the entropy of each column in the bits array using vectorized operations."""
    if bits_array.ndim != 2:
        raise ValueError("Input array must be 2D")
    
    n_rows, n_cols = bits_array.shape
    
    # Count 1s in each column (vectorized)
    ones_count = np.sum(bits_array, axis=0)  # Sum along rows for each column
    zeros_count = n_rows - ones_count        # Remaining are zeros
    
    # Calculate probabilities (vectorized)
    p_ones = ones_count / n_rows
    p_zeros = zeros_count / n_rows
    
    # Calculate entropy: H = -p0*log2(p0) - p1*log2(p1)
    # Handle log(0) by using np.where to avoid division by zero
    entropies = np.zeros(n_cols)
    
    # Only calculate log where probabilities > 0
    mask_zeros = p_zeros > 0
    mask_ones = p_ones > 0
    
    # Vectorized entropy calculation
    entropies -= np.where(mask_zeros, p_zeros * np.log2(p_zeros+1e-9), 0)
    entropies -= np.where(mask_ones, p_ones * np.log2(p_ones+1e-9), 0)
    
    return entropies


def _log_transformation_function(dev_bits, m, n_b, size_compressed, size_uncompressed):
    dev_bits_str = " ".join([f"{a:0{mi}b}" for a, mi in zip(dev_bits, m)])
    logger.debug(
        (
            "Transformation function:\n"
            f"  len_d:            {sum(int(hamming(a, type(a)(0))) for a in dev_bits):.0f}\n"
            f"  n_b:              {n_b:d}\n"
            f"  size (comp):      {size_compressed:.0f} (bytes)\n"
            f"  size (original):  {size_uncompressed:.0f} (bytes)\n"
            f"  cr:               {size_uncompressed / size_compressed:.3f}\n"
            f"  deviation bits:   {dev_bits_str}"
        ),
    )


def entro_select(data, entropy=None, include_base_counts=False, tau=10):
    """
    Estimate the transformation function for a given dataset by entropy values of bit
    positions from low to high. 

    Args:
        data (array): Numpy integer array.
        include_base_counts (bool): Include base counts in data size. Defaults False.


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
    dev_bits_str = " ".join([f"{a:0{mi}b}" for a, mi in zip(dev_bits, m)])
    n_zeros = sum(f"{a:0{mi}b}".count('0') for a, mi in zip(dev_bits, m))
    logger.debug(f"{dev_bits_str} (constant bits: {n_zeros})")

    # Initial transformation function
    global_best_n_b = len(base_tree.nodes)
    global_best_dev_bits = dev_bits.copy()
    global_best_size_compressed, _ = get_compressed_size(
        n, d, global_best_n_b, global_best_dev_bits, m
    )
    _log_transformation_function(
        dev_bits, m, global_best_n_b, global_best_size_compressed, size_uncompressed
    )

    # Calculate entropy for all bit columns once (entropy-based selection strategy)
    if entropy is None:
        logger.debug("Calculating entropy for all bit columns")
        entropy = calculate_column_entropy(data_unpack)

    entropy_sorted_indices = np.argsort(entropy)
    logger.debug(f"Entropy sorted bit indices: {entropy_sorted_indices}")
    entropy_sorted_indices_non_constant = entropy_sorted_indices[n_zeros:]

    
    # Sort bit indices by entropy (low to high, excluding constant bits which have entropy ~0)
    # Constant bits already handled above, so we start from non-constant bits
    # non_constant_bit_indices = []
    # for bit_idx in range(data_unpack.shape[1]): 
    #     if entropy[bit_idx] > 1e-9:  # Skip constant bits (entropy ~0)
    #         non_constant_bit_indices.append(bit_idx)
    
    # Sort non-constant bits by entropy (low to high)
    # entropy_sorted_indices = sorted(non_constant_bit_indices, key=lambda x: entropy[x])
    
    logger.debug(f"Number of non-constant bits to process: {len(entropy_sorted_indices_non_constant)}")
    
    # Iterate through bits in order of increasing entropy and add them to the base
    not_de = 0
    not_de_max = tau  # Use tau parameter for early termination
    
    for bit_global_idx in entropy_sorted_indices_non_constant:
        # Determine which column and bit position this global bit index corresponds to
        column = None
        bit_position = None
        cumsum_bits = 0
        for i in range(d):
            if cumsum_bits + m[i] > bit_global_idx:
                column = i
                bit_position = m[i] - 1 - (bit_global_idx - cumsum_bits)  # local bit index
                bit_position = UINT_LOOKUP[m[i]](bit_position)
                break
            cumsum_bits += m[i]
        
        if column is None:
            continue
        
        # Check if this bit is still in dev_bits (not yet assigned to base)
        if not (dev_bits[column] & (1 << bit_position)):
            continue
        
        # Try adding this bit to the base
        n_b_new = base_tree.count_new_bases(data_unpack[:, bit_global_idx])
        dev_bits_new = dev_bits.copy()
        dev_bits_new[column] = dev_bits_new[column] ^ UINT_LOOKUP[m[column]](1 << bit_position)
        size_compressed_new, _ = get_compressed_size(
            n, d, n_b_new, dev_bits_new, m, counts=include_base_counts
        )
        
        logger.debug(
            f"Bit {bit_position:3d} from column {column:2d} (global idx: {bit_global_idx}, "
            f"entropy: {entropy[bit_global_idx]:.4f}), "
            f"n_b: {n_b_new}, "
            f"estimated size: {size_compressed_new:.0f}"
        )
        
        # Always expand the tree (commit the change)
        base_tree.expand(data_unpack[:, bit_global_idx])
        dev_bits[column] = dev_bits[column] ^ UINT_LOOKUP[m[column]](1 << bit_position)
        
        # Update global optimum if this improves compression
        if size_compressed_new < global_best_size_compressed:
            global_best_n_b = len(base_tree.nodes)
            global_best_dev_bits = dev_bits.copy()
            global_best_size_compressed = size_compressed_new
            logger.debug(f"New global best cost: {global_best_size_compressed}")
            not_de = 0
        else:
            not_de += 1
            logger.debug(f"No improvement for {not_de} iterations")
            if not_de >= not_de_max:
                logger.debug(f"No improvement for {not_de_max} iterations, stopping")
                break
        
        # Early termination if we have too many bases
        if len(base_tree.nodes) >= n * 0.8:
            logger.debug(f"Too many bases ({len(base_tree.nodes)}), stopping")
            break

    # Update final deviation bits and base tree results
    base_tree.nodes = base_tree.nodes[:global_best_n_b]
    dev_bits = global_best_dev_bits
    _log_transformation_function(
        dev_bits, m, global_best_n_b, global_best_size_compressed, size_uncompressed
    )

    return dev_bits, global_best_size_compressed, global_best_n_b
