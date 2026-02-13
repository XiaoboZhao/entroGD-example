import os
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import cv2
import io
import copy


def decimal_to_binary(data, bit_len, n, d, total_bits):
    """
    Convert an array of integers to a binary representation using np.unpackbits.
    
    Args:
        data: 2D numpy array of integers. The dtype of data should be appropriate for its values.
        bit_len: List of bit lengths (e.g., 8, 16, 32, 64) for each column.
        n: Number of rows in the data.
        d: Number of columns in the data.
        total_bits: Total number of bits in the output array.
    
    Returns:
        2D numpy array of binary values (0s and 1s).
    """
    # Create an empty binary array
    bit_data = np.zeros((n, total_bits), dtype=np.uint8)
    
    current_bit_idx = 0
    for j in range(d):
        col_bits = bit_len[j]
        
        # Get the column and ensure it has the correct data type for unpacking.
        # The data comes in as int64, so we can safely cast down.
        dtype = np.dtype(f'uint{col_bits}')
        
        # Convert to the specified dtype AND ensure big-endian byte order.
        # The '>' prefix in the dtype string forces big-endian.
        column_data = data[:, j].astype(dtype.newbyteorder('>'))
        
        # View the data as bytes (uint8) to use with unpackbits.
        # Because we forced big-endian, the byte order is now correct.
        byte_view = column_data.view(np.uint8).reshape(n, -1)
        
        # Unpack the bits. bitorder='big' is correct, but the byte order of the
        # input data is what really matters.
        unpacked = np.unpackbits(byte_view, axis=1)
        
        # Place the unpacked bits into the correct slice of the main array.
        bit_data[:, current_bit_idx : current_bit_idx + col_bits] = unpacked
        
        current_bit_idx += col_bits
            
    return bit_data

class TreeNode:
    """Node in the entropy-based compression tree."""
    def __init__(self, data_indices, bit_column=None, bit_value=None, level=0):
        self.data_indices = data_indices  # Indices of data points in this node
        self.bit_column = bit_column      # Which bit column was used to split
        self.bit_value = bit_value        # 0 or 1 - the value this node represents
        self.level = level                # Depth in tree
        self.left = None                  # Child for bit_value = 0
        self.right = None                 # Child for bit_value = 1
        self.is_leaf = True               # Whether this is a leaf node
        self.size = len(data_indices)     # Number of data points in this node
    
    def split(self, bit_data, bit_column):
        """Split this node based on a bit column."""
        if self.size <= 1:
            return False  # Can't split further
        
        # Get the bit values for this column for our data indices
        bit_values = bit_data[self.data_indices, bit_column]
        
        # Split indices based on bit values
        mask_0 = bit_values == 0
        mask_1 = bit_values == 1
        
        indices_0 = self.data_indices[mask_0]
        indices_1 = self.data_indices[mask_1]
        
        # Only create children if both groups are non-empty
        if len(indices_0) > 0 and len(indices_1) > 0:
            self.left = TreeNode(indices_0, bit_column, 0, self.level + 1)
            self.right = TreeNode(indices_1, bit_column, 1, self.level + 1)
            self.bit_column = bit_column
            self.is_leaf = False
            return True
        
        return False  # No effective split
    
    def get_all_leaves(self):
        """Get all leaf nodes in the subtree."""
        if self.is_leaf:
            return [self]
        
        leaves = []
        if self.left:
            leaves.extend(self.left.get_all_leaves())
        if self.right:
            leaves.extend(self.right.get_all_leaves())
        return leaves
    
    def calculate_compression_size(self, total_bits, num_constant_bits, num_low_entro_bits=0, num_condensed=0):
        """Calculate compression size for this tree structure."""
        leaves = self.get_all_leaves()
        num_bases = len(leaves)

        leave_sizes = [leaf.size for leaf in leaves]
        
        if num_bases == len(self.data_indices):
            return float('inf'), num_bases, 0, 0  # No compression achieved
        
        # Calculate bits needed
        num_base_bits = num_constant_bits + num_low_entro_bits
        num_dev_bits = total_bits - num_base_bits
        
        # Compression size calculation
        n = len(self.data_indices)

        num_leave_sizes_not_one = sum(1 for size in leave_sizes if size > 1)

        ## only save base ID for base counts > 1
        # compressed_size = (
        #     num_bases * num_base_bits  # base size
        #     + n * (np.ceil(np.log2(max(num_leave_sizes_not_one, 1))) + num_dev_bits)  # dev size
        #     + num_condensed * np.ceil(np.log2(max(n - num_condensed, 1)))  # condensed sample weight size
        # )
        
        # save all base IDs
        compressed_size = (
            num_bases * num_base_bits  # base size
            + n * (np.ceil(np.log2(max(num_bases, 1))) + num_dev_bits)  # dev size
            + num_condensed * np.ceil(np.log2(max(n - num_condensed, 1)))  # condensed sample weight size
        )
    
        
        return compressed_size, num_bases, num_base_bits, num_dev_bits, leave_sizes

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


def entroGD_cluster(data, SCHEMA=None, max_bases=1e20, binary=False, bit_len=None, d=None):
    """Tree-based optimized version of EntroGD compression."""
    
    # # Check if input data is already binary (only contains 0s and 1s)
    # is_binary = np.all((data == 0) | (data == 1))
    
    if binary:
        # print("Input data is already binary, skipping conversion.")
        bit_data = data.astype(np.uint8)
        n, total_bits = data.shape
        
        # For binary data, we don't need bit length calculation
        # print(f"Binary data shape: {bit_data.shape}")
    else:
        print("Input data is not binary, converting to binary representation.")

        # Quantize each column using 10**precision and convert to integers
        precisions = []
        for i, col in enumerate(data.columns):
            prec = int(SCHEMA[col]['precision'])
            precisions.append(prec)
        scales = (10 ** np.array(precisions, dtype=np.int64)).astype(np.int64)
        data = np.rint(data.to_numpy(dtype=np.float64) * scales).astype(np.int64)

        # Find the minimum of each column in the data array
        col_mins = np.min(data, axis=0)
        print(f"col_mins: {col_mins}")
        # Subtract the minimum from each column to ensure no negative values
        data = data - col_mins

        # Find the maximum of each column in the data array
        col_maxs = np.max(data, axis=0)
        print(f"col_maxs: {col_maxs}")

        bit_len = []
        for val in col_maxs:
            # Ensure support up to uint64
            if val > (1 << 64) - 1:
                raise ValueError("Value exceeds uint64 max (18446744073709551615)")
            elif val > 0xFFFFFFFF:
                bit_len.append(64)
            elif val > 0xFFFF:
                bit_len.append(32)
            elif val > 0xFF:
                bit_len.append(16)
            else:
                bit_len.append(8)

        n, d = data.shape
        total_bits = sum(bit_len)
        bit_data = decimal_to_binary(data, bit_len, n, d, total_bits)
        
        print(f"Binary data shape: {bit_data.shape}")
    

    # Calculate entropy for all columns once
    entropy = calculate_column_entropy(bit_data)
    col_bit_cumsum = np.cumsum([0] + bit_len)
    constant_bit_indices = np.where(entropy < 0)[0]  # Indices of constant bits
    # Convert to set for O(1) lookup instead of O(n) with array
    constant_bit_set = set(constant_bit_indices)
    num_constant_bits = len(constant_bit_indices)
    # print(f'number of constant bits: {num_constant_bits}')


    msb_bit_position_col = col_bit_cumsum.copy()[:d]  # Initialize MSB positions for each column
    for i in range(d):
        # Find the first constant bit in each column (if any)
        start_idx, end_idx = col_bit_cumsum[i], col_bit_cumsum[i + 1]
        # Use set for faster membership testing
        for bit_idx in range(start_idx, end_idx):
            if bit_idx not in constant_bit_set:
                msb_bit_position_col[i] = bit_idx
                break
        else:
            msb_bit_position_col[i] = end_idx

    
    # Build MSB indices more efficiently by selecting first non-constant MSB bit from each column,
    # then second, then third, etc., until we reach num_high_entropy_bits or all columns are exhausted
    msb_high_sorted_indices = []

    # Track which columns still have available bits - use set for O(1) removal
    active_columns = set(range(d))
    bit_position_col = msb_bit_position_col.copy()  # Copy to track current bit position in each column



    # Initialize tree with root node containing all data
    root = TreeNode(np.arange(n))
    current_trees = [root]  # Track current leaf nodes
    num_bases = len(root.get_all_leaves())

    while active_columns:
        current_round_bits = []
        
        # For each active column, get the bit at current position (if it exists and is non-constant)
        cols_to_remove = []
        for col in active_columns:
            if bit_position_col[col] >= col_bit_cumsum[col+1]:
                # Column exhausted, mark for removal
                cols_to_remove.append(col)
                continue
                
            bit_index = bit_position_col[col]
            current_round_bits.append(bit_index)
        
        # Remove exhausted columns (do this after iteration to avoid modifying set during iteration)
        active_columns -= set(cols_to_remove)
        
        # Sort current round bits by entropy (highest first) and add to result
        current_round_bits_sorted = sorted(current_round_bits, key=lambda x: entropy[x], reverse=True)

        for bit_column in current_round_bits_sorted:

            # Try to split each current leaf node
            new_trees = []
            any_split = False
            
            for tree_node in current_trees:
                if tree_node.is_leaf and tree_node.split(bit_data, bit_column):
                    # Successfully split - add children to new trees
                    new_trees.extend([tree_node.left, tree_node.right])
                    any_split = True

                else:
                    # Couldn't split or not a leaf - keep as is
                    new_trees.append(tree_node)
            
            if not any_split:
                continue  # No splits happened, try next bit column
            
            current_trees = new_trees
            num_bases = len(root.get_all_leaves())
            if num_bases >= max_bases:
                print(f"Reached max bases ({num_bases}), stopping")

                cluster_indices = [leaf.data_indices for leaf in root.get_all_leaves()]
                return cluster_indices, entropy
            
        bit_position_col = bit_position_col + 1  # Move to next bit in each column
