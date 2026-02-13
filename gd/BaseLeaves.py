import logging

import numpy as np

from gd.common import get_nonzero_bit_idices

logger = logging.getLogger(__name__)


class BaseLeafNode:
    """Class for storing indices mapping to a base."""

    def __init__(self, sample_idxs):
        self.sample_idxs = sample_idxs


class BaseLeaves:
    """
    This class implements a "leaf only" version of BaseTree. That is, only the current
    layer of the tree - the leaves - is retained in memory.
    """

    def __init__(self):
        self.nodes = []  # list containing leaf nodes (i.e. bases)

    def log_stats(self, message, size_compressed, size_uncompressed, dev_bits):
        """
        Log the current statistics for the bases with a custom message.

        Args:
            message (str): Custom message to display in log output.
            size_compressed (int): Compressed data size current configuration (bytes).
            size_uncompressed (inf): Size of uncompressed data (bytes).
            dev_bits (list): List of deviation bits in bit-packed format.
        """
        len_d = len(get_nonzero_bit_idices(dev_bits))
        len_b = len(get_nonzero_bit_idices(dev_bits, invert=True))
        n_b = len(self.nodes)
        dev_bits_str = str([bin(a) for a in dev_bits])
        cr = size_compressed / size_uncompressed
        logger.debug(
            (
                "%s\n"
                "  len_b:  %d\n"
                "  len_d:  %d\n"
                "  n_b:    %d\n"
                "  size (compressed):  %.0f (bytes)\n"
                "  size (original):    %.0f (bytes)\n"
                "  compression ratio:  %.3f}\n"
                "  deviation bits:  %s"
            ),
            message,
            len_b,
            len_d,
            n_b,
            size_compressed,
            size_uncompressed,
            cr,
            dev_bits_str,
        )

    def count_new_bases(self, bit_values):
        """
        Returns the number of bases resulting from adding one additional bit to the base
        tree. The input should be the values of the additional bit. The idea of this
        function is to quickly count bases without having to extend the tree.

        Args:
            bit_values (array): Boolean array containing bit values for the bit which
                may be added to the tree.

        Returns:
            int: Number of bases after from adding the given bit to the base bits set.
        """
        n_b = 0
        for node in self.nodes:
            a = bit_values[node.sample_idxs].sum(axis=0)
            n_b += 2 - (a == 0) - (a == node.sample_idxs.shape[0])
        return n_b

    def expand(self, binary_data):
        """
        Update the list of bases by adding a given bit to the base.

        Args:
            binary_data (list): 1D boolean array containing the data for the bit that is
                to be added to the base.

        Returns:
            array: Current list of base nodes.
        """
        # If no bases have yet been created (i.e. empty nodes list), create a temporary
        # base with all data from which to start expansion
        if len(self.nodes) == 0:
            self.nodes.append(BaseLeafNode(np.arange(len(binary_data))))

        # Iterate through all bases/leaf nodes and check if they need to be split based
        # on the given binary data. If a node needs to be split, then (1) the node's
        # sample indices are updated so that the samples with 0 in the given bit index
        # are mapped to this node and (2) a new node is created with the sample indices
        # for the samples with 1 in the given bit index.
        for node in self.nodes:
            binary_data_at_node = binary_data[node.sample_idxs]
            if any(binary_data_at_node) and (not all(binary_data_at_node)):
                sample_idxs0 = node.sample_idxs[~binary_data_at_node]
                sample_idxs1 = node.sample_idxs[binary_data_at_node]
                node.sample_idxs = sample_idxs0
                self.nodes.append(BaseLeafNode(sample_idxs1))

        return self.nodes
