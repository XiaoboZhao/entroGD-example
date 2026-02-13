import logging
import os
from math import ceil, floor
import time

import numpy as np
import pandas as pd

from gd.bitwise_correlation import (
    get_dev_bits_glean_plus,
    get_dev_bits_gdd,
    get_dev_bits_infocom_plus,
    get_dev_bits_infocom,
)
from gd.common import (
    get_base_bit_indices,
    get_deviation_bit_indices,
    get_min_bit_length,
    pack_bits,
    sample_from_chunks,
    unpack_bits,
)
from gd.config import BUFFER_LOOKUP
from gd.data_loader import load_dataset
from gd.elias_gamma_coding import elias_gamma_encode, elias_gamma_decode
from gd.greedy_select import greedy_select
from gd.preprocessing import preprocess, preprocess_revert
from gd.entro_select import entro_select


STRING_JOIN_CHAR = ","
STRING_TERMINATION_CHAR = ";"
PARAMS_TYPE_LOOKUP = {
    "id": np.uint8(0),
    "categorical": np.uint8((1 << 4) + 0),
    "date": np.uint8((2 << 4) + 0),
    "time_micro": np.uint8((3 << 4) + 0),
    "time": np.uint8((3 << 4) + 1),
    "timestamp_micro": np.uint8((4 << 4) + 0),
    "timestamp": np.uint8((4 << 4) + 1),
    np.float32: np.uint8((5 << 4) + 0),
    np.float64: np.uint8((5 << 4) + 1),
    np.int8: np.uint8((6 << 4) + 0),
    np.int16: np.uint8((6 << 4) + 1),
    np.int32: np.uint8((6 << 4) + 2),
    np.int64: np.uint8((6 << 4) + 3),
    np.uint8: np.uint8((7 << 4) + 0),
    np.uint16: np.uint8((7 << 4) + 1),
    np.uint32: np.uint8((7 << 4) + 2),
    np.uint64: np.uint8((7 << 4) + 3),
}
PARAMS_TYPE_LOOKUP_REVERSE = {v: k for k, v in PARAMS_TYPE_LOOKUP.items()}

logger = logging.getLogger(__name__)
def configure_logging(level=logging.WARNING):
    """Configure logging level for the GD module."""
    logging.basicConfig(level=level)
    logger.setLevel(level)
# Set default logging level to WARNING to suppress INFO messages
configure_logging(logging.WARNING)


class GD:
    @staticmethod
    def _get_base_hashes(data_chunk, dev_bits):
        bases_tmp = data_chunk.copy()
        for i, c in enumerate(data_chunk.columns):
            bases_tmp[c] = np.bitwise_and(bases_tmp[c], ~dev_bits[i])
        return bases_tmp.apply(lambda x: hash(tuple(x)), axis=1)

    @staticmethod
    def _strings_to_bytes(input):
        # TODO docstring: input must be a list of strings (or integers?)
        list_of_strings = [str(x) for x in input]
        joined_string = STRING_JOIN_CHAR.join(list_of_strings) + STRING_TERMINATION_CHAR
        return np.array([ord(c) for c in joined_string], dtype=np.uint8)

    @staticmethod
    def _bytes_to_strings(input):
        # TODO docstring: input must be a numpy binary array of uint8
        # parses input until a stop character is reached
        output = []
        current_string = ""
        for byte in input:
            if chr(byte) == STRING_TERMINATION_CHAR:
                if len(current_string):
                    output.append(current_string)
                break
            elif chr(byte) == STRING_JOIN_CHAR:
                output.append(current_string)
                current_string = ""
            else:
                current_string += chr(byte)
        return output

    @staticmethod
    def compress(
        data,
        schema=None,
        entropy=None,
        has_header=True,
        out_path=None,
        export_counts=False,
        export_synopsis=False,
        gd_version="gd-greedy",
        config_sample_size=int(1e6),
        exploration=0.1,
        balance=0.02,
        k=5,
        sep=",",
        chunk_size=int(1e7),
        drop_trailing_base_bits=0,
        drop_trailing_dev_bits=0,
        tau=10,
    ):
        """
        Compress data using Generalized Deduplication. Data must be tabular.

        Args:
            data (str or DataFrame): Data to compress.
            schema (dict, optional): Data schema. Defaults to None.
            has_header (bool, optional): Whether the data has headers. Only
                relevant if loading from a file. Defaults to True.
            out_path (str, optional): Output path. Defaults to None.
            export_counts (bool, optional): Include base counts. Defaults to False.
            export_synopsis (bool, optional): Export the base/counts synopsis as a
                separate file. Defaults to False.
            gd_version (str, optional): GD compressor version. Defaults to "gd-greedy".
            config_sample_size (int, optional): Sample size for configuring GD.
                Defaults to 1e6.
            exploration (float, optional): GreedySelect exploration factor. Default 0.1.
            balance (float, optional): GreedySelect balance factor. Defaults 0.02.
            k (int, optional): Estimated number of clusters in the data. Defaults to 5.
            sep (str, optional): Data delimeter (if loading from file). DEfaults to ,.
            chunk_size (int, optional): Processing chunk size. Defaults to 1e7.
            drop_trailing_base_bits (int, optional): Number of trailing (least
                significant) base bits to omit. Compression is lossy if this is more
                than zero.
            drop_trailing_dev_bits (int, optional): Number of trailing (least
                significant) deviation bits to omit. Compression is lossy if this is
                more than zero.
        """
        # Setup output filenames
        data_name = os.path.splitext(os.path.basename(data))[0]
        if out_path is None:
            out_path = f"gd_compressed_data/{data_name}_out.gd"
            out_bases = f"gd_compressed_data/{data_name}_out_bases.gd"
        else:
            if isinstance(data, str):
                basename = os.path.splitext(os.path.basename(data))[0]
                out_bases = os.path.join(out_path, f"{basename}_bases.gd")
                out_path = os.path.join(out_path, f"{basename}.gd")
            else:
                out_bases = os.path.join(out_path, "out_bases.gd")
                out_path = os.path.join(out_path, "out.gd")

        # Load/prepare data
        if isinstance(data, str) and os.path.exists(os.path.dirname(data)):  # filepath
            data_chunks = load_dataset(
                data, schema, chunk_size=chunk_size, has_header=has_header, sep=sep
            )
        else:  # dataframe
            assert isinstance(data, pd.DataFrame)
            data_chunks = [data]
        n = np.sum([chunk.shape[0] for chunk in data_chunks])
        d = data_chunks[0].shape[1]
        n_chunks = len(data_chunks)

        config_start = time.perf_counter()
        # Data preprocessing
        logger.info("Preprocessing...")

        column_names = data_chunks[0].columns.copy()
        data_chunks, minimum_values, categorical_maps, na_indicators = preprocess(
            data_chunks, gd_version, schema
        )
        bits_per_value = [dtype.itemsize * 8 for dtype in data_chunks[0].dtypes]

        # Select deviation bits
        logger.info("Selecting deviation bits...")

        training_data = sample_from_chunks(data_chunks, sample_size=config_sample_size)
        if gd_version == "gd-greedy":
            dev_bits, _, _ = greedy_select(
                training_data,
                include_base_counts=export_counts,
                alpha=exploration,
                L=balance,
            )
        elif gd_version == "gd-d":
            dev_bits, _, _ = get_dev_bits_gdd(training_data, export_counts)
        elif gd_version == "gd-infocom":
            dev_bits, _, _ = get_dev_bits_infocom(training_data, export_counts)
        elif gd_version == "gd-infocom+":
            dev_bits, _, _ = get_dev_bits_infocom_plus(training_data, export_counts)
        elif gd_version == "gd-glean":
            dev_bits, _, _ = get_dev_bits_glean_plus(training_data, export_counts, k)
        elif gd_version == "gd-glean+":
            dev_bits, _, _ = get_dev_bits_glean_plus(training_data, export_counts, k)
        elif gd_version == "gd-entro":
            dev_bits, _, _ = entro_select(training_data, entropy=entropy, tau=tau)
        else:
            raise ValueError("Unsupported GD version: {:s}".format(gd_version))

        # Compute bash hashes
        # NOTE 1: The motivation for this is that by representing bases as byte strings,
        # we can compare them (and therefore deduplicate them) much faster
        # NOTE 2: I probably can't use multiprocessing here because this will copy the
        # data for each process and potentially cause a memory issue.
        config_time = time.perf_counter() - config_start
        logger.info(f"Configuration finished in {config_time} seconds.")

        logger.info("Computing base hashes...")

        base_bytes = []
        for chunk in data_chunks:
            base_bytes.append(GD._get_base_hashes(chunk, dev_bits))
        base_bytes = pd.concat(base_bytes)

        # Deduplicate bases
        logger.info("Deduplicating bases...")

        bases_dict = {}
        base_ids = np.zeros(n, dtype=np.uint32)
        next_id = 0
        for i, b_bytes in enumerate(base_bytes):
            try:  # existing base
                bases_dict[b_bytes]["count"] += 1
                base_ids[i] = bases_dict[b_bytes]["id"]
            except KeyError:  # new base
                bases_dict[b_bytes] = {
                    "id": next_id,  # unique base id
                    "chunk": i // chunk_size,  # chunk where base first appears
                    "index": i % chunk_size,  # index within chunk
                    "count": 1,
                }
                base_ids[i] = next_id
                next_id += 1
                # NOTE: From Python 3.6, dictionaries maintain the order of insertion.
        bases_dedup = pd.DataFrame(
            [data_chunks[b["chunk"]].iloc[b["index"]] for b in bases_dict.values()],
        )
        bases_dedup = bases_dedup.astype(data_chunks[0].dtypes)
        bases_dedup_unpack = unpack_bits(bases_dedup)
        base_bit_indices = get_base_bit_indices(
            dev_bits, drop_trailing_bits=drop_trailing_base_bits
        )
        base_bits = bases_dedup_unpack[:, base_bit_indices].ravel()
        n_bases = bases_dedup.shape[0]
        logger.info(f"Total unique bases: {n_bases}")

        logger.info(f"Bases:\n{bases_dedup}")


        # Export parameters
        logger.info("Exporting parameters...")

        general_params = [
            np.uint16(d),
            np.uint8(export_counts),
            np.uint32(n_bases),
            np.uint64(n),
            np.uint8(drop_trailing_base_bits),
            np.uint8(drop_trailing_dev_bits),
        ]
        column_params = [
            [
                PARAMS_TYPE_LOOKUP[schema[c]["type"]],  # original type
                np.uint8(schema[c]["precision"]),  # original precision
                np.uint8(bits_per_value[i]),  # encoded bit length
                np.uint8(na_indicators[i]),
                dev_bits[i],  # encoded deviation bits
                np.int64(minimum_values[i]),  # NOTE: signed
            ]
            for i, c in enumerate(column_names)
        ]
        if len(categorical_maps):
            categorical_params = [
                [np.uint8(i), *GD._strings_to_bytes(categorical_maps[i])]
                for i in categorical_maps
            ]
            categorical_params.append([np.uint8(ord(STRING_TERMINATION_CHAR))])
        else:
            categorical_params = [[np.uint8(ord(STRING_TERMINATION_CHAR))]]
        params = [
            general_params,
            *column_params,
            *categorical_params,
            GD._strings_to_bytes(column_names),
        ]
        params = [p for sublist in params for p in sublist]
        params_bits = []
        for p in params:
            if isinstance(p, np.uint8):
                params_bits.append(np.unpackbits(p))
            else:
                fmt = BUFFER_LOOKUP[str(p.dtype)]
                params_bits.append(
                    np.unpackbits(np.ascontiguousarray(p, dtype=fmt).view(">u1"))
                )
        params_bits = np.hstack(params_bits)
        assert len(params_bits) % 8 == 0  # whole number of bytes
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(np.packbits(params_bits))
        if export_synopsis:
            with open(out_bases, "wb") as f:
                f.write(np.packbits(params_bits))

        # Export bases
        # NOTE: "carry_over" contains the overflow after the last whole byte. This is
        # written to the output file at the start of the deviations.
        logger.info("Exporting bases...")

        n_bytes = base_bits.shape[0] // 8
        carry_over = base_bits[n_bytes * 8 :]
        carry_over_analytics = base_bits[n_bytes * 8 :]
        with open(out_path, "ab") as f:
            f.write(np.packbits(base_bits[: n_bytes * 8]))
        if export_synopsis:
            with open(out_bases, "ab") as f:
                f.write(np.packbits(base_bits[: n_bytes * 8]))

        # Export base IDs
        # 1. add elias gamma encoding of len_id to carry_over.
        # 2. identify number of bytes to write to file
        # 3. write base ids and update carry over
        logger.info("Exporting base IDs...")

        len_id = get_min_bit_length(max(base_ids))
        logger.debug("Base ID length: %d bit(s)", len_id)

        base_ids_bits = unpack_bits(base_ids)[:, -len_id:]
        to_write = np.hstack(
            (carry_over, elias_gamma_encode(len_id), base_ids_bits.ravel())
        )
        n_bytes = to_write.shape[0] // 8
        carry_over = to_write[n_bytes * 8 :]
        with open(out_path, "ab") as f:
            f.write(np.packbits(to_write[: n_bytes * 8]))

        # Export deviations
        # 1. divide deviation bits into chunks, unpack and extract deviation bits
        # 2. combine with carry_over
        # 3. identify number of bytes to write to file
        # 4. update carry over and write to file
        logger.info("Exporting deviations...")

        dev_bit_indices = get_deviation_bit_indices(
            dev_bits, drop_trailing_bits=drop_trailing_dev_bits
        )
        logger.debug("Deviations length: %d", len(dev_bit_indices))

        for i in range(n_chunks):
            logger.info(f"Writing deviations chunk {i + 1}/{n_chunks}")

            deviation_bits = unpack_bits(data_chunks[i])[:, dev_bit_indices]
            deviation_bits = np.hstack((carry_over, deviation_bits.ravel()))
            n_bytes = deviation_bits.shape[0] // 8
            carry_over = deviation_bits[n_bytes * 8 :]
            with open(out_path, "ab") as f:
                f.write(np.packbits(deviation_bits[: n_bytes * 8]))

        # Export: base counts & final carry over
        if export_counts and (gd_version != "gd-entro"):
            logger.info("Exporting base counts...")

            base_counts = [np.uint32(bases_dict[b]["count"]) for b in bases_dict]
            max_count = max(base_counts)
            len_bc = get_min_bit_length(max_count)
            bc_bits = unpack_bits(base_counts)[:, -len_bc:]
            bc_bits_main = np.hstack(
                (carry_over, elias_gamma_encode(len_bc), bc_bits.ravel())
            )
            bc_bits_analytics = np.hstack(
                (carry_over_analytics, elias_gamma_encode(len_bc), bc_bits_main.ravel())
            )
            logger.debug("Maximum base count: %d", max_count)

            logger.debug("Base count length: %d bit(s)", len_bc)

            with open(out_path, "ab") as f:
                f.write(np.packbits(bc_bits_main))
            if export_synopsis:
                with open(out_bases, "ab") as f:
                    f.write(np.packbits(bc_bits_analytics))
        else:
            with open(out_path, "ab") as f:
                f.write(np.packbits(carry_over))
            if export_synopsis:
                with open(out_bases, "ab") as f:
                    f.write(np.packbits(carry_over_analytics))

        # Log basic statistics
        logger.debug(f"Parameters:  {len(params_bits) / 8:14.0f} bytes")

        logger.debug(f"Bases:       {len(base_bits) / 8:14.0f} bytes")

        logger.debug(f"Base ids:    {len(base_ids_bits.ravel()) / 8:14.0f} bytes")

        logger.debug(f"Deviations:  {len(deviation_bits) / 8:14.0f} bytes")

        if export_counts and (gd_version != "gd-entro"):
            logger.debug(f"Base counts: {len(bc_bits) / 8:14.0f} bytes")


        return config_time

    @staticmethod
    def load_parameters(compressed):
        """Load GD parameters given the compressed data as a bytes array. Also returns
        the index just after the parameters in the array."""

        params = {}
        byte_values_16 = 1 << (8 * np.arange(2, dtype=np.int32)[::-1])
        byte_values_32 = 1 << (8 * np.arange(4, dtype=np.int32)[::-1])
        byte_values_64 = 1 << (8 * np.arange(8, dtype=np.int64)[::-1])

        # Global parameters
        # NOTE: Base counts are ignored since they are not required for decompression.
        params["d"] = compressed[0:2].dot(byte_values_16)
        params["n_b"] = compressed[3:7].dot(byte_values_32)
        params["n"] = compressed[7:15].dot(byte_values_64)
        params["drop_trailing_base_bits"] = compressed[15]
        params["drop_trailing_dev_bits"] = compressed[16]
        start_idx = 17

        # Column-level parameters
        params["m"] = [None] * params["d"]
        params["dev_bits"] = []
        params["data_types"] = {}
        params["minimum_values"] = {}
        params["na_indicators"] = []
        for i in range(params["d"]):
            params["data_types"][i] = {
                "type": PARAMS_TYPE_LOOKUP_REVERSE[compressed[start_idx]],
                "precision": compressed[start_idx + 1],
            }
            params["m"][i] = compressed[start_idx + 2]
            params["na_indicators"].append(compressed[start_idx + 3])

            m_bytes = int(params["m"][i] / 8)
            dev_bits_dtype = eval(f"np.uint{params['m'][i]}")
            db = compressed[start_idx + 4 : start_idx + 4 + m_bytes]
            if m_bytes == 8:
                dev_bits_i = dev_bits_dtype(db.dot(byte_values_64))
            else:
                dev_bits_i = dev_bits_dtype(db.dot(1 << (8 * np.arange(m_bytes)[::-1])))
            params["dev_bits"].append(dev_bits_i)

            mv = compressed[start_idx + 4 + m_bytes : start_idx + 4 + m_bytes + 8]
            params["minimum_values"][i] = mv.dot(byte_values_64)
            params["minimum_values"][i] = np.int64(params["minimum_values"][i])

            start_idx += 4 + m_bytes + 8
            # 4 bytes for data type, data precision, m and missing value indicator
            # m_bytes for deviation bits
            # 8 bytes for minimum values

        # Categorical maps
        params["categorical_maps"] = {}
        while compressed[start_idx] != ord(STRING_TERMINATION_CHAR):
            col_idx = compressed[start_idx]
            if col_idx >= params["d"]:
                raise ValueError("Invalid data")
            params["categorical_maps"][col_idx] = GD._bytes_to_strings(
                compressed[start_idx + 1 :]
            )
            length_strings = np.sum([len(s) for s in params["categorical_maps"][col_idx]])
            length_padding = len(params["categorical_maps"][col_idx]) - 1
            start_idx += length_strings + length_padding + 2
        start_idx += 1

        # Column names
        params["column_names"] = GD._bytes_to_strings(compressed[start_idx:])
        start_idx += np.sum([len(s) for s in params["column_names"]]) + len(
            params["column_names"]
        )

        return params, start_idx

    @staticmethod
    def decompress(filepath, export=False, out_path=None):
        """
        Decompress data from a file.

        Args:
            filepath (str): Path to compressed data.
            export (bool, optional): Whether to save the decompressed data to a file.
                Defaults to False.
            out_path (str, optional): Full filepath for where to store decompressed
                file. Defaults to {filepath}_decompressed.csv

        Returns:
            DataFrame: Decompressed data.
        """
        with open(filepath, "rb") as f:
            compressed = np.fromfile(f, dtype=np.uint8)
        unpacked = np.unpackbits(compressed)

        # Load parameters
        params, start_idx = GD.load_parameters(compressed)

        # Load base, deviation and base ID bits
        base_bit_indices = get_base_bit_indices(
            params["dev_bits"], drop_trailing_bits=params["drop_trailing_base_bits"]
        )
        dev_bit_indices = get_deviation_bit_indices(
            params["dev_bits"], drop_trailing_bits=params["drop_trailing_dev_bits"]
        )
        len_b = len(base_bit_indices)
        len_d = len(dev_bit_indices)
        bases_start = start_idx * 8
        bases_end = start_idx * 8 + params["n_b"] * len_b
        base_bits = unpacked[bases_start:bases_end].reshape(params["n_b"], len_b)
        len_id = elias_gamma_decode(unpacked[bases_end:])
        ids_start = bases_end + int(2 * np.floor(np.log2(len_id))) + 1
        ids_end = ids_start + params["n"] * len_id
        base_id_bits = unpacked[ids_start:ids_end].reshape(params["n"], len_id)
        devs_end = ids_end + params["n"] * len_d
        deviation_bits = unpacked[ids_end:devs_end].reshape(params["n"], len_d)

        # Re-duplicate bases & combine with deviations
        base_ids = base_id_bits.dot(1 << np.arange(len_id)[::-1])
        bases_bits_reduplicated = np.array([base_bits[id] for id in base_ids])
        data = np.zeros((params["n"], np.sum(params["m"])), dtype=np.uint8)
        data[:, base_bit_indices] = bases_bits_reduplicated
        data[:, dev_bit_indices] = deviation_bits
        data = pack_bits(data, params["m"])

        # Convert back to the original data types
        data = preprocess_revert(
            data,
            params["data_types"],
            params["minimum_values"],
            params["categorical_maps"],
            params["na_indicators"],
            params["column_names"],
        )

        # TODO add compensation for dropped base/deviation bits. That is, estimate each
        # data point by taking the midpoint of the min and max possible values given the
        # omitted base/deviation bits.

        # Export
        if export:
            if out_path is None:
                basename = os.path.splitext(os.path.basename(filepath))[0]
                out_path = os.path.dirname(filepath)
                out_path = os.path.join(out_path, f"{basename}_decompressed.csv")
            data.to_csv(out_path, index=False)

        return data

    @staticmethod
    def load_bases(filepath, revert_preprocessing=True):
        """
        Load only the GD bases from the compressed data.

        Args:
            filepath (str): Path to the compressed data (or the bases/synopsis).
            revert_preprocessing (bool, optional): Whether to revert preprocessing when
                returning the bases. Defaults to True.

        Returns:
            array: GD bases.
        """
        # Memoery map the compressed file into a binary array
        compressed = np.memmap(filepath, dtype=np.uint8, mode="r")

        # Load parameters
        params, start_idx = GD.load_parameters(compressed)

        # Load bases
        base_bit_indices = get_base_bit_indices(
            params["dev_bits"], drop_trailing_bits=params["drop_trailing_base_bits"]
        )
        len_b = len(base_bit_indices)
        end_idx = start_idx + ceil(params["n_b"] * len_b / 8)
        base_bits = compressed[start_idx:end_idx]
        base_bits = np.unpackbits(base_bits)
        base_bits = base_bits[: params["n_b"] * len_b].reshape(params["n_b"], len_b)
        bases = np.zeros((params["n_b"], np.sum(params["m"])), dtype=np.uint8)
        bases[:, base_bit_indices] = base_bits
        bases = pack_bits(bases, params["m"])

        # Convert back to the original data types
        if revert_preprocessing:
            bases = preprocess_revert(
                bases,
                params["data_types"],
                params["minimum_values"],
                params["categorical_maps"],
                params["na_indicators"],
                params["column_names"],
            )

        # TODO base counts
        # if params["include_base_counts"]:
        #     len_bc = elias_gamma_decode(base_counts)
        #     start_index += n_b * len_bc

        return bases, len_b

    @staticmethod
    def load_base_ids(filepath):
        """Load base ids from a compressed (serialised) file."""

        # Memory map the compressed file into a binary array
        compressed = np.memmap(filepath, dtype=np.uint8, mode="r")

        # Calculate base id length
        params, bases_start_idx = GD.load_parameters(compressed)
        len_b = len(
            get_base_bit_indices(
                params["dev_bits"], drop_trailing_bits=params["drop_trailing_base_bits"]
            )
        )
        bases_end_idx = bases_start_idx + floor(params["n_b"] * len_b / 8)
        bases_end_carryover = params["n_b"] * len_b % 8
        buffer = 100  # unpack extra bytes to ensure len_id encoding is captured
        len_id_bits = np.unpackbits(compressed[bases_end_idx : bases_end_idx + buffer])
        len_id = elias_gamma_decode(len_id_bits[bases_end_carryover:])

        # Extract base ids from the serialised data
        size_len_id = int(2 * np.floor(np.log2(len_id))) + 1
        size_ids = len_id * params["n"]
        base_ids_bits = np.unpackbits(
            compressed[bases_end_idx : bases_end_idx + size_ids // 8 + buffer]
        )[bases_end_carryover + size_len_id :][:size_ids]
        base_ids = np.zeros((params["n"], 32), dtype=np.uint8)
        base_ids[:, 32 - len_id :] = base_ids_bits.reshape(params["n"], len_id)
        base_ids = pack_bits(base_ids, 32)

        return base_ids[0].values

    @staticmethod
    def load_deviations(filepath, revert_preprocessing=True, dev_max=False):
        """Load deviations from a compressed (serialised) file."""

        # Memory map the compressed file into a binary array
        compressed = np.memmap(filepath, dtype=np.uint8, mode="r")

        # Load parameters
        params, bases_start_idx = GD.load_parameters(compressed)
        dev_bits_indices = get_deviation_bit_indices(
            params["dev_bits"], drop_trailing_bits=params["drop_trailing_dev_bits"]
        )
        len_d = len(dev_bits_indices)
        len_b = np.sum(params["m"]).astype(int) - len_d

        # Calculate start and end index of deviations
        size_bases = len_b * params["n_b"]
        bases_end_idx = bases_start_idx + floor(size_bases / 8)
        bases_end_carryover = params["n_b"] * len_b % 8
        buffer = 100  # unpack extra bytes to ensure len_id encoding is captured
        len_id_bits = np.unpackbits(
            compressed[bases_end_idx : bases_end_idx + buffer]
        )[bases_end_carryover:]
        len_id = elias_gamma_decode(len_id_bits)
        size_len_id = int(2 * np.floor(np.log2(len_id))) + 1
        size_ids = len_id * params["n"]
        devs_start = bases_start_idx + floor((size_bases + size_len_id + size_ids) / 8)
        size_devs = len_d * params["n"]
        devs_end = bases_start_idx + ceil(
            (size_bases + size_len_id + size_ids + size_devs) / 8
        )

        # Load deviations
        deviations_bits = compressed[devs_start:devs_end]
        deviations_bits = np.unpackbits(deviations_bits)
        overlap = (size_bases + size_len_id + size_ids) % 8
        deviations_bits = deviations_bits[overlap : overlap + size_devs]
        deviations = np.zeros((params["n"], np.sum(params["m"])), dtype=np.uint8)
        deviations[:, dev_bits_indices] = deviations_bits.reshape(params["n"], len_d)
        deviations = pack_bits(deviations, params["m"])

        # Convert back to the original data types
        if revert_preprocessing:
            deviations = preprocess_revert(
                deviations,
                params["data_types"],
                {i: 0 for i in range(deviations.shape[1])},
                params["categorical_maps"],
                params["na_indicators"],
                params["column_names"],
            )
        
        if dev_max:
            deviations_max = np.zeros((1, np.sum(params["m"])), dtype=np.uint8)
            deviations_max[:, dev_bits_indices] = np.ones((1, len_d), dtype=np.uint8)
            deviations_max = pack_bits(deviations_max, params["m"])
        
        if revert_preprocessing and dev_max:
            deviations_max = preprocess_revert(
                deviations_max,
                params["data_types"],
                {i: 0 for i in range(deviations_max.shape[1])},
                params["categorical_maps"],
                params["na_indicators"],
                params["column_names"],
            )
            return deviations, deviations_max
        
        else:
            return deviations
