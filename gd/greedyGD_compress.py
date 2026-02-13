import logging
import datetime
import os
from time import perf_counter

import numpy as np
import pandas as pd

from gd.config import LOG_FORMAT
from gd.GD import GD
from gd.data_loader import load_dataset

# GD_VERSION = "gd-greedy"
# GD_VERSION = "gd-entro"
GD_TRAINING_SIZE = 1000000
CHECK_DECOMPRESSION = True
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# FILEPATH = "tests/test_data/uncompressed/california-housing-train.csv"
# SCHEMA = {
#     "MedInc": {"type": np.float64, "precision": 4},
#     "HouseAge": {"type": np.float64, "precision": 1},
#     "AveRooms": {"type": np.float64, "precision": 6},
#     "AveBedrms": {"type": np.float64, "precision": 6},
#     "Population": {"type": np.float64, "precision": 1},
#     "AveOccup": {"type": np.float64, "precision": 6},
#     "Latitude": {"type": np.float64, "precision": 2},
#     "Longitude": {"type": np.float64, "precision": 2},
#     "MedHouseVal": {"type": np.float64, "precision": 5},
# }

# FILEPATH = "tests/test_data/uncompressed/aarhus-citylab.csv"
# SCHEMA = {
#     "light": {"type": np.float32, "precision": 2},
#     "humidity": {"type": np.float32, "precision": 2},
#     "noise": {"type": np.float32, "precision": 2},
#     "temperature": {"type": np.float32, "precision": 2},
# }
# FILEPATH = "tests/test_data/uncompressed/winequality-red.csv"
# SCHEMA = {
#     "fixed_acidity": {"type": np.float32, "precision": 1},
#     "volatile_acidity": {"type": np.float32, "precision": 3},
#     "citric_acid": {"type": np.float32, "precision": 2},
#     "residual_sugar": {"type": np.float32, "precision": 2},
#     "chlorides": {"type": np.float32, "precision": 3},
#     "free_sulfur_dioxide": {"type": np.float32, "precision": 1},
#     "total_sulfur_dioxide": {"type": np.float32, "precision": 1},
#     "density": {"type": np.float32, "precision": 5},
#     "pH": {"type": np.float32, "precision": 2},
#     "sulphates": {"type": np.float32, "precision": 2},
#     "alcohol": {"type": np.float32, "precision": 2},
#     "quality": {"type": np.uint8, "precision": 0}
# }
# FILEPATH = "tests/test_data/uncompressed/uci-household_power_consumption_10k.csv"
# SCHEMA = {
#     "global_active_power": {"type": np.float32, "precision": 3},
#     "global_reactive_power": {"type": np.float32, "precision": 3},
#     "voltage": {"type": np.float32, "precision": 2},
#     "global_intensity": {"type": np.float32, "precision": 1},
#     "sub_metering_1": {"type": np.float32, "precision": 1},
#     "sub_metering_2": {"type": np.float32, "precision": 1},
#     "sub_metering_3": {"type": np.float32, "precision": 1},
# }
# FILEPATH = "tests/test_data/uncompressed/kristian_many_dimensions.csv"
# SCHEMA_r = {f"r{i}": {"type": np.float32, "precision": 4} for i in range(1, 201)}
# SCHEMA_i = {f"i{i}": {"type": np.float32, "precision": 4} for i in range(1, 201)}
# SCHEMA = {**SCHEMA_r, **SCHEMA_i}


def setup_logging(log_filepath, file_level=logging.INFO, console_level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    logging.basicConfig(
        level=logging.DEBUG, format=LOG_FORMAT, handlers=[file_handler, console_handler]
    )
def configure_logging(logger, level=logging.WARNING):
    """Configure logging level for the GD module."""
    logging.basicConfig(level=level)
    logger.setLevel(level)
    
def greedyGD_compress(FILEPATH, SCHEMA, entropy=None, GD_VERSION="gd-greedy", tau=10, condensed_sample=0, max_condensed_weight=1):

    data_name = os.path.splitext(os.path.basename(FILEPATH))[0]
    # Setup logging

    log_filepath = f"gd_compressed_data/{data_name}_{GD_VERSION}_{TIMESTAMP}.log"
    setup_logging(log_filepath, console_level=logging.INFO, file_level=logging.DEBUG)
    logger = logging.getLogger()
    configure_logging(logger, level=logging.WARNING) # Set to DEBUG for detailed logs, INFO for general info, WARNING to reduce verbosity

    # Compute dataset size (without loading the file into memory)
    with open(FILEPATH, "r") as f:
        n = sum(1 for _ in f) - 1
    b = sum([np.dtype(SCHEMA[c]["type"]).itemsize for c in SCHEMA])
    logger.info(f"Total rows:    {n:,}")
    logger.info(f"Bytes per row: {b}")
    size_uncompressed = n * b

    # Compress
    out_filepath = f"gd_compressed_data/{data_name}_out.gd"
    out_filepath_bases = f"gd_compressed_data/{data_name}_out_bases.gd"
    logger.info("Starting compression...")
    t_start = perf_counter()
    config_time = GD.compress(
        data=FILEPATH,
        schema=SCHEMA,
        entropy=entropy,
        has_header=True,
        export_counts=True,
        export_synopsis=True,
        gd_version=GD_VERSION,
        config_sample_size=GD_TRAINING_SIZE,
        # drop_trailing_base_bits=0,
        # drop_trailing_dev_bits=0,
        tau=tau,
    )
    t_compression = perf_counter() - t_start
    logger.info(f"{GD_VERSION} compression finished in {t_compression} seconds.")

    # Check that decompression is lossless
    if CHECK_DECOMPRESSION:
        logger.info("Checking that decompression is lossless...")
        data = pd.concat(load_dataset(FILEPATH, SCHEMA))
        decompressed_data = GD.decompress(filepath=out_filepath)
        logger.info(f"Lossless compression: {data.equals(decompressed_data)}"),
        if not data.equals(decompressed_data):
            logger.error(f"Difference:\n{data.compare(decompressed_data)}")

    # Report statistics
    size_compressed = os.path.getsize(out_filepath)
    size_bases = os.path.getsize(out_filepath_bases)
    if condensed_sample > 0 and GD_VERSION == "gd-entro":
        size_condensed_weights = condensed_sample * (np.ceil(np.ceil(np.log2(max_condensed_weight)) / 8))
        size_compressed += size_condensed_weights
    cr = size_compressed / size_uncompressed
    cr_bases = size_bases / size_uncompressed
    logger.info(f"Uncompressed size:            {size_uncompressed} bytes")
    logger.info(f"Compressed size:              {size_compressed} bytes")
    logger.info(f"Compression ratio:            {cr:.3f}")
    cr_wo_condensed = cr
    if condensed_sample > 0 and GD_VERSION == "gd-entro":
        logger.info(f"  (including {condensed_sample} condensed samples with max weight {max_condensed_weight}, size {size_condensed_weights} bytes)")
        size_uncompressed_wo_condensed = (n - condensed_sample) * b
        cr_wo_condensed = size_compressed / size_uncompressed_wo_condensed
        logger.info(f"  (excluding condensed samples, cr {cr_wo_condensed:.3f})")
    logger.info(f"Compression ratio (inverse):  {1 / cr:.3f}")

    # Export metadata
    logger.info("Saving metadata...")
    metadata_filepath = f"gd_compressed_data/out_{TIMESTAMP}_gd_metadata.txt"
    with open(metadata_filepath, "w") as f:
        f.write("------------- Parameters -------------\n")
        f.write(f"DATA                      {FILEPATH}\n")
        f.write(f"GD_VERSION                {GD_VERSION}\n")
        f.write(f"GD_TRAINING_SIZE          {GD_TRAINING_SIZE}\n")
        # f.write(f"GD_BATCH_SIZE                 {GD_BATCH_SIZE}\n")
        # f.write(f"GD_EXPLORATION_FACTOR         {GD_EXPLORATION_FACTOR}\n")
        # f.write(f"GD_BALANCING_FACTOR           {GD_BALANCING_FACTOR}\n")
        # f.write(f"GD_BASE_COUNTS                {GD_BASE_COUNTS}\n")

        f.write("\n------------- Runtime -------------\n")
        f.write(f"Compression (total)              {t_compression:.3f} s\n")

        f.write("\n------------- Storage -------------\n")
        f.write(f"Uncompressed size:        {size_uncompressed} (bytes)\n")
        f.write(f"Compressed size (total):  {size_compressed} (bytes)\n")
        f.write(f"Compressed size (bases):  {size_bases} (bytes)\n")
        f.write(f"CR:                       {cr:.3f}\n")
        f.write(f"CR (inverse):             {1 / cr:.3f}\n")
        f.write(f"CR (bases):               {cr_bases:.3f}\n")
        f.write(f"CR (bases, inverse):      {1 / cr_bases:.3f}\n")

    logger.info("Done.")

    return cr, cr_wo_condensed, config_time
