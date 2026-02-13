import os

import numpy as np

# General
NAME_DELIMITER = "-"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# Directories
TEST_DATA_DIR = os.path.join("tests", "test_data")
TEMP_DATA_DIR = os.path.join("tests", "temp_data")

# Data types
N_EXPONENT_BITS_LOOKUP = {32: 8, 64: 12}
INT_LOOKUP = {16: np.int16, 32: np.int32, 64: np.int64}
UINT_LOOKUP = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
FLOAT_LOOKUP = {32: np.float32, 64: np.float64}
BUFFER_LOOKUP = {
    "int8": ">u1",
    "int16": ">u2",
    "int32": ">u4",
    "int64": ">u8",
    "uint8": ">u1",
    "uint16": ">u2",
    "uint32": ">u4",
    "uint64": ">u8",
}
