"""Original author: Rasmus Vestergaard"""

import numpy as np


def elias_gamma_encode(x):
    """
    Codes x with Elias Gamma Coding.

    https://en.wikipedia.org/wiki/Elias_gamma_coding
    """
    if x < 0:
        print(x)
    assert x > 0
    N = np.floor(np.log2(x)).astype(int)
    out = "0" * N + format(x, "b")

    return np.array(list(out), dtype=int)


def elias_gamma_decode(y):
    """
    Decodes y from Elias Gamma Coding.

    https://en.wikipedia.org/wiki/Elias_gamma_coding
    """

    # Count initial number of zeros.
    n = 0
    while y[n] == 0:
        n += 1

    # Read next n bit integer.
    out = 0
    for bit in y[n : 2 * n + 1]:
        out = (out << 1) | bit

    return out


if __name__ == "__main__":
    for x in np.arange(1, 1000):
        y = elias_gamma_encode(x)
        z = elias_gamma_decode(y)
        assert x == z
    print("Elias Gamma test passed.")
