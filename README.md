# entroGD

Implementation of Entropy-guided Generalized Deduplication (entroGD) for data compression, as proposed in:

> [EntroGD: Scalable Generalized Deduplication for Efficient Direct Analytics on Compressed IoT Data](https://arxiv.org/abs/2511.04148)

## Example

An example of entroGD is given in [example.py](example.py). It demonstrates:

1. Loading and preprocessing a dataset into binary representation
2. Running `entroGD_cluster` to generate condensed samples
3. Compressing the data using the entropy-guided GD compressor
