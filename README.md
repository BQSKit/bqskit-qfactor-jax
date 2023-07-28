# GPU support for BQSKit
`bqskitgpu` is a Python package that implements circuit instantiation and simulation on GPU to accelerate [BQSKit](https://github.com/bqsKit/bqskit), it uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as an abstraction layer of the GPUs, seamlessly utilizing JIT and parralizim

## Installation
`bqskitgpu` is available for Python 3.8+ on Linux, macOS.

First, install JAX with GPU support, you may refer to JAX's [installation instructions](https://github.com/google/jax#installation).

Next, install the package using pip

```sh
pip install bqskitgpu
```

# Runnig bqskitgpu
Please set the environment variable XLA_PYTHON_CLIENT_PREALLOCATE=False when using this package

Rferes to the examples to see some basic usage.

## License
