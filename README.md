# GPU support for BQSKit
`bqskitgpu` is a Python package that implements circuit instantiation and simulation on GPU to accelerate [BQSKit](https://github.com/bqsKit/bqskit), it uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as an abstraction layer of the GPUs, seamlessly utilizing JIT compilation and GPU parallelism.

## Installation
`bqskitgpu` is available for Python 3.8+ on Linux, macOS.

First, install JAX with GPU support, you may refer to JAX's [installation instructions](https://github.com/google/jax#installation).

Next, install the package using pip

```sh
pip install bqskitgpu
```

# Runnig bqskitgpu
Please set the environment variable XLA_PYTHON_CLIENT_PREALLOCATE=False when using this package.

Please take a look at the examples to see some basic usage.

When using several workers on the same GPU, we recommend using [Nvidia's MPS](https://docs.nvidia.com/deploy/mps/index.html). You may initiate it using the command line
```sh
nvidia-cuda-mps-control -d
```

You can disable it by running this command line:
```sh
echo quit | nvidia-cuda-mps-control
```

# References
Kukliansky, Alon, et al. "QFactor--A Domain-Specific Optimizer for Quantum Circuit Instantiation." arXiv preprint [arXiv:2306.08152](https://arxiv.org/abs/2306.08152) (2023).

## License
