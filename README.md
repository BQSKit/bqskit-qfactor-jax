# QFactor implementation on GPUs using JAX
`bqskit-qfactor-jax` is a Python package that implements circuit instantiation and simulation on GPU to accelerate [BQSKit](https://github.com/bqsKit/bqskit), it uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as an abstraction layer of the GPUs, seamlessly utilizing JIT compilation and GPU parallelism.

## Installation
`bqskit-qfactor-jax` is available for Python 3.8+ on Linux, macOS.

First, install JAX with GPU support, you may refer to JAX's [installation instructions](https://github.com/google/jax#installation).

Next, install the package using pip

```sh
pip install bqskit-qfactor-jax
```

# Running bqskit-qfactor-jax
Please set the environment variable XLA_PYTHON_CLIENT_PREALLOCATE=False when using this package.

Please take a look at the [examples](https://github.com/BQSKit/bqskit-qfactor-jax/tree/main/examples) to see some basic usage.

When using several workers on the same GPU, we recommend using [Nvidia's MPS](https://docs.nvidia.com/deploy/mps/index.html). You may initiate it using the command line
```sh
nvidia-cuda-mps-control -d
```

You can disable it by running this command line:
```sh
echo quit | nvidia-cuda-mps-control
```

# References
Kukliansky, Alon, et al. "QFactor:A Domain-Specific Optimizer for Quantum Circuit Instantiation." arXiv preprint [arXiv:2306.08152](https://arxiv.org/abs/2306.08152) (2023).

## License
The software in this repository is licensed under a **BSD free software
license** and can be used in source or binary form for any purpose as long
as the simple licensing requirements are followed. See the
**[LICENSE](https://github.com/BQSKit/bqskit-qfactor-jax/blob/main/LICENSE)** file
for more information.

## Copyright

Quantum Fast Circuit Optimizer (QFactor) JAX implementation Copyright (c) 2023,
The Regents of the University of California, through Lawrence
Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy) and Alon Kukliansky.  All rights reserved.