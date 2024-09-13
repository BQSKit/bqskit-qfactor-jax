# QFactor and QFactor-Sample Implementations on GPUs Using JAX
`bqskit-qfactor-jax` is a Python package that implements circuit instantiation using the [QFactor](https://ieeexplore.ieee.org/abstract/document/10313638) and [QFactor-Sample](https://arxiv.org/abs/2405.12866) algorithms on GPUs to accelerate [BQSKit](https://github.com/bqskit/bqskit). It uses [JAX](https://jax.readthedocs.io/en/latest/index.html) as an abstraction layer of the GPUs, seamlessly utilizing JIT compilation and GPU parallelism.

## Installation
`bqskit-qfactor-jax` is available for Python 3.9+ on Linux. It can be installed using pip

```sh
pip install bqskit-qfactor-jax
```

If you are experiencing issues with JAX please refer to JAX's [installation instructions](https://github.com/google/jax#installation).


## Basic Usage
QFactor and QFactor-Sample are instantiation algorithms that, given a unitary matrix and a parameterized circuit, optimize the circuit parameters to best approximate the target unitary matrix.

```python
import numpy as np
from bqskit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix

from qfactorjax.qfactor_sample_jax import QFactorSampleJax



# Load a circuit from QASM
circuit = Circuit.from_file("template.qasm")

# Load the target unitary
unitary_target = UnitaryMatrix.from_file("target.mat")

# Create the instantiator object
qfactor_sample_gpu_instantiator = QFactorSampleJax()

# Perform the instantiation
circuit.instantiate(
        unitary_target,
        multistarts=16,
        method=qfactor_sample_gpu_instantiator,
    )

# Calculate and print final distance
dist = circuit.get_unitary().get_distance_from(unitary_target, 1)

print('Final Distance: ', dist)
```

Please look at the [examples](https://github.com/BQSKit/bqskit-qfactor-jax/tree/main/examples) for a more detailed usage, especially at performance comparison between QFactor and QFactor-Sample.


## GPU Configuration and Memory Management
Please set the environment variable XLA_PYTHON_CLIENT_PREALLOCATE=False when using this package. Also, if you encounter OOM issues consider setting XLA_PYTHON_CLIENT_ALLOCATOR=platform.


When using several workers on the same GPU, we recommend using [Nvidia's MPS](https://docs.nvidia.com/deploy/mps/index.html). You may initiate it using the command line
```sh
nvidia-cuda-mps-control -d
```

You can disable it by running this command line:
```sh
echo quit | nvidia-cuda-mps-control
```

## References
If you are using QFactor please cite:\
Kukliansky, Alon, et al. "QFactor: A Domain-Specific Optimizer for Quantum Circuit Instantiation." 2023 IEEE International Conference on Quantum Computing and Engineering (QCE). Vol. 1. IEEE, 2023. [Link](https://ieeexplore.ieee.org/abstract/document/10313638).

If you are using QFactor-Sample please cite:\
Kukliansky, Alon, et al. "Leveraging Quantum Machine Learning Generalization to Significantly Speed-up Quantum Compilation" arXiv preprint [arXiv:2405.12866](https://arxiv.org/abs/2405.12866) (2024).

## License
The software in this repository is licensed under a **BSD free software
license** and can be used in source or binary form for any purpose as long
as the simple licensing requirements are followed. See the
**[LICENSE](https://github.com/BQSKit/bqskit-qfactor-jax/blob/main/LICENSE)** file
for more information.

## Copyright

Quantum Fast Circuit Optimizer (QFactor) JAX implementation Copyright (c) 2024,
U.S. Federal Government and the Government of Israel. All rights reserved.
